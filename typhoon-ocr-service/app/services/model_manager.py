import asyncio
import os
import random
import threading
import time

from typing import AsyncGenerator, Dict, Any, Optional
from PIL import Image

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

from app.core.config import get_settings
from app.core.exceptions import ModelLoadError, OCRProcessingError, OCRTimeoutError
from app.core.logging import get_logger

logger = get_logger("typhoon-ocr.model_manager")

DEFAULT_TYPHOON_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


class ModelManager:
    """
    Singleton Manager for Typhoon OCR models.
    Supports dynamic model switching via MODEL_NAME in .env, precision dtype,
    thread-safe inference, retries, timeouts, seeds, and SSE streaming.
    """

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.model_name = ""
        self._lock = threading.Lock()
        self._is_loaded = False
        self.is_loading = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    async def load_in_background(self) -> None:
        if self._is_loaded or self.is_loading:
            return
        self.is_loading = True
        logger.info("Background model loading task started for '%s'...", get_settings().MODEL_NAME)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.load)
        except Exception as exc:
            logger.error("Background model load failed: %s", exc)
        finally:
            self.is_loading = False

    def _resolve_device(self, requested_device: str) -> str:
        req = (requested_device or "auto").lower().strip()
        if req == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            return "cpu"
        if req == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _resolve_dtype(self, dtype_str: str, device: str) -> torch.dtype | str:
        ds = (dtype_str or "auto").lower().strip()
        if ds == "float16":
            return torch.float16
        if ds == "bfloat16":
            return torch.bfloat16
        if ds == "float32":
            return torch.float32
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load(self) -> None:
        settings = get_settings()
        if self._is_loaded:
            logger.info("Model %s is already loaded.", self.model_name)
            return

        self.model_name = settings.MODEL_NAME
        self.device = self._resolve_device(settings.DEVICE)
        torch_dtype = self._resolve_dtype(settings.MODEL_DTYPE, self.device)

        logger.info(
            "Loading Typhoon OCR model '%s' (device=%s, dtype=%s)...",
            self.model_name,
            self.device,
            torch_dtype,
        )

        try:
            os.environ["HF_HOME"] = settings.HF_HOME
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            logger.info("[Step 1/2] Checking/downloading model weights for '%s'...", self.model_name)
            local_model_path = self.model_name
            try:
                from huggingface_hub import snapshot_download
                local_model_path = snapshot_download(repo_id=self.model_name)
                logger.info("[Step 1/2 Complete] Model files verified/cached at '%s'.", local_model_path)
            except Exception as dl_err:
                logger.warning("Download check notice: %s. Proceeding with repository name...", dl_err)

            logger.info("[Step 2/2] Loading PyTorch tensors into %s memory...", self.device)
            self.processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                local_model_path,
                dtype=torch_dtype if isinstance(torch_dtype, torch.dtype) else "auto",
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device == "cpu" and self.model is not None:
                self.model.to("cpu")

            if hasattr(self.model, "eval"):
                self.model.eval()

            self._is_loaded = True
            logger.info("Successfully loaded model '%s' on %s.", self.model_name, self.device)
        except Exception as exc:
            self._is_loaded = False
            logger.error("Failed to load model '%s': %s", self.model_name, exc, exc_info=True)
            raise ModelLoadError(f"Could not load model '{self.model_name}': {exc}") from exc

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.processor = None
            self._is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded model manager.")

    def _prepare_inputs(self, image: Image.Image, prompt: str, system_prompt: str = ""):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}]
            })

        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt if prompt and prompt.strip() else DEFAULT_TYPHOON_PROMPT},
        ]
        messages.append({"role": "user", "content": user_content})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    def _generate_sync(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        if not self._is_loaded or self.model is None or self.processor is None:
            raise ModelLoadError("Model is not loaded.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)

        with self._lock:
            start_time = time.perf_counter()
            inputs = self._prepare_inputs(image, prompt, system_prompt)
            
            # Move inputs to target device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[0][input_len:]
            
            output_text = self.processor.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000.0

            prompt_tokens = int(input_len)
            completion_tokens = int(len(generated_tokens))
            total_tokens = prompt_tokens + completion_tokens

            return {
                "text": output_text,
                "model_used": self.model_name,
                "processing_time_ms": round(processing_time_ms, 2),
                "device_used": self.device,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

    async def predict_with_retry(
        self,
        image: Image.Image,
        prompt: str = "",
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Executes sync model inference on threadpool with retry loop and timeout enforcement.
        Guarantees cleanup of transient resources on failure.
        """
        last_exception = None
        attempts = max(1, max_retries)

        for attempt in range(1, attempts + 1):
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self._generate_sync,
                        image,
                        prompt,
                        system_prompt,
                        temperature,
                        max_tokens,
                        seed,
                    ),
                    timeout=timeout,
                )
                return result
            except asyncio.TimeoutError as exc:
                last_exception = OCRTimeoutError(f"Request timed out after {timeout} seconds (attempt {attempt}/{attempts})")
                logger.warning("Attempt %d/%d timed out after %ss", attempt, attempts, timeout)
            except Exception as exc:
                last_exception = OCRProcessingError(f"Inference error (attempt {attempt}/{attempts}): {exc}")
                logger.warning("Attempt %d/%d failed with error: %s", attempt, attempts, exc)

            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

        raise last_exception or OCRProcessingError("OCR processing failed after all retries.")

    async def predict_stream(
        self,
        image: Image.Image,
        prompt: str = "",
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generates streaming output chunks for Server-Sent Events (SSE).
        Guarantees thread lock release and resource cleanup via try...finally.
        """
        if not self._is_loaded or self.model is None or self.processor is None:
            raise ModelLoadError("Model is not loaded.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        inputs = self._prepare_inputs(image, prompt, system_prompt)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        def _run_stream():
            with self._lock, torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        try:
            while True:
                # Poll streamer non-blockingly
                chunk = await loop.run_in_executor(None, self._get_next_stream_chunk, streamer)
                if chunk is None:
                    break
                if chunk:
                    yield chunk
        finally:
            thread.join(timeout=2.0)

    def _get_next_stream_chunk(self, streamer: TextIteratorStreamer) -> Optional[str]:
        try:
            return next(streamer)
        except StopIteration:
            return None


model_manager = ModelManager()
