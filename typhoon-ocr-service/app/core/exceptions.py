class TyphoonOCRError(Exception):
    """Base exception for typhoon-ocr-service"""

    pass


class ModelLoadError(TyphoonOCRError):
    """Raised when loading model fails"""

    pass


class OCRProcessingError(TyphoonOCRError):
    """Raised when processing image or generating text fails"""

    pass


class OCRTimeoutError(TyphoonOCRError):
    """Raised when inference exceeds specified timeout"""

    pass


class InvalidImageError(TyphoonOCRError):
    """Raised when provided file is not a valid image"""

    pass
