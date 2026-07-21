class ModelNotReadyError(Exception):
    """Raised when inference is requested before the model finished loading."""


class ModelLoadError(Exception):
    """Raised when the model cannot be loaded."""


class TextValidationError(Exception):
    """Raised when input text fails validation (length/empty/etc)."""