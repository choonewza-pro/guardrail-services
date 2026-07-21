class ModelNotReadyError(Exception):
    """Raised when inference is requested before the model finished loading."""


class ModelLoadError(Exception):
    """Raised when the model cannot be loaded."""


class ImageDecodeError(Exception):
    """Raised when an uploaded file cannot be decoded as an image."""


class UnsupportedMIMEError(Exception):
    """Raised when the uploaded MIME type is not allowed."""


class FileTooLargeError(Exception):
    """Raised when uploaded file exceeds the size limit."""