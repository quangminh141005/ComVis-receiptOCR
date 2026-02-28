"""
src/preprocessing/__init__.py
------------------------------
Exposes all preprocessing steps as a clean package-level API.
"""

from .grayscale    import to_grayscale, upscale_if_needed
from .denoising    import denoise
from .contrast     import enhance_contrast
from .thresholding import binarize
from .morphology   import morphological_cleanup


__all__ = [
    "to_grayscale",
    "upscale_if_needed",
    "deskew_image",
    "denoise",
    "enhance_contrast",
    "binarize",
    "morphological_cleanup",
    "crop_to_content",
]