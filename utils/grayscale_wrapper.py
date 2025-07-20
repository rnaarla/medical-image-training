"""Backward compatibility import for grayscale_wrapper.py"""
import warnings
warnings.warn("Importing from root is deprecated. Use new structure: from src.core.grayscale_wrapper import *", DeprecationWarning)
from src.core.grayscale_wrapper import *
