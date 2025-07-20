"""Backward compatibility import for medical_data_pipeline.py"""
import warnings
warnings.warn("Importing from root is deprecated. Use new structure: from src.medical.medical_data_pipeline import *", DeprecationWarning)
from src.medical.medical_data_pipeline import *
