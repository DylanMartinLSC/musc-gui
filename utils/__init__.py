"""
Utility modules for MuSc GUI
"""

from .hardware_detector import (
    get_gpu_info,
    get_recommended_device,
    get_recommended_model,
    check_model_compatibility
)

from .model_catalog import (
    get_models_by_category,
    get_model_display_name,
    get_model_info,
    get_recommended_model as get_catalog_recommended_model,
    format_model_details,
    get_category_display_name,
    MODEL_INFO,
    MODEL_CATEGORIES
)

__all__ = [
    'get_gpu_info',
    'get_recommended_device',
    'get_recommended_model',
    'check_model_compatibility',
    'get_models_by_category',
    'get_model_display_name',
    'get_model_info',
    'format_model_details',
    'get_category_display_name',
    'MODEL_INFO',
    'MODEL_CATEGORIES'
]
