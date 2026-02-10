"""
Model Catalog for MuSc GUI

Provides detailed information about available AI models including
performance characteristics, hardware requirements, and use cases.
"""

# Model categories for organization
MODEL_CATEGORIES = {
    'fast': 'Fast Models (Testing/Older GPUs)',
    'balanced': 'Balanced Models (Recommended)',
    'accurate': 'High Accuracy Models (Powerful GPU Required)'
}


# Comprehensive model information database
MODEL_INFO = {
    # ===== FAST MODELS =====
    'vit_tiny_patch16_224.augreg_in21k': {
        'display_name': 'Tiny - Fastest (For Testing)',
        'category': 'fast',
        'download_size': '40MB',
        'gpu_memory': '~1GB',
        'speed_estimate': '30+ FPS on older GPUs, 5-10 FPS on CPU',
        'accuracy': 'Low',
        'use_case': 'Testing, CPU mode, or resource-constrained systems',
        'recommended_image_sizes': [224, 256],
        'description': 'Smallest and fastest model. Good for testing or when GPU is not available.',
        'pros': ['Extremely fast', 'Low memory usage', 'Works on CPU'],
        'cons': ['Lower accuracy', 'May miss subtle defects'],
        'min_gpu_memory': 1.0
    },

    'dino_deitsmall16': {
        'display_name': 'Small - Fast',
        'category': 'fast',
        'download_size': '160MB',
        'gpu_memory': '~1.5GB',
        'speed_estimate': '20-30 FPS on RTX 3060',
        'accuracy': 'Medium',
        'use_case': 'Production on older GPUs or when speed is critical',
        'recommended_image_sizes': [224, 256, 512],
        'description': 'Good balance of speed and accuracy for older hardware.',
        'pros': ['Fast processing', 'Reasonable accuracy', 'Low GPU requirements'],
        'cons': ['Not as accurate as larger models'],
        'min_gpu_memory': 1.5
    },

    'vit_small_patch16_224.dino': {
        'display_name': 'Small - Alternative',
        'category': 'fast',
        'download_size': '160MB',
        'gpu_memory': '~1.5GB',
        'speed_estimate': '20-30 FPS on RTX 3060',
        'accuracy': 'Medium',
        'use_case': 'Alternative to dino_deitsmall16',
        'recommended_image_sizes': [224, 256],
        'description': 'Another fast option with similar performance to dino_deitsmall16.',
        'pros': ['Fast processing', 'Good for real-time'],
        'cons': ['Medium accuracy'],
        'min_gpu_memory': 1.5
    },

    # ===== BALANCED MODELS =====
    'dinov2_vitb14': {
        'display_name': '✓ DINOv2 Base - RECOMMENDED',
        'category': 'balanced',
        'download_size': '350MB',
        'gpu_memory': '~3GB',
        'speed_estimate': '15-20 FPS on RTX 3060',
        'accuracy': 'High',
        'use_case': 'Most production environments - best balance of speed and accuracy',
        'recommended_image_sizes': [224, 384, 512],
        'description': 'Best overall choice. Excellent accuracy with good speed on modern GPUs.',
        'pros': ['High accuracy', 'Good speed', 'Reliable', 'Industry standard'],
        'cons': ['Requires modern GPU'],
        'min_gpu_memory': 3.0,
        'recommended': True
    },

    'ViT-B-16': {
        'display_name': 'ViT Base-16 - Alternative',
        'category': 'balanced',
        'download_size': '350MB',
        'gpu_memory': '~2.5GB',
        'speed_estimate': '15-20 FPS on RTX 3060',
        'accuracy': 'High',
        'use_case': 'Alternative to DINOv2 Base',
        'recommended_image_sizes': [224, 256, 384],
        'description': 'Solid performer with good accuracy and speed.',
        'pros': ['Good accuracy', 'Balanced performance'],
        'cons': ['Not as accurate as DINOv2'],
        'min_gpu_memory': 2.5
    },

    'ViT-B-32': {
        'display_name': 'ViT Base-32',
        'category': 'balanced',
        'download_size': '350MB',
        'gpu_memory': '~2.5GB',
        'speed_estimate': '18-25 FPS on RTX 3060',
        'accuracy': 'Medium-High',
        'use_case': 'Slightly faster than ViT-B-16',
        'recommended_image_sizes': [224, 384],
        'description': 'Faster variant of ViT Base with slightly lower accuracy.',
        'pros': ['Good speed', 'Lower memory than B-16'],
        'cons': ['Slightly less accurate than B-16'],
        'min_gpu_memory': 2.5
    },

    'dino_vitbase16': {
        'display_name': 'DINO Base-16',
        'category': 'balanced',
        'download_size': '350MB',
        'gpu_memory': '~2.5GB',
        'speed_estimate': '15-20 FPS on RTX 3060',
        'accuracy': 'High',
        'use_case': 'Original DINO model - reliable choice',
        'recommended_image_sizes': [224, 256, 512],
        'description': 'Original DINO architecture, proven and reliable.',
        'pros': ['Well-tested', 'Good accuracy'],
        'cons': ['Older architecture than DINOv2'],
        'min_gpu_memory': 2.5
    },

    'dino_vitbase8': {
        'display_name': 'DINO Base-8',
        'category': 'balanced',
        'download_size': '350MB',
        'gpu_memory': '~2.5GB',
        'speed_estimate': '12-18 FPS on RTX 3060',
        'accuracy': 'High',
        'use_case': 'Higher resolution features than base16',
        'recommended_image_sizes': [224, 256],
        'description': 'Captures finer details but slower than base16.',
        'pros': ['Fine detail detection', 'Good for small defects'],
        'cons': ['Slower than base16'],
        'min_gpu_memory': 2.5
    },

    # ===== HIGH ACCURACY MODELS =====
    'dinov2_vitl14': {
        'display_name': 'DINOv2 Large - Best Accuracy',
        'category': 'accurate',
        'download_size': '1.1GB',
        'gpu_memory': '~5GB',
        'speed_estimate': '8-12 FPS on RTX 3060',
        'accuracy': 'Very High',
        'use_case': 'Critical inspection where accuracy is paramount',
        'recommended_image_sizes': [224, 512],
        'description': 'Highest accuracy model. Use when detecting subtle defects is critical.',
        'pros': ['Excellent accuracy', 'Best for subtle defects', 'Industry-leading'],
        'cons': ['Slower', 'High GPU memory requirement', 'Large download'],
        'min_gpu_memory': 5.0
    },

    'ViT-L-14': {
        'display_name': 'ViT Large-14',
        'category': 'accurate',
        'download_size': '1GB',
        'gpu_memory': '~4.5GB',
        'speed_estimate': '8-12 FPS on RTX 3060',
        'accuracy': 'Very High',
        'use_case': 'High-precision applications',
        'recommended_image_sizes': [224, 384, 512],
        'description': 'Large ViT model with excellent accuracy.',
        'pros': ['High accuracy', 'Good for complex defects'],
        'cons': ['Slower', 'High memory usage'],
        'min_gpu_memory': 4.5
    },

    'google/siglip-so400m-patch14-384': {
        'display_name': 'SigLIP - Fine Details',
        'category': 'accurate',
        'download_size': '1.5GB',
        'gpu_memory': '~6GB',
        'speed_estimate': '5-8 FPS on RTX 3060',
        'accuracy': 'Very High',
        'use_case': 'Extreme detail inspection, research',
        'recommended_image_sizes': [384, 392],
        'description': 'Specialized for fine detail detection. Slowest but most thorough.',
        'pros': ['Exceptional detail', 'Best for tiny defects'],
        'cons': ['Very slow', 'Very high memory', 'Large download', 'Not for real-time'],
        'min_gpu_memory': 6.0
    },
}


def get_models_by_category(category=None):
    """Get models organized by category

    Args:
        category: 'fast', 'balanced', or 'accurate' (None returns all)

    Returns:
        dict or list: Models organized by category or list if category specified
    """
    if category:
        return [name for name, info in MODEL_INFO.items() if info['category'] == category]

    categorized = {cat: [] for cat in ['fast', 'balanced', 'accurate']}
    for name, info in MODEL_INFO.items():
        categorized[info['category']].append(name)

    return categorized


def get_model_display_name(model_name):
    """Get user-friendly display name for a model

    Args:
        model_name: Technical model name

    Returns:
        str: Display name
    """
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]['display_name']
    return model_name


def get_model_info(model_name):
    """Get detailed information about a model

    Args:
        model_name: Technical model name

    Returns:
        dict: Model information or None if not found
    """
    return MODEL_INFO.get(model_name)


def get_recommended_model():
    """Get the recommended default model

    Returns:
        str: Model name
    """
    for name, info in MODEL_INFO.items():
        if info.get('recommended', False):
            return name
    return 'dinov2_vitb14'  # Fallback


def format_model_details(model_name):
    """Format model details for display in UI

    Args:
        model_name: Technical model name

    Returns:
        str: Formatted details string
    """
    info = get_model_info(model_name)
    if not info:
        return "No information available"

    details = []
    details.append(f"Download: {info['download_size']}")
    details.append(f"GPU Memory: {info['gpu_memory']}")
    details.append(f"Speed: {info['speed_estimate']}")
    details.append(f"Accuracy: {info['accuracy']}")
    details.append(f"\nBest for: {info['use_case']}")

    return " | ".join(details[:3]) + "\n" + details[3] + details[4]


def get_category_display_name(category):
    """Get display name for a category

    Args:
        category: 'fast', 'balanced', or 'accurate'

    Returns:
        str: Display name with icon
    """
    icons = {
        'fast': '🚀',
        'balanced': '⚖️',
        'accurate': '🎯'
    }
    icon = icons.get(category, '')
    name = MODEL_CATEGORIES.get(category, category)
    return f"{icon} {name}"


def filter_models_by_gpu_memory(available_memory_gb):
    """Filter models that will work with available GPU memory

    Args:
        available_memory_gb: Available GPU memory in GB

    Returns:
        list: List of compatible model names
    """
    compatible = []
    for name, info in MODEL_INFO.items():
        if info['min_gpu_memory'] <= available_memory_gb:
            compatible.append(name)
    return compatible


if __name__ == "__main__":
    # Test the model catalog
    print("=== Model Catalog Test ===\n")

    print("Categories:")
    for cat, models in get_models_by_category().items():
        print(f"\n{get_category_display_name(cat)}:")
        for model in models:
            info = get_model_info(model)
            print(f"  - {info['display_name']}")
            print(f"    {info['description']}")

    print("\n\n=== Recommended Model ===")
    rec_model = get_recommended_model()
    print(f"{rec_model}: {get_model_display_name(rec_model)}")
    print(format_model_details(rec_model))

    print("\n\n=== Models for 4GB GPU ===")
    compatible = filter_models_by_gpu_memory(4.0)
    print(f"Found {len(compatible)} compatible models:")
    for model in compatible:
        print(f"  - {get_model_display_name(model)}")
