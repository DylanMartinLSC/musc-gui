"""
Hardware Detection Utility for MuSc GUI

Detects available GPUs and system capabilities to help users
make informed configuration choices.
"""

import torch


def get_gpu_info():
    """Detect available GPUs and return user-friendly information

    Returns:
        list: List of dictionaries with GPU information:
            - index: GPU device index
            - name: GPU model name
            - memory_gb: Total memory in GB
            - memory_free_gb: Available memory in GB
            - recommended: Boolean, True for first GPU
    """
    gpus = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)

                # Get free memory
                torch.cuda.set_device(i)
                memory_free = torch.cuda.mem_get_info()[0] / (1024**3)

                gpus.append({
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'memory_free_gb': memory_free,
                    'recommended': i == 0  # Default to first GPU
                })
            except Exception as e:
                print(f"Warning: Could not get info for GPU {i}: {e}")

    return gpus


def get_recommended_device():
    """Return recommended device with explanation

    Returns:
        dict: Device recommendation with:
            - device: 'cpu' or GPU index (int)
            - name: Device name
            - memory_gb: Total memory (if GPU)
            - memory_free_gb: Free memory (if GPU)
            - recommendation: User-friendly message
            - warning: Warning message (if applicable)
            - performance_tier: 'high', 'medium', 'low', or 'cpu'
    """
    gpus = get_gpu_info()

    if not gpus:
        return {
            'device': 'cpu',
            'name': 'CPU Mode',
            'warning': 'No GPU detected. Processing will be 10-30x slower. Consider using a computer with an NVIDIA GPU for real-time detection.',
            'recommendation': 'CPU mode - Very slow, not suitable for real-time use',
            'performance_tier': 'cpu'
        }

    # Recommend first GPU with most free memory
    gpu = max(gpus, key=lambda g: g['memory_free_gb'])

    # Determine performance tier based on memory
    if gpu['memory_gb'] >= 8:
        tier = 'high'
        tier_desc = 'Excellent for all models'
    elif gpu['memory_gb'] >= 4:
        tier = 'medium'
        tier_desc = 'Good for most models'
    else:
        tier = 'low'
        tier_desc = 'Use smaller models'

    result = {
        'device': gpu['index'],
        'name': gpu['name'],
        'memory_gb': gpu['memory_gb'],
        'memory_free_gb': gpu['memory_free_gb'],
        'recommendation': f"GPU detected: {gpu['name']} ({gpu['memory_gb']:.1f}GB total, {gpu['memory_free_gb']:.1f}GB free) - {tier_desc}",
        'performance_tier': tier
    }

    # Add warning if memory is low
    if gpu['memory_free_gb'] < 2:
        result['warning'] = f"Low free GPU memory ({gpu['memory_free_gb']:.1f}GB). Close other GPU applications or use a smaller model."

    return result


def get_recommended_model(performance_tier=None):
    """Get recommended model based on hardware capabilities

    Args:
        performance_tier: 'high', 'medium', 'low', or 'cpu' (auto-detected if None)

    Returns:
        str: Recommended model name
    """
    if performance_tier is None:
        device_info = get_recommended_device()
        performance_tier = device_info['performance_tier']

    recommendations = {
        'high': 'dinov2_vitb14',  # Can handle any model, use balanced default
        'medium': 'dinov2_vitb14',  # Recommended balanced model
        'low': 'dino_deitsmall16',  # Faster, smaller model
        'cpu': 'vit_tiny_patch16_224.augreg_in21k'  # Fastest model for CPU
    }

    return recommendations.get(performance_tier, 'dinov2_vitb14')


def format_device_display(device_index, gpu_info_list):
    """Format device information for display in dropdown

    Args:
        device_index: GPU index
        gpu_info_list: List from get_gpu_info()

    Returns:
        str: Formatted display string
    """
    for gpu in gpu_info_list:
        if gpu['index'] == device_index:
            recommended = " ✓ RECOMMENDED" if gpu['recommended'] else ""
            return f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f}GB){recommended}"

    return f"GPU {device_index}"


def check_model_compatibility(model_name, device_info):
    """Check if a model is compatible with the current hardware

    Args:
        model_name: Name of the model
        device_info: Result from get_recommended_device()

    Returns:
        dict: Compatibility information:
            - compatible: Boolean
            - warning: Warning message (if applicable)
            - recommendation: Alternative recommendation (if incompatible)
    """
    # Model memory requirements (approximate, in GB)
    model_memory_requirements = {
        'vit_tiny_patch16_224.augreg_in21k': 1.0,
        'dino_deitsmall16': 1.5,
        'vit_small_patch16_224.dino': 1.5,
        'ViT-B-32': 2.5,
        'ViT-B-16': 2.5,
        'dino_vitbase16': 2.5,
        'dino_vitbase8': 2.5,
        'dinov2_vitb14': 3.0,
        'ViT-L-14': 4.5,
        'dinov2_vitl14': 5.0,
        'google/siglip-so400m-patch14-384': 6.0,
    }

    required_memory = model_memory_requirements.get(model_name, 3.0)

    if device_info['device'] == 'cpu':
        if required_memory > 2.0:
            return {
                'compatible': True,  # Will work but be very slow
                'warning': f"This model will be VERY slow on CPU. Consider using a smaller model like 'vit_tiny_patch16_224.augreg_in21k'.",
                'recommendation': 'vit_tiny_patch16_224.augreg_in21k'
            }
        return {'compatible': True}

    # GPU mode
    available_memory = device_info.get('memory_free_gb', device_info.get('memory_gb', 0))

    if required_memory > available_memory:
        return {
            'compatible': False,
            'warning': f"This model requires ~{required_memory:.1f}GB GPU memory, but only {available_memory:.1f}GB is available. Choose a smaller model.",
            'recommendation': get_recommended_model(device_info['performance_tier'])
        }

    if required_memory > available_memory * 0.8:  # Using >80% of memory
        return {
            'compatible': True,
            'warning': f"This model will use most of your GPU memory ({required_memory:.1f}GB of {available_memory:.1f}GB available). Close other GPU applications if you experience issues.",
        }

    return {'compatible': True}


if __name__ == "__main__":
    # Test the hardware detection
    print("=== GPU Detection Test ===\n")

    gpus = get_gpu_info()
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Total Memory: {gpu['memory_gb']:.2f} GB")
            print(f"    Free Memory: {gpu['memory_free_gb']:.2f} GB")
            print(f"    Recommended: {gpu['recommended']}")
    else:
        print("No GPUs detected - CPU mode only")

    print("\n=== Recommended Device ===\n")
    device_info = get_recommended_device()
    print(f"Device: {device_info['device']}")
    print(f"Name: {device_info['name']}")
    print(f"Recommendation: {device_info['recommendation']}")
    if 'warning' in device_info:
        print(f"Warning: {device_info['warning']}")

    print("\n=== Recommended Model ===\n")
    model = get_recommended_model()
    print(f"Recommended model for your hardware: {model}")
