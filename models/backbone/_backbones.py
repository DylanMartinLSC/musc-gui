import timm  # noqa
import torchvision.models as models  # noqa
import models.backbone.vision_transformer as vits
import models.backbone.dino_vision_transformer as dino_vits
import torch

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch8_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch8_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch8_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch8_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch8_224", pretrained=True)',
    "vit_swin_base_win12": 'timm.create_model("swin_base_patch4_window12_384.ms_in22k", pretrained=True)',
    "vit_swin_base_win7": 'timm.create_model("swin_base_patch4_window7_224.ms_in22k", pretrained=True)',
    "vit_swin_large_win12": 'timm.create_model("swin_large_patch4_window12_384.ms_in22k", pretrained=True)',
    "vit_swin_large_win7": 'timm.create_model("swin_large_patch4_window7_224.ms_in22k", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)'
}


def load(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. dino family models ---
    if name.startswith("dino_"):
        if name == "dino_deitsmall16":
            patch_size = 16
            model_type = "vit_small"   # small variant (384-dim)
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif name == "dino_vitbase16":
            patch_size = 16
            model_type = "vit_base"    # base variant (768-dim)
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif name == "dino_vitbase8":
            patch_size = 8
            model_type = "vit_base"    # base variant (768-dim)
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError(f"Unknown dino model name: {name}")
        # Here we assume your vision transformer modules are available in the 'vits' namespace.
        from models.backbone import vision_transformer as vits
        model = vits.__dict__[model_type](patch_size=patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        # Download and load the state dict from the URL.
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url,
            map_location=device
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    # --- 2. dinov2 models ---
    if name.startswith("dinov2_"):
        # Use Facebook's hub loader to load dinov2 models
        model = torch.hub.load('facebookresearch/dinov2', name)
        model.to(device)
        return model

    # --- 3. CLIP-based models with names like ViT-B-32, ViT-B-16, ViT-L-14 ---
    if name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        # Assume these are available through your open_clip module.
        import models.backbone.open_clip as open_clip
        model, _, _ = open_clip.create_model_and_transforms(name)
        model.to(device)
        return model

    # --- 4. HuggingFace-based CLIP and TinyCLIP models ---
    # (Note: add missing commas in your combo list if needed.)
    if (name.startswith("google/") or name.startswith("jinaai/") or 
        name.startswith("wkcn/") or name.startswith("sidmanale643/") or 
        name.startswith("sachin/")):
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(name)
        # Optionally, if you need the processor you might store or return it as well:
        processor = CLIPProcessor.from_pretrained(name)
        model.to(device)
        return model

    # --- 5. Facebook DeiT models ---
    if name.startswith("facebook/"):
        # For example: "facebook/deit-tiny-patch16-224" or
        # "facebook/deit-tiny-distilled-patch16-224"
        import timm
        # timm model names typically need underscores instead of slashes.
        model = timm.create_model(name.replace("/", "_"), pretrained=True)
        model.to(device)
        return model

    # --- 6. Apple MobileViT models ---
    if name.startswith("apple/mobilevit-small"):
        import timm
        model = timm.create_model("mobilevit_small", pretrained=True)
        model.to(device)
        return model

    # --- New: TIMM models with names starting with "vit_" ---
    if name.startswith("vit_"):
        import timm
        model = timm.create_model(name, pretrained=True)
        model.to(device)
        return model

    # --- 7. Fallback ---
    from models.backbone import _backbones as backbones
    return eval(backbones._BACKBONES[name])


    # --- 7. Fallback: _BACKBONES dictionary ---
    # If the model isn't handled by any of the above branches, fall back to your
    # existing _BACKBONES dictionary. (Ensure that the keys in your dictionary are correct.)
    from models.backbone import _backbones as backbones
    return eval(backbones._BACKBONES[name])
