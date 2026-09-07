"""Image transform pipelines."""

import logging
from PIL import Image
import os
from torchvision import transforms
from typing import Tuple, List

logger = logging.getLogger(__name__)

def get_pretrain_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get transformations for pretraining (augmentations)."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_finetune_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get transformations for finetuning (with RandomAffine)."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_eval_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get transformations for evaluation (no augmentations)."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_and_stack_maps(image_dir: str, patient_code: str, eye: str, map_types: List[str] = None) -> Image.Image:
    """Load 3 grayscale maps and stack them into a 3-channel RGB-like image."""
    if map_types is None:
        map_types = ['Axial', 'Anterior', 'Posterior']
        
    channels = []
    for map_type in map_types:
        filename = f"{patient_code}_{eye}_{map_type}.png"
        # Support either passing root dir or the eye subfolder directly
        cand1 = os.path.join(image_dir, filename)
        cand2 = os.path.join(image_dir, patient_code, eye, filename)
        if os.path.exists(cand1):
            filepath = cand1
        elif os.path.exists(cand2):
            filepath = cand2
        else:
            logger.error(f"Image map not found in {image_dir} for {patient_code} {eye} {map_type}")
            raise FileNotFoundError(f"Missing map file: {cand1} or {cand2}")
            
        img = Image.open(filepath)
        if img.mode != 'L':
            img = img.convert('L')
        channels.append(img)
        
    if len(channels) != 3:
        raise ValueError(f"Expected 3 map types, got {len(channels)}")

    # Normalise all channels to the same size (use the largest dimension found).
    # Some patients have maps that were scanned at slightly different resolutions.
    widths, heights = zip(*(c.size for c in channels))
    target_w, target_h = max(widths), max(heights)
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        logger.debug(
            "Map size mismatch for %s %s — resizing all channels to (%d, %d)",
            patient_code, eye, target_w, target_h,
        )
        channels = [c.resize((target_w, target_h), Image.LANCZOS) for c in channels]

    stacked_img = Image.merge('RGB', tuple(channels))
    return stacked_img
