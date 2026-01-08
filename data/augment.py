import os
import json
import cv2
import random
import numpy as np
from math import sqrt
from tqdm import tqdm
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform
import yaml

# ---------- Custom Transforms ----------
class ColorOverlay(A.DualTransform):
    """Apply a random color overlay to the image."""

    def __init__(self, colors, alpha_range=(0.2, 0.5), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.colors = colors
        self.alpha_range = alpha_range

    def apply(self, img, **params):
        color = random.choice(self.colors)
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

        # Convert RGB to BGR (OpenCV format)
        color_bgr = np.array([color[2], color[1], color[0]], dtype=np.uint8)

        overlay = np.full(img.shape, color_bgr, dtype=np.uint8)
        return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    def apply_to_keypoints(self, keypoints, **params):
        return keypoints  # Keypoints unchanged by color overlay

    def get_transform_init_args_names(self):
        return ("colors", "alpha_range")


# ---------- Config ----------
def load_cfg(path="augmentations.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Global config - will be set when run_all_augmentations is called
CFG = None

# ---------- IO ----------
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = data['categories']
    info = data.get("info", {})
    licenses = data.get("licenses", [])
    return data, images, annotations, categories, info, licenses

# ---------- Core augmentation helper ----------
def apply_augmentation(name, aug, p, coco_data, images, annotations, img_dir, original_image_ids):
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1

    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]

    added_count = 0
    for ann in tqdm(original_annotations, desc=f"Applying {name}", unit="img"):
        if random.random() > p:
            continue

        img = images[ann['image_id']]
        image_path = os.path.join(img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        kps = ann["keypoints"]
        # Split into (x, y) coordinates and visibility
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]

        try:
            # For area-based augmentations, use actual image dimensions
            if "occlusion" in name:
                img_height = img["height"]
                img_width = img["width"]
                occlusion_cfg = CFG["augmentations"]["occlusion"]
                a0, a1 = occlusion_cfg["area_pct"]
                min_holes = occlusion_cfg.get("min_holes", 1)
                max_holes = occlusion_cfg.get("max_holes", 1)
                dynamic_occlusion = A.Compose([A.CoarseDropout(
                    min_holes=min_holes, max_holes=max_holes, 
                    min_height=int(a0 * img_height), max_height=int(a1 * img_height),
                    min_width=int(a0 * img_width), max_width=int(a1 * img_width), 
                    fill_value=(0,0,0), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_occlusion(image=image, keypoints=keypoints)
            elif "crop" in name:
                img_height = img["height"]
                img_width = img["width"]
                ratio = CFG["augmentations"]["crop"]["ratio"]
                dynamic_crop = A.Compose([A.RandomResizedCrop(
                    height=img_height, width=img_width, scale=ratio, ratio=(1.33,1.77), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_crop(image=image, keypoints=keypoints)
            else:
                augmented = aug(image=image, keypoints=keypoints)
        except Exception as e:
            print(f"⚠️ Augmentation failed for {img['file_name']}: {e}")
            continue

        aug_img = augmented["image"]
        aug_kps = augmented["keypoints"]

        # Get image dimensions
        img_height, img_width = aug_img.shape[:2]

        # Reconstruct flat keypoints safely, always length = 51
        aug_kps_flat = []
        for i in range(len(vis)):  # 17 keypoints
            if i < len(aug_kps):
                x, y = aug_kps[i]
            else:
                # If Albumentations dropped a keypoint, fallback to original coords
                x, y = keypoints[i]

            # Determine visibility - set to 0 if keypoint is outside image bounds
            v = vis[i]
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                v = 0  # Keypoint is outside image bounds
            
            aug_kps_flat.extend([x, y, v])

        # Update bbox for geometric augmentations
        updated_bbox = ann["bbox"]
        if any(geom_aug in name for geom_aug in ["crop", "rotation", "scale", "translation"]):
            # Calculate new bbox from keypoints
            visible_keypoints = [(aug_kps_flat[i], aug_kps_flat[i+1]) for i in range(0, len(aug_kps_flat), 3) if aug_kps_flat[i+2] > 0]
            if visible_keypoints:
                x_coords = [kp[0] for kp in visible_keypoints]
                y_coords = [kp[1] for kp in visible_keypoints]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add 5% margin around the bbox (with safety check)
                if x_max > x_min and y_max > y_min:
                    margin_x = (x_max - x_min) * 0.05
                    margin_y = (y_max - y_min) * 0.05
                    x_min -= margin_x
                    x_max += margin_x
                    y_min -= margin_y
                    y_max += margin_y
                    
                    # Clamp bbox to image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)
                    
                    updated_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    # If all keypoints are at same position, use original bbox
                    updated_bbox = ann["bbox"]
            else:
                # If no visible keypoints, use original bbox
                updated_bbox = ann["bbox"]

        base_name, ext = os.path.splitext(img["file_name"])
        new_fname = f"{base_name}_{name}{ext}"
        out_path = os.path.join(img_dir, new_fname)
        cv2.imwrite(out_path, aug_img)

        coco_data["images"].append({
            "id": next_img_id,
            "file_name": new_fname,
            "height": aug_img.shape[0],
            "width": aug_img.shape[1]
        })

        coco_data["annotations"].append({
            "id": next_ann_id,
            "image_id": next_img_id,
            "category_id": ann["category_id"],
            "bbox": updated_bbox,
            "area": updated_bbox[2] * updated_bbox[3],  # width * height
            "iscrowd": ann["iscrowd"],
            "keypoints": aug_kps_flat
        })

        images[next_img_id] = coco_data["images"][-1]
        # Note: annotations is already coco_data["annotations"], no need to append again

        next_img_id += 1
        next_ann_id += 1
        added_count += 1

    return added_count

# ---------- Build augmentations from YAML ----------
def build_all_augs():
    aug_list = []
    augmentations = CFG["augmentations"]

    # Brightness + Contrast (combined)
    if "brightness_contrast" in augmentations:
        brightness_limit = augmentations["brightness_contrast"]["brightness_limit"]
        contrast_limit = augmentations["brightness_contrast"]["contrast_limit"]
        p = augmentations["brightness_contrast"].get("probability", 1.0)
        aug_list.append(("brightness_contrast", A.Compose([A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Color overlay
    if "color_overlay" in augmentations:
        colors = augmentations["color_overlay"]["colors"]
        alpha_range = augmentations["color_overlay"]["alpha"]
        p = augmentations["color_overlay"].get("probability", 0.2)
        aug_list.append(("color_overlay", A.Compose([ColorOverlay(colors=colors, alpha_range=alpha_range, p=p)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Crop
    if "crop" in augmentations:
        ratio = augmentations["crop"]["ratio"]
        p = augmentations["crop"].get("probability", 1.0)
        # Create a placeholder transform - actual dimensions will be set at runtime
        aug_list.append(("crop", A.Compose([A.RandomResizedCrop(height=1, width=1, scale=ratio, ratio=(1.33,1.77), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Gamma
    if "gamma" in augmentations:
        gamma_limit = augmentations["gamma"]["gamma_limit"]
        p = augmentations["gamma"].get("probability", 1.0)
        aug_list.append(("gamma", A.Compose([A.RandomGamma(gamma_limit=gamma_limit, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Gaussian blur
    if "gaussian_blur" in augmentations:
        sigma_limit = augmentations["gaussian_blur"]["sigma_limit"]
        p = augmentations["gaussian_blur"].get("probability", 1.0)
        aug_list.append(("gaussian_blur", A.Compose([A.GaussianBlur(blur_limit=(3,7), sigma_limit=sigma_limit, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Hue + Saturation (combined)
    if "hue_saturation" in augmentations:
        hue = augmentations["hue_saturation"]["hue"]
        saturation = augmentations["hue_saturation"]["saturation"]
        p = augmentations["hue_saturation"].get("probability", 1.0)
        aug_list.append(("hue_saturation", A.Compose([A.ColorJitter(brightness=0, contrast=0, saturation=saturation, hue=hue, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Motion blur
    if "motion_blur" in augmentations:
        blur_limit = augmentations["motion_blur"]["blur_limit"]
        p = augmentations["motion_blur"].get("probability", 1.0)
        aug_list.append(("motion_blur", A.Compose([A.MotionBlur(blur_limit=blur_limit, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Occlusion
    if "occlusion" in augmentations:
        a0, a1 = augmentations["occlusion"]["area_pct"]
        p = augmentations["occlusion"].get("probability", 1.0)
        # Create a placeholder transform - actual dimensions will be set at runtime
        aug_list.append(("occlusion", A.Compose([A.CoarseDropout(min_holes=1, max_holes=1, min_height=1, max_height=1,
                                                                    min_width=1, max_width=1, fill_value=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Rotation
    if "rotation" in augmentations:
        limit = augmentations["rotation"]["limit"]
        p = augmentations["rotation"].get("probability", 1.0)
        aug_list.append(("rotate", A.Compose([A.Rotate(limit=limit, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))


    # Scale
    if "scale" in augmentations:
        scale = augmentations["scale"]["scale"]
        p = augmentations["scale"].get("probability", 1.0)
        aug_list.append(("scale", A.Compose([A.Affine(scale=scale, fit_output=False, mode=cv2.BORDER_CONSTANT, cval=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Shadow
    if "shadow" in augmentations:
        shadow_dimension = augmentations["shadow"]["shadow_dimension"]
        p = augmentations["shadow"].get("probability", 1.0)
        aug_list.append(("shadow", A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=1, 
                                                           shadow_dimension=shadow_dimension, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Translation
    if "translation" in augmentations:
        translate_percent = augmentations["translation"]["translate_percent"]
        p = augmentations["translation"].get("probability", 1.0)
        aug_list.append(("translation", A.Compose([A.Affine(translate_percent=translate_percent, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    return aug_list

# ---------- Dependent augmentation helper ----------
def apply_dependent_augmentations(coco_data, images, annotations, img_dir, original_image_ids):
    """Apply augmentations with dependencies (1 geometric per color)."""
    augmentations = CFG["augmentations"]
    
    # Separate color and geometric augmentations
    color_augs = []
    geometric_augs = []
    
    for aug_name, aug_config in augmentations.items():
        if aug_config.get("is_geometric", False):
            geometric_augs.append(aug_name)
        else:
            color_augs.append(aug_name)
    
    print(f"Color augmentations: {color_augs}")
    print(f"Geometric augmentations: {geometric_augs}")
    
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1
    
    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]
    
    added_count = 0
    for ann in tqdm(original_annotations, desc="Applying dependent augmentations", unit="img"):
        img = images[ann['image_id']]
        image_path = os.path.join(img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        kps = ann["keypoints"]
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]
                
        # Apply each color augmentation independently with its own probability
        for color_aug_name in color_augs:
            aug_config = augmentations[color_aug_name]
            p = aug_config.get("probability", 1.0)
            
            # Check if this color augmentation should be applied
            if random.random() <= p:
                # Build the color augmentation
                if color_aug_name == "brightness_contrast":
                    brightness_limit = aug_config["brightness_limit"]
                    contrast_limit = aug_config["contrast_limit"]
                    aug = A.Compose([A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "color_overlay":
                    colors = aug_config["colors"]
                    alpha_range = aug_config["alpha"]
                    aug = A.Compose([ColorOverlay(colors=colors, alpha_range=alpha_range, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "gamma":
                    gamma_limit = aug_config["gamma_limit"]
                    aug = A.Compose([A.RandomGamma(gamma_limit=gamma_limit, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "gaussian_blur":
                    sigma_limit = aug_config["sigma_limit"]
                    aug = A.Compose([A.GaussianBlur(blur_limit=(3,7), sigma_limit=sigma_limit, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "hue_saturation":
                    hue = aug_config["hue"]
                    saturation = aug_config["saturation"]
                    aug = A.Compose([A.ColorJitter(brightness=0, contrast=0, saturation=saturation, hue=hue, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "motion_blur":
                    blur_limit = aug_config["blur_limit"]
                    aug = A.Compose([A.MotionBlur(blur_limit=blur_limit, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "occlusion":
                    a0, a1 = aug_config["area_pct"]
                    min_holes = aug_config.get("min_holes", 1)
                    max_holes = aug_config.get("max_holes", 1)
                    img_height = img["height"]
                    img_width = img["width"]
                    aug = A.Compose([A.CoarseDropout(
                        min_holes=min_holes, max_holes=max_holes, 
                        min_height=int(a0 * img_height), max_height=int(a1 * img_height),
                        min_width=int(a0 * img_width), max_width=int(a1 * img_width), 
                        fill_value=(0,0,0), p=1.0
                    )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                elif color_aug_name == "shadow":
                    shadow_dimension = aug_config["shadow_dimension"]
                    aug = A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=1, 
                                                   shadow_dimension=shadow_dimension, p=1.0)],
                                   keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                else:
                    continue
                
                try:
                    # Apply color augmentation
                    augmented = aug(image=image, keypoints=keypoints)
                    current_image = augmented["image"]
                    current_keypoints = augmented["keypoints"]
                    
                    # Apply exactly one geometric augmentation to this color-augmented image
                    if geometric_augs:
                        chosen_geometric = random.choice(geometric_augs)
                        geom_config = augmentations[chosen_geometric]
                        
                        # Build the geometric augmentation
                        if chosen_geometric == "crop":
                            ratio = geom_config["ratio"]
                            img_height = img["height"]
                            img_width = img["width"]
                            geom_aug = A.Compose([A.RandomResizedCrop(height=img_height, width=img_width, scale=ratio, ratio=(1.33,1.77), p=1.0)],
                                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        elif chosen_geometric == "rotation":
                            limit = geom_config["limit"]
                            geom_aug = A.Compose([A.Rotate(limit=limit, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0)],
                                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        elif chosen_geometric == "scale":
                            scale = geom_config["scale"]
                            geom_aug = A.Compose([A.Affine(scale=scale, fit_output=False, mode=cv2.BORDER_CONSTANT, cval=(0,0,0), p=1.0)],
                                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        elif chosen_geometric == "translation":
                            translate_percent = geom_config["translate_percent"]
                            geom_aug = A.Compose([A.Affine(translate_percent=translate_percent, p=1.0)],
                                               keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        else:
                            continue
                        
                        try:
                            geom_augmented = geom_aug(image=current_image, keypoints=current_keypoints)
                            current_image = geom_augmented["image"]
                            current_keypoints = geom_augmented["keypoints"]
                            applied_augs = [color_aug_name, chosen_geometric]
                        except Exception as e:
                            print(f"⚠️ Geometric augmentation {chosen_geometric} failed: {e}")
                            applied_augs = [color_aug_name]  # Only color augmentation applied
                    else:
                        applied_augs = [color_aug_name]  # Only color augmentation applied
                    
                    # Get image dimensions
                    img_height, img_width = current_image.shape[:2]
                    
                    # Reconstruct flat keypoints safely, always length = 51
                    aug_kps_flat = []
                    for i in range(len(vis)):  # 17 keypoints
                        if i < len(current_keypoints):
                            x, y = current_keypoints[i]
                        else:
                            # If Albumentations dropped a keypoint, fallback to original coords
                            x, y = keypoints[i]
                        
                        # Determine visibility - set to 0 if keypoint is outside image bounds
                        v = vis[i]
                        if x < 0 or x >= img_width or y < 0 or y >= img_height:
                            v = 0  # Keypoint is outside image bounds
                        
                        aug_kps_flat.extend([x, y, v])
                    
                    # Update bbox for geometric augmentations
                    updated_bbox = ann["bbox"]
                    if any(geom_aug in applied_augs for geom_aug in ["crop", "rotation", "scale", "translation"]):
                        # Calculate new bbox from keypoints
                        visible_keypoints = [(aug_kps_flat[i], aug_kps_flat[i+1]) for i in range(0, len(aug_kps_flat), 3) if aug_kps_flat[i+2] > 0]
                        if visible_keypoints:
                            x_coords = [kp[0] for kp in visible_keypoints]
                            y_coords = [kp[1] for kp in visible_keypoints]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            # Add 5% margin around the bbox (with safety check)
                            if x_max > x_min and y_max > y_min:
                                margin_x = (x_max - x_min) * 0.05
                                margin_y = (y_max - y_min) * 0.05
                                x_min -= margin_x
                                x_max += margin_x
                                y_min -= margin_y
                                y_max += margin_y
                                
                                # Clamp bbox to image bounds
                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(img_width, x_max)
                                y_max = min(img_height, y_max)
                                
                                updated_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                            else:
                                # If all keypoints are at same position, use original bbox
                                updated_bbox = ann["bbox"]
                        else:
                            # If no visible keypoints, use original bbox
                            updated_bbox = ann["bbox"]
                    
                    # Create filename with all applied augmentations
                    base_name, ext = os.path.splitext(img["file_name"])
                    aug_suffix = "_".join(applied_augs)
                    new_fname = f"{base_name}_{aug_suffix}{ext}"
                    out_path = os.path.join(img_dir, new_fname)
                    cv2.imwrite(out_path, current_image)
                    
                    coco_data["images"].append({
                        "id": next_img_id,
                        "file_name": new_fname,
                        "height": current_image.shape[0],
                        "width": current_image.shape[1]
                    })
                    
                    coco_data["annotations"].append({
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": ann["category_id"],
                        "bbox": updated_bbox,
                        "area": updated_bbox[2] * updated_bbox[3],  # width * height
                        "iscrowd": ann["iscrowd"],
                        "keypoints": aug_kps_flat
                    })
                    
                    images[next_img_id] = coco_data["images"][-1]
                    # Note: annotations is already coco_data["annotations"], no need to append again
                    
                    next_img_id += 1
                    next_ann_id += 1
                    added_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Color augmentation {color_aug_name} failed: {e}")
                    continue
    
    return added_count

# ---------- Public callable ----------
def run_all_augmentations(augmentations_yaml="augmentations.yaml", json_path="annotations.json", img_dir="images/"):
    # Load the config from the specified YAML file
    global CFG
    CFG = load_cfg(augmentations_yaml)
    
    coco_data, images, annotations, categories, info, licenses = load_data(json_path)
    original_image_ids = set(images.keys())

    # Check if augmentations are independent or dependent
    independent_augs = CFG.get("independent_augs", True)
    
    if independent_augs:
        # Original behavior - apply each augmentation independently
        aug_list = build_all_augs()
        for name, aug, p in aug_list:
            added = apply_augmentation(name, aug, p, coco_data, images, annotations, img_dir, original_image_ids)
            print(f"Augmentation '{name}' added {added} new images to {img_dir}")
    else:
        # Dependent behavior - apply color augmentations with 1 geometric per color
        dependence_key = CFG.get("dependence_key", "")
        if dependence_key == "1 geometric per color":
            added = apply_dependent_augmentations(coco_data, images, annotations, img_dir, original_image_ids)
            print(f"Dependent augmentations added {added} new images to {img_dir}")
        else:
            print(f"Unknown dependence_key: {dependence_key}")
            return

    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"✅ All augmentations applied and JSON saved to {json_path}")

