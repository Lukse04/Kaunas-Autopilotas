import os
import json
import random
import shutil
import numpy as np
from PIL import Image
from labelme import utils

# --- Configuration ---
input_dir = '../lane_detection/data/frames pavyzdys'
output_root = ''
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Create output directories
for split in split_ratios:
    os.makedirs(os.path.join(output_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masks', split), exist_ok=True)

# Gather all JSON/PNG pairs
pairs = []
for fname in os.listdir(input_dir):
    if fname.endswith('.json'):
        base = fname[:-5]
        img_path = os.path.join(input_dir, base + '.png')
        json_path = os.path.join(input_dir, fname)
        if os.path.exists(img_path):
            pairs.append((img_path, json_path))

# Shuffle pairs for random splitting
random.shuffle(pairs)

# Determine split indices
n = len(pairs)
train_end = int(split_ratios['train'] * n)
val_end = train_end + int(split_ratios['val'] * n)

# Process each split
for idx, (img_path, json_path) in enumerate(pairs):
    # Determine current split
    if idx < train_end:
        split = 'train'
    elif idx < val_end:
        split = 'val'
    else:
        split = 'test'

    # Load image and JSON annotation
    data = json.load(open(json_path))
    img = np.array(Image.open(img_path))

    # --- Dynamic label mapping to avoid KeyError ---
    shapes = data.get('shapes', [])
    label_name_to_value = {'background': 0}
    for shape in shapes:
        lbl_name = shape.get('label', '')
        if lbl_name:
            label_name_to_value[lbl_name] = 1

    # Convert shapes to label mask
    lbl, _ = utils.shapes_to_label(
        img_shape=img.shape,
        shapes=shapes,
        label_name_to_value=label_name_to_value
    )

    # Create binary mask: line pixels = 255, background = 0
    mask = (lbl == 1).astype(np.uint8) * 255

    # Save mask and copy image to respective split folders
    base_name = os.path.basename(img_path)
    mask_name = os.path.splitext(base_name)[0] + '_mask.png'
    Image.fromarray(mask).save(os.path.join(output_root, 'masks', split, mask_name))
    shutil.copy2(img_path, os.path.join(output_root, 'images', split, base_name))

# Print summary
print("Dataset preparation complete:")
print(f"- Train:  {train_end} samples")
print(f"- Val:    {val_end - train_end} samples")
print(f"- Test:   {n - val_end} samples")