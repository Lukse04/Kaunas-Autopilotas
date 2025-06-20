import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt

# Test script for DeepLabV3+ lane detection
# Automatically locates project root relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# Paths
images_dir = os.path.join(project_root, 'images', 'test')
masks_dir  = os.path.join(project_root, 'masks', 'test')
preds_dir  = os.path.join(project_root, 'predictions')

# Ensure output directory exists
os.makedirs(preds_dir, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.DeepLabV3Plus(
    encoder_name='resnet50',  # match training encoder or set via argument
    encoder_weights=None,
    in_channels=3,
    classes=1
)
ckpt_path = os.path.join(project_root, 'best_deeplab.pth')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device).eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
mask_size = (512, 512)

# Process test set
dice_scores = []
for fname in sorted(os.listdir(images_dir)):
    if not fname.lower().endswith('.png'):
        continue

    # Load image and true mask
    img_path = os.path.join(images_dir, fname)
    mask_name = fname.replace('.png', '_mask.png')
    mask_path = os.path.join(masks_dir, mask_name)

    img = Image.open(img_path).convert('RGB')
    mask_img = Image.open(mask_path).convert('L')

    # Resize for model input and ground truth
    img_resized  = img.resize(mask_size, resample=Image.BILINEAR)
    mask_resized = mask_img.resize(mask_size, resample=Image.NEAREST)

    # Prepare tensor
    x = transform(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0, 0].cpu().numpy()
    mask_pred = pred > 0.5

    # Compute Dice on resized masks
    mask_true = np.array(mask_resized) > 128
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = mask_pred.sum() + mask_true.sum()
    dice = 2 * intersection / union if union > 0 else 1.0
    dice_scores.append(dice)

    # Upscale prediction to original size
    pred_img = Image.fromarray((mask_pred * 255).astype(np.uint8))
    pred_full = pred_img.resize(img.size, resample=Image.NEAREST)
    pred_full.save(os.path.join(preds_dir, fname))

    # Visualize per-image results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.array(mask_img), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_full, cmap='gray')
    plt.title(f'Prediction (Dice: {dice:.4f})')
    plt.axis('off')

    plt.show()

# Print overall metric
mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
print(f"Mean Dice on test set: {mean_dice:.4f}")
