import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from multiprocessing import freeze_support

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for lane detection')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0',
                        help='Encoder (resnet34, resnet50, efficientnet-b0, etc)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--multiscale', action='store_true',
                        help='Enable multi-scale training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best_deeplab.pth if it exists')
    return parser.parse_args()

# Dataset class
class LaneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=None):
        self.images = sorted(os.listdir(images_dir))
        self.masks  = sorted(os.listdir(masks_dir))
        self.img_dir  = images_dir
        self.mask_dir = masks_dir
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img_np = np.array(Image.open(img_path).convert('RGB'))
        mask_np = np.array(Image.open(mask_path).convert('L')) / 255.0

        if self.augment:
            augmented = self.augment(image=img_np, mask=mask_np)
            img = augmented['image'].float() / 255.0
            mask = augmented['mask']
            if mask.dtype != torch.float32:
                mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            img = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        return img, mask

# Prepare augmentations
def get_transforms(multiscale=False):
    if multiscale:
        ops = [A.RandomResizedCrop(512,512, scale=(0.5,1.5), ratio=(1.0,1.0), p=1.0)]
    else:
        ops = [A.Resize(512,512)]
    ops += [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Perspective(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ]
    train_t = A.Compose(ops)
    val_t = A.Compose([A.Resize(512,512), ToTensorV2()])
    return train_t, val_t

# Main training loop
def main():
    args = parse_args()
    base_dir = os.path.abspath('.')
    train_images = os.path.join(base_dir, 'images', 'train')
    train_masks  = os.path.join(base_dir, 'masks', 'train')
    val_images   = os.path.join(base_dir, 'images', 'val')
    val_masks    = os.path.join(base_dir, 'masks', 'val')

    # Transforms
    train_transform, val_transform = get_transforms(multiscale=args.multiscale)

    # Datasets & Loaders
    train_ds = LaneDataset(train_images, train_masks, augment=train_transform)
    val_ds   = LaneDataset(val_images,   val_masks,   augment=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model, optimizer, loss, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.DeepLabV3Plus(encoder_name=args.encoder,
                              encoder_weights='imagenet',
                              in_channels=3, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode='binary')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # Resume from checkpoint if requested
    best_val = float('inf')
    ckpt_path = os.path.join(base_dir, 'best_deeplab.pth')
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming training from checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Training and validation
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = bce(preds, masks) + dice(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += (bce(preds, masks) + dice(preds, masks)).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print("  → Saved best model")

    print("Training complete.")

if __name__ == '__main__':
    freeze_support(); main()
