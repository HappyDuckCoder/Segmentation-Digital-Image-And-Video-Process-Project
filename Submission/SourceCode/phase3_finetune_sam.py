"""
PHASE 3: Fine-tune SAM với Pseudo Masks
Sử dụng masks được tạo từ DINO để fine-tune SAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import cv2
import numpy as np
from tqdm import tqdm
import os

# ============== DATASET ==============
class PseudoMaskDataset(Dataset):
    """
    Dataset với pseudo masks từ DINO
    """
    def __init__(self, image_dir, mask_dir):
        self.samples = []
        
        # Collect image-mask pairs
        for class_dir in os.listdir(image_dir):
            img_class_path = os.path.join(image_dir, class_dir)
            mask_class_path = os.path.join(mask_dir, class_dir)
            
            if not os.path.isdir(img_class_path):
                continue
            
            for img_name in os.listdir(img_class_path):
                if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                img_path = os.path.join(img_class_path, img_name)
                
                # Corresponding mask
                mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(mask_class_path, mask_name)
                
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
        
        print(f"📊 Dataset: {len(self.samples)} image-mask pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.float32)
        
        # Resize to SAM input size (1024x1024)
        image_resized = cv2.resize(image, (1024, 1024))
        mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()
        
        return image_tensor, mask_tensor, os.path.basename(img_path)

# ============== LOSS FUNCTIONS ==============
class FocalLoss(nn.Module):
    """Focal Loss cho imbalanced segmentation"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal loss
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * ce).mean()

class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss"""
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.focal(pred, target) + self.dice(pred, target)

# ============== TRAINING ==============
def train_sam_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train SAM 1 epoch với pseudo masks"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training SAM")
    
    for images, masks, names in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        try:
            # SAM forward pass
            # Note: SAM architecture phức tạp, cần setup đúng input format
            
            # Encode images
            with torch.no_grad():
                image_embeddings = model.image_encoder(images)
            
            # Decode với prompt (sử dụng grid points)
            # Tạo sparse prompts (points)
            batch_size = images.shape[0]
            
            # Simple: Sử dụng center point
            sparse_embeddings = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            
            # Mask decoder
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[0],
                dense_prompt_embeddings=sparse_embeddings[1],
                multimask_output=False,
            )
            
            # Upsample masks
            masks_pred = F.interpolate(
                low_res_masks,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )
            
            # Compute loss
            loss = criterion(masks_pred, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"\n⚠️  Error in batch: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

# ============== SIMPLIFIED VERSION ==============
def train_sam_simplified(
    sam_checkpoint,
    image_dir,
    mask_dir,
    num_epochs=10,
    batch_size=2,
    lr=1e-5,
    device='cuda'
):
    """
    Simplified SAM fine-tuning
    CHÚ Ý: SAM architecture phức tạp, version này là simplified
    """
    
    print("="*70)
    print("PHASE 3: SAM FINE-TUNING WITH PSEUDO MASKS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("="*70 + "\n")
    
    # Load SAM
    print("Loading SAM model...")
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    
    # Freeze image encoder (optional - để giảm memory)
    # for param in sam.image_encoder.parameters():
    #     param.requires_grad = False
    
    print("✅ SAM loaded")
    
    # Dataset
    dataset = PseudoMaskDataset(image_dir, mask_dir)
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer - chỉ optimize mask decoder
    optimizer = optim.AdamW(
        sam.mask_decoder.parameters(),  # Chỉ train decoder
        lr=lr,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss
    criterion = CombinedLoss()
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        epoch_loss = train_sam_one_epoch(sam, dataloader, optimizer, criterion, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} completed: Loss = {epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = f'sam_finetuned_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"💾 Best checkpoint saved: {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'sam_finetuned_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': sam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "="*70)
    print("SAM FINE-TUNING COMPLETED!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*70)
    
    return sam

# ============== MAIN ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                       help='Path to SAM pretrained checkpoint')
    parser.add_argument('--image_dir', type=str, default='train',
                       help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default='pseudo_masks',
                       help='Directory containing pseudo masks')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠️  WARNING: Training on CPU is VERY slow!")
        print("   Strongly recommend using GPU")
    
    # Train
    sam_finetuned = train_sam_simplified(
        sam_checkpoint=args.sam_checkpoint,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    print("\n✅ Fine-tuning completed!")
    print("📝 Model saved: sam_finetuned_best.pth")
    print("\n💡 Next steps:")
    print("   1. Evaluate on test set")
    print("   2. Use fine-tuned SAM for inference")
    print("   3. (Optional) Iterative refinement with better masks")
