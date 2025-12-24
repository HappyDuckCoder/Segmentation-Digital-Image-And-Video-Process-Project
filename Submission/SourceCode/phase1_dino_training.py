"""
PHASE 1: DINO Self-Supervised Learning
Train ViT với DINO để học features tốt cho segmentation

DINO tốt hơn SimCLR cho segmentation vì:
- Self-attention maps chứa thông tin spatial
- Có thể trực tiếp extract để tạo masks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import timm  # Thư viện ViT models

# ============== CONFIG ==============
class DINOConfig:
    # Paths
    data_dir = "train"  # Thư mục chứa tất cả ảnh (bỏ qua labels)
    output_dir = "dino_checkpoints"
    
    # Model
    arch = "vit_small"  # vit_small, vit_base
    patch_size = 8      # 8 or 16
    
    # Training
    batch_size = 32
    num_epochs = 1
    lr = 0.0005
    weight_decay = 0.04
    
    # DINO specific
    teacher_temp = 0.04
    student_temp = 0.1
    center_momentum = 0.9
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ============== DATASET ==============
class DINODataset(Dataset):
    """Dataset cho DINO - trả về 2 crops của cùng ảnh"""
    def __init__(self, root_dir, transform):
        self.samples = []
        
        # Thu thập tất cả ảnh
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append(img_path)
        
        self.transform = transform
        print(f"📊 DINO Dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        
        # 2 global crops (224x224)
        global_crop1 = self.transform['global'](img)
        global_crop2 = self.transform['global'](img)
        
        return [global_crop1, global_crop2]

# ============== AUGMENTATIONS ==============
def get_dino_transforms():
    """DINO augmentations: global + local crops"""
    
    # Global crops (224x224)
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Local crops (96x96) - smaller crops cho multi-crop
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {'global': global_transform, 'local': local_transform}

# ============== DINO MODEL ==============
class DINOHead(nn.Module):
    """Projection head cho DINO"""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
    
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOModel(nn.Module):
    """Complete DINO model: ViT backbone + projection head"""
    def __init__(self, arch='vit_small_patch8_224'):
        super().__init__()
        
        # ViT backbone (dùng timm)
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        embed_dim = self.backbone.embed_dim
        
        # Projection head
        self.head = DINOHead(embed_dim)
        
    def forward(self, x):
        # Extract features từ ViT
        x = self.backbone.forward_features(x)
        
        # Global average pooling
        if len(x.shape) == 3:  # [B, N, D]
            x = x[:, 0]  # CLS token
        
        # Project
        x = self.head(x)
        return x

# ============== DINO LOSS ==============
class DINOLoss(nn.Module):
    """
    DINO Loss: Knowledge distillation từ teacher sang student
    
    Key ideas:
    - Teacher: EMA của student (không train trực tiếp)
    - Student: Train với gradient descent
    - Centering: Tránh collapse
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, 
                 center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        """
        student_output: list of [B, out_dim] tensors (2 global + 2 local crops)
        teacher_output: list of [B, out_dim] tensors (chỉ 2 global crops)
        """
        # Teacher: softmax với temperature và centering
        temp = teacher_output / self.teacher_temp
        teacher_out = torch.softmax((temp - self.center), dim=-1)
        teacher_out = teacher_out.detach()  # Stop gradient
        
        # Student: softmax với temperature
        student_out = torch.softmax(student_output / self.student_temp, dim=-1)
        
        # Cross-entropy loss
        loss = -torch.sum(teacher_out * torch.log(student_out + 1e-8), dim=-1)
        loss = loss.mean()
        
        # Update center (EMA)
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center với EMA"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)

# ============== TRAINING ==============
def train_dino(config):
    """Train DINO model"""
    
    print("="*70)
    print("PHASE 1: DINO SELF-SUPERVISED TRAINING")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print("="*70 + "\n")
    
    # Dataset
    transforms_dict = get_dino_transforms()
    dataset = DINODataset(config.data_dir, transforms_dict)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Models: Student và Teacher
    student = DINOModel(f'{config.arch}_patch{config.patch_size}_224').to(config.device)
    teacher = DINOModel(f'{config.arch}_patch{config.patch_size}_224').to(config.device)
    
    # Teacher = copy của student (không train trực tiếp)
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Optimizer chỉ cho student
    optimizer = optim.AdamW(
        student.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.num_epochs
    )
    
    # Loss
    criterion = DINOLoss(
        out_dim=2048,
        teacher_temp=config.teacher_temp,
        student_temp=config.student_temp,
        center_momentum=config.center_momentum
    ).to(config.device)
    
    # Training loop
    os.makedirs(config.output_dir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        student.train()
        teacher.eval()
        
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for crops in pbar:
            # crops = [global1, global2, local1, local2]
            # Move to device
            crops = [crop.to(config.device) for crop in crops]
            
            # Student: forward tất cả crops
            student_outputs = []
            for crop in crops:
                output = student(crop)
                student_outputs.append(output)
            
            # Teacher: chỉ forward global crops
            with torch.no_grad():
                teacher_outputs = []
                for crop in crops[:2]:  # Chỉ 2 global crops
                    output = teacher(crop)
                    teacher_outputs.append(output)
            
            # Compute loss
            # Mỗi crop của student học từ tất cả global crops của teacher
            loss = 0
            for s_out in student_outputs:
                for t_out in teacher_outputs:
                    loss += criterion(s_out, t_out)
            loss = loss / (len(student_outputs) * len(teacher_outputs))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher (EMA của student)
            with torch.no_grad():
                m = 0.996  # momentum
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data = m * param_t.data + (1 - m) * param_s.data
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            save_path = os.path.join(
                config.output_dir, 
                f'dino_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, save_path)
            print(f"💾 Checkpoint saved: {save_path}")
    
    print("\n" + "="*70)
    print("DINO TRAINING COMPLETED!")
    print("="*70)
    
    return student, teacher

# ============== MAIN ==============
if __name__ == "__main__":
    # Cài đặt timm nếu chưa có
    # pip install timm
    
    config = DINOConfig()
    student, teacher = train_dino(config)
    
    # Save final model
    torch.save({
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
    }, 'dino_final.pth')
    
    print("\n✅ Model saved: dino_final.pth")
    print("📝 Next step: Run phase2_generate_masks.py")
