"""
PHASE 2: Generate Pseudo Masks từ DINO features
Sử dụng self-attention maps của ViT để tạo segmentation masks
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm

# ============== LOAD DINO MODEL ==============
def load_dino_model(checkpoint_path, arch='vit_small_patch8_224', device='cuda'):
    """Load trained DINO model"""
    print(f"Loading DINO model from {checkpoint_path}...")
    
    # Create model
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract backbone weights (bỏ qua projection head)
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint
    
    # Filter only backbone weights
    backbone_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone_dict[k.replace('backbone.', '')] = v
    
    model.load_state_dict(backbone_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

# ============== EXTRACT ATTENTION MAPS ==============
@torch.no_grad()
def extract_attention_maps(model, image_tensor, device='cuda'):
    """
    Extract self-attention maps từ ViT
    
    ViT self-attention chứa thông tin về:
    - Patch nào attend vào patch nào
    - Foreground vs background
    
    Returns:
        attention_map: [H, W] normalized attention
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Hook để lấy attention maps
    attention_maps = []
    
    def hook_fn(module, input, output):
        # output = attention weights [B, num_heads, num_patches, num_patches]
        attention_maps.append(output)
    
    # Register hooks vào các attention layers
    hooks = []
    for name, module in model.named_modules():
        if 'attn.softmax' in name or isinstance(module, torch.nn.Softmax):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    _ = model(image_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get last layer attention
    if len(attention_maps) > 0:
        # [B, num_heads, num_patches, num_patches]
        attn = attention_maps[-1]
        
        # Average over heads
        attn = attn.mean(dim=1)  # [B, num_patches, num_patches]
        
        # Get attention từ CLS token đến các patches
        # CLS token (index 0) attention cho biết patch nào quan trọng
        attn = attn[0, 0, 1:]  # [num_patches]
        
        # Reshape to spatial dimensions
        patch_size = 8  # hoặc 16, tùy model
        num_patches = int(np.sqrt(attn.shape[0]))
        attn = attn.reshape(num_patches, num_patches)
        
        # Normalize
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        return attn.cpu().numpy()
    
    return None

# ============== ALTERNATIVE: Token-based Segmentation ==============
@torch.no_grad()
def extract_features_for_segmentation(model, image_tensor, device='cuda'):
    """
    Extract patch features và cluster để segment
    Approach tốt hơn attention maps
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Extract features
    features = model.forward_features(image_tensor)  # [B, num_patches+1, embed_dim]
    
    # Remove CLS token
    features = features[:, 1:, :]  # [B, num_patches, embed_dim]
    
    # Reshape
    B, N, D = features.shape
    patch_size = 8
    num_patches_per_side = int(np.sqrt(N))
    
    features = features.reshape(B, num_patches_per_side, num_patches_per_side, D)
    features = features[0]  # [H, W, D]
    
    return features.cpu().numpy()

# ============== GENERATE MASK FROM FEATURES ==============
def generate_mask_from_features(features, method='kmeans', threshold=0.5):
    """
    Generate binary mask từ features
    
    Methods:
    - 'kmeans': K-means clustering (foreground vs background)
    - 'pca': PCA + threshold
    - 'threshold': Simple threshold trên feature norm
    """
    H, W, D = features.shape
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Flatten features
        features_flat = features.reshape(-1, D)
        
        # K-means với 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)
        labels = labels.reshape(H, W)
        
        # Assume foreground là cluster có feature norm lớn hơn
        cluster_norms = []
        for i in range(2):
            cluster_features = features_flat[labels.flatten() == i]
            cluster_norms.append(np.linalg.norm(cluster_features, axis=1).mean())
        
        foreground_cluster = np.argmax(cluster_norms)
        mask = (labels == foreground_cluster).astype(np.float32)
        
    elif method == 'pca':
        from sklearn.decomposition import PCA
        
        features_flat = features.reshape(-1, D)
        
        # PCA xuống 1 dimension
        pca = PCA(n_components=1)
        pca_features = pca.fit_transform(features_flat)
        pca_features = pca_features.reshape(H, W)
        
        # Threshold
        mask = (pca_features > np.median(pca_features)).astype(np.float32)
        
    elif method == 'threshold':
        # Simple: threshold trên feature norm
        feature_norms = np.linalg.norm(features, axis=2)
        threshold_value = np.percentile(feature_norms, 50)  # Median
        mask = (feature_norms > threshold_value).astype(np.float32)
    
    return mask

# ============== POST-PROCESSING ==============
def postprocess_mask(mask, min_size=100):
    """Clean up mask: remove small regions, morphological operations"""
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    
    # Keep only large components
    mask_clean = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask_clean[labels == i] = 255
    
    return mask_clean

# ============== GENERATE MASKS FOR DATASET ==============
def generate_pseudo_masks(
    dino_checkpoint,
    image_dir,
    output_dir,
    arch='vit_small_patch8_224',
    method='kmeans',
    device='cuda'
):
    """Generate pseudo masks cho toàn bộ dataset"""
    
    print("="*70)
    print("PHASE 2: PSEUDO MASK GENERATION")
    print("="*70)
    print(f"Method: {method}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load DINO model
    model = load_dino_model(dino_checkpoint, arch, device)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all images
    image_files = []
    for class_dir in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.endswith(('.jpg', '.png', '.jpeg')):
                    image_files.append((img_path, class_dir, img_name))
    
    print(f"📊 Processing {len(image_files)} images...")
    
    # Process each image
    for img_path, class_name, img_name in tqdm(image_files):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (W, H)
            
            # Transform
            image_tensor = transform(image)
            
            # Extract features
            features = extract_features_for_segmentation(model, image_tensor, device)
            
            # Generate mask
            mask = generate_mask_from_features(features, method=method)
            
            # Resize mask to original size
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Post-process
            mask = postprocess_mask(mask)
            
            # Save mask
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            mask_filename = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(class_output_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path}: {e}")
            continue
    
    print("\n" + "="*70)
    print("PSEUDO MASK GENERATION COMPLETED!")
    print(f"Masks saved to: {output_dir}")
    print("="*70)

# ============== VISUALIZE ==============
def visualize_samples(image_dir, mask_dir, num_samples=5):
    """Visualize pseudo masks"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*3))
    fig.suptitle('Image → Pseudo Mask → Overlay', fontsize=16)
    
    # Get random samples
    class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    class_name = class_dirs[0]
    
    image_class_dir = os.path.join(image_dir, class_name)
    mask_class_dir = os.path.join(mask_dir, class_name)
    
    image_files = [f for f in os.listdir(image_class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    for idx, img_file in enumerate(samples):
        # Load image
        img_path = os.path.join(image_class_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_class_dir, mask_file)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Original image
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title('Original')
            axes[idx, 0].axis('off')
            
            # Mask
            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title('Pseudo Mask')
            axes[idx, 1].axis('off')
            
            # Overlay
            image_np = np.array(image)
            overlay = image_np.copy()
            overlay[mask > 128] = [255, 0, 0]  # Red overlay
            blended = cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)
            axes[idx, 2].imshow(blended)
            axes[idx, 2].set_title('Overlay')
            axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pseudo_masks_visualization.png', dpi=150, bbox_inches='tight')
    print("💾 Visualization saved: pseudo_masks_visualization.png")
    plt.show()

# ============== MAIN ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dino_checkpoint', type=str, required=True,
                       help='Path to DINO checkpoint')
    parser.add_argument('--image_dir', type=str, default='train',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='pseudo_masks',
                       help='Output directory for masks')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'pca', 'threshold'],
                       help='Mask generation method')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    args = parser.parse_args()
    
    # Generate masks
    generate_pseudo_masks(
        dino_checkpoint=args.dino_checkpoint,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        method=args.method
    )
    
    # Visualize
    if args.visualize:
        visualize_samples(args.image_dir, args.output_dir)
    
    print("\n📝 Next step: Run phase3_finetune_sam.py")
