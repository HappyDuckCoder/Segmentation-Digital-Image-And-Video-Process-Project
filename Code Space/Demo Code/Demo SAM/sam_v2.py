from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Load and test with an image
image_path = "assets/leaves.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# ...existing code...

# TỰ ĐỘNG PHÁT HIỆN TẤT CẢ CÁC VÙNG BỊ HƯ
def detect_damaged_regions(image, color_threshold=None):
    """
    Phát hiện vùng lá bị hư dựa trên màu sắc bất thường
    (vùng vàng, nâu, đen khác với xanh lá)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Định nghĩa vùng màu bị hư (vàng, nâu, đen)
    # Vùng vàng/nâu
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Vùng nâu/đen
    lower_brown = np.array([0, 50, 20])
    upper_brown = np.array([20, 255, 100])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Kết hợp các mask
    damaged_mask = cv2.bitwise_or(mask_yellow, mask_brown)
    
    # Làm sạch noise
    kernel = np.ones((3, 3), np.uint8)
    damaged_mask = cv2.morphologyEx(damaged_mask, cv2.MORPH_OPEN, kernel)
    damaged_mask = cv2.morphologyEx(damaged_mask, cv2.MORPH_CLOSE, kernel)
    
    return damaged_mask

# 7. SỬ DỤNG SAM AUTOMATIC MASK GENERATOR
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Bỏ qua vùng quá nhỏ
)

# Tạo tất cả các masks
print("Generating all masks...")
all_masks = mask_generator.generate(image)
print(f"Found {len(all_masks)} masks")

# LỌC CHỈ LẤY VÙNG BỊ HƯ
damaged_color_mask = detect_damaged_regions(image)

filtered_masks = []
for mask_data in all_masks:
    mask = mask_data['segmentation']
    # Tính % overlap với vùng bị hư
    overlap = np.logical_and(mask, damaged_color_mask > 0)
    overlap_ratio = overlap.sum() / (mask.sum() + 1e-8)
    
    # Chỉ giữ lại mask có >= 30% overlap với vùng bị hư
    if overlap_ratio > 0.3:
        filtered_masks.append(mask_data)

print(f"Found {len(filtered_masks)} damaged regions")

# LƯU TỪNG VÙNG BỊ HƯ THÀNH ẢNH RIÊNG
output_dir = "damaged_regions"
os.makedirs(output_dir, exist_ok=True)

for idx, mask_data in enumerate(filtered_masks):
    mask = mask_data['segmentation']
    bbox = mask_data['bbox']  # [x, y, w, h]
    
    # Crop vùng chứa mask
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w].copy()
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Áp dụng mask (làm nền trắng)
    cropped_image[~cropped_mask] = 255
    
    # Lưu ảnh
    output_path = os.path.join(output_dir, f"damaged_{idx+1}.png")
    cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    
    print(f"Saved: {output_path}")

# VISUALIZATION TẤT CẢ VÙNG BỊ HƯ
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ảnh gốc
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Mask phát hiện màu bị hư
axes[1].imshow(image)
axes[1].imshow(damaged_color_mask, cmap='Reds', alpha=0.5)
axes[1].set_title("Color-based Damaged Detection")
axes[1].axis('off')

# Tất cả vùng bị hư được SAM phát hiện
axes[2].imshow(image)
for mask_data in filtered_masks:
    mask = mask_data['segmentation']
    axes[2].contour(mask, colors=['red'], linewidths=2)
axes[2].set_title(f"SAM Detected Damaged Regions ({len(filtered_masks)})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("all_damaged_regions.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved {len(filtered_masks)} damaged regions to '{output_dir}/' folder")
print("Visualization saved as 'all_damaged_regions.png'")