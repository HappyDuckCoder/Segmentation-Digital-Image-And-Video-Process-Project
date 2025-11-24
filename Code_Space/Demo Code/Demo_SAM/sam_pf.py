from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. TỰ ĐỘNG CHỌN DEVICE (GPU nếu có)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Debug: Verify model_type and registry
print(f"model_type: {model_type}")
print(f"Available models: {list(sam_model_registry.keys())}")
print(f"Selected builder: {sam_model_registry[model_type]}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Load and test with an image
image_path = "assets/PT2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

# ...existing code...

# 5. TÁCH LÁ RA KHỎI NỀN (LOẠI BỎ BÓNG ĐEN)
def extract_leaf_mask(image):
    """
    Tách lá ra khỏi nền và bóng đổ bằng cách phân tích màu sắc và độ sáng
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 1. Range cho màu xanh lá (lá khỏe)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 2. Vùng vàng/nâu (lá bị hư) - CHỈ lấy vùng có màu sắc rõ ràng (saturation > 25)
    lower_yellow = np.array([10, 25, 40])  # Tăng saturation để loại bỏ bóng xám
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3. Vùng trắng/xám nhạt (nấm, mốc) - saturation thấp NHƯNG value cao
    lower_white = np.array([0, 0, 150])  # Chỉ lấy vùng sáng
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 4. LOẠI BỎ BÓNG ĐEN - chỉ lấy vùng đen có màu sắc (saturation > 20)
    # Bóng đổ thường có saturation rất thấp (xám đen thuần)
    lower_dark_leaf = np.array([0, 20, 10])  # Saturation > 20 để loại bỏ bóng xám
    upper_dark_leaf = np.array([180, 255, 80])
    mask_dark = cv2.inRange(hsv, lower_dark_leaf, upper_dark_leaf)
    
    # Kết hợp TẤT CẢ các mask
    leaf_mask = cv2.bitwise_or(mask_green, mask_yellow)
    leaf_mask = cv2.bitwise_or(leaf_mask, mask_white)
    leaf_mask = cv2.bitwise_or(leaf_mask, mask_dark)
    
    # Làm sạch noise
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    
    # Lấy contour lớn nhất (chiếc lá chính) - loại bỏ bóng rơi xa
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sắp xếp theo diện tích
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Chỉ giữ lại contour lớn nhất (lá chính)
        leaf_mask = np.zeros_like(leaf_mask)
        cv2.drawContours(leaf_mask, [contours[0]], -1, 255, -1)
    
    return leaf_mask

# 6. TỰ ĐỘNG PHÁT HIỆN TẤT CẢ CÁC VÙNG BỊ HƯ
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
# Tách lá ra nền trắng
print("Extracting leaf from background...")
leaf_mask = extract_leaf_mask(image)
leaf_with_white_bg = image.copy()
leaf_with_white_bg[leaf_mask == 0] = [255, 255, 255]  # Nền trắng

# 7. SỬ DỤNG SAM AUTOMATIC MASK GENERATOR
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.7,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Bỏ qua vùng quá nhỏ
)

# Tạo tất cả các masks trên ảnh đã tách nền
print("Generating all masks...")
all_masks = mask_generator.generate(leaf_with_white_bg)
print(f"Found {len(all_masks)} masks")

# 8. LỌC CHỈ LẤY VÙNG BỊ HƯ
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

# # 9. LƯU TỪNG VÙNG BỊ HƯ THÀNH ẢNH RIÊNG
output_dir = "damaged_regions"
# os.makedirs(output_dir, exist_ok=True)

# for idx, mask_data in enumerate(filtered_masks):
#     mask = mask_data['segmentation']
#     bbox = mask_data['bbox']  # [x, y, w, h]
    
#     # Crop vùng chứa mask
#     x, y, w, h = bbox
#     cropped_image = image[y:y+h, x:x+w].copy()
#     cropped_mask = mask[y:y+h, x:x+w]
    
#     # Áp dụng mask (làm nền trắng)
#     cropped_image[~cropped_mask] = 255
    
#     # Lưu ảnh
#     output_path = os.path.join(output_dir, f"damaged_{idx+1}.png")
#     cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    
#     print(f"Saved: {output_path}")

# 10. VISUALIZATION TẤT CẢ VÙNG BỊ HƯ
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ảnh gốc với nền trắng
axes[0].imshow(leaf_with_white_bg)
axes[0].set_title("Leaf on White Background")
axes[0].axis('off')

# Mask phát hiện màu bị hư - làm nổi bật vùng hư
damaged_highlight = leaf_with_white_bg.copy()
# Tô vùng bị hư bằng màu đỏ sáng
damaged_highlight[damaged_color_mask > 0] = [255, 0, 0]  # Đỏ thuần
# Blend với ảnh gốc để vẫn thấy chi tiết
blended = cv2.addWeighted(leaf_with_white_bg, 0.4, damaged_highlight, 0.6, 0)
axes[1].imshow(blended)
axes[1].set_title("Color-based Damaged Detection")
axes[1].axis('off')

# Tất cả vùng bị hư được SAM phát hiện - TÔ MÀU thay vì chỉ viền
sam_visualization = leaf_with_white_bg.copy()
# Tạo overlay màu cho từng vùng bị hư
for idx, mask_data in enumerate(filtered_masks):
    mask = mask_data['segmentation']
    # Tô màu đỏ với độ trong suốt
    color_mask = np.zeros_like(sam_visualization)
    color_mask[mask] = [255, 0, 0]  # Đỏ
    sam_visualization = cv2.addWeighted(sam_visualization, 1, color_mask, 0.5, 0)
    
    # Vẽ viền để phân biệt các vùng
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_visualization, contours, -1, (255, 255, 0), 2)  # Viền vàng

axes[2].imshow(sam_visualization)
axes[2].set_title(f"SAM Detected Damaged Regions ({len(filtered_masks)})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("all_damaged_regions.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved {len(filtered_masks)} damaged regions to '{output_dir}/' folder")
print("Visualization saved as 'all_damaged_regions.png'")

#Best version