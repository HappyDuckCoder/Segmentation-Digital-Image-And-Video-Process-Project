from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Load and test with an image
image_path = "assets/test.png"  # or any image in your assets
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set the image
predictor.set_image(image)

# Test with a point prompt
input_point = np.array([[500, 375]])
input_label = np.array([1])

masks, iou_predictions, low_res_masks = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(f"Mask shape: {masks.shape}")
print(f"IOU predictions: {iou_predictions}")

# Visualize all 3 masks
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original image
axes[0, 0].imshow(image)
axes[0, 0].plot(input_point[:, 0], input_point[:, 1], 'g*', markersize=15, label='Input point')
axes[0, 0].set_title("Original Image with Point")
axes[0, 0].axis('off')
axes[0, 0].legend()

# Three mask predictions
for i in range(3):
    row = (i + 1) // 2
    col = (i + 1) % 2
    axes[row, col].imshow(image)
    axes[row, col].imshow(masks[i], cmap='viridis', alpha=0.5)
    axes[row, col].set_title(f"Mask {i+1} (IOU: {iou_predictions[i]:.4f})")
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("sam_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as sam_results.png")