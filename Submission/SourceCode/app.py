import streamlit as st
from PIL import Image
import os
from datetime import datetime
import numpy as np
import torch
from segment_anything import sam_model_registry
import torch.nn.functional as F
import cv2

# --- Folder lưu ảnh ---
output_folder = "assets"
os.makedirs(output_folder, exist_ok=True)

# --- Page config ---
st.set_page_config(
    page_title="SAM Segmentation Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS nâng cao ---
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
}
.stImage img {
    border-radius: 12px;
    box-shadow: 4px 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.stImage img:hover {
    transform: scale(1.05);
}
.stButton>button {
    background-color: #4B0082;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}
.stButton>button:hover {
    background-color: #6A0DAD;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png","jpeg"])
run_btn = st.sidebar.button("Run Segmentation")

# --- SAM real ---
@st.cache_resource
def load_sam_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # SAM gốc
    sam_pre = sam_model_registry['vit_b'](checkpoint='models/sam_vit_b_01ec64.pth')
    sam_pre.eval()
    sam_pre = sam_pre.to(device)
    # SAM fine-tune
    sam_ft = sam_model_registry['vit_b'](checkpoint='models/sam_vit_b_01ec64.pth')
    checkpoint = torch.load('sam_finetuned_best.pth', map_location=device)
    sam_ft.load_state_dict(checkpoint['model_state_dict'])
    sam_ft.eval()
    sam_ft = sam_ft.to(device)
    return sam_pre, sam_ft, device

sam_pre, sam_ft, device = load_sam_models()

def samProcess(image, model, device):
    img = cv2.resize(image, (1024, 1024))
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        image_embeddings = model.image_encoder(img_tensor)
        sparse_embeddings = model.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings[0],
            dense_prompt_embeddings=sparse_embeddings[1],
            multimask_output=False,
        )
        mask_pred = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
        mask_pred = (mask_pred.sigmoid() > 0.5).float().cpu().numpy()[0,0]
        score = float(iou_predictions.cpu().numpy()[0,0])
        mask = (mask_pred * 255).astype('uint8')
    return mask, score
# --- Tabs ---
tab1, tab2 = st.tabs(["Upload & Segmentation", "Gallery"])

# --- Tab 1: Upload & Segmentation ---
with tab1:
    st.header("📤 Upload Image")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        if run_btn:
            # Gọi SAM gốc
            mask_pre, score_pre = samProcess(np.array(image), sam_pre, device)
            # Gọi SAM fine-tune
            mask_ft, score_ft = samProcess(np.array(image), sam_ft, device)

            # Lưu ảnh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_part = uploaded_file.name.rsplit(".", 1)[0]
            ext = uploaded_file.name.rsplit(".", 1)[-1]
            filename_pre = f"{timestamp}_{name_part}_SAMpre.{ext}"
            filename_ft = f"{timestamp}_{name_part}_SAMft.{ext}"
            output_path_pre = os.path.join(output_folder, filename_pre)
            output_path_ft = os.path.join(output_folder, filename_ft)
            Image.fromarray(mask_pre).save(output_path_pre)
            Image.fromarray(mask_ft).save(output_path_ft)

            st.success(f"Segmentation done! Saved as {filename_pre} & {filename_ft}")

            # Hiển thị song song
            st.subheader("So sánh kết quả SAM gốc và Fine-tune")
            col1, col2 = st.columns(2)
            with col1:
                st.image(output_path_pre, caption=f"SAM gốc (score={score_pre:.2f})", use_container_width=True)
                with st.expander("Chi tiết SAM gốc"):
                    st.write(f"Score: {score_pre:.2f}")
                    st.write(f"Filename: {filename_pre}")
            with col2:
                st.image(output_path_ft, caption=f"SAM fine-tune (score={score_ft:.2f})", use_container_width=True)
                with st.expander("Chi tiết SAM fine-tune"):
                    st.write(f"Score: {score_ft:.2f}")
                    st.write(f"Filename: {filename_ft}")

# --- Tab 2: Gallery ---
with tab2:
    st.header("🖼 Gallery")
    all_images = [f for f in os.listdir(output_folder) if f.lower().endswith((".png","jpg","jpeg"))]
    
    if all_images:
        cols = st.columns(3)
        for idx, img_name in enumerate(all_images):
            img_path = os.path.join(output_folder, img_name)
            with cols[idx % 3]:
                st.image(img_path, caption=f"{img_name} - Segmented by SAM", use_container_width=True)
                with st.expander("Details"):
                    img = Image.open(img_path)
                    st.write(f"Filename: {img_name}")
                    st.write(f"Size: {img.size[0]} x {img.size[1]}")
                    st.write(f"Mode: {img.mode}")
    else:
        st.info("No images in assets folder yet.")
