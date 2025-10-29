import streamlit as st
from PIL import Image
import numpy as np

def samProcess():
    pass

def main():
    st.title("🔍 SAM Segmentation Demo (Streamlit)")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Run SAM Segmentation"):
            # Gọi model xử lý
            mask, score = samProcess(image)
            
            # Hiển thị kết quả
            st.image(mask, caption=f"Segmentation (score={score:.2f})", use_column_width=True)

if __name__ == "__main__":
    main()
