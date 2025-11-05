import streamlit as st
from PIL import Image

st.title("Demo Upload và Hiển Thị Ảnh")

# Tạo uploader, chỉ chấp nhận file ảnh
uploaded_file = st.file_uploader("Chọn 1 ảnh để hiển thị", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mở ảnh bằng PIL
    img = Image.open(uploaded_file)
    
    # Hiển thị ảnh
    st.image(img, caption="Ảnh bạn vừa upload", use_column_width=True)
