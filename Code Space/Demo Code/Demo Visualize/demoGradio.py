import gradio as gr
from PIL import Image

# Hàm xử lý ảnh (chỉ hiển thị lại)
def show_image(img):
    return img

# Tạo interface
iface = gr.Interface(
    fn=show_image,          # hàm xử lý
    inputs=gr.Image(type="pil"),  # input là ảnh PIL
    outputs=gr.Image(type="pil"), # output cũng là ảnh PIL
    title="Demo Upload và Hiển Thị Ảnh",
    description="Upload ảnh để xem ngay!"
)

# Chạy app
iface.launch(share=True)
