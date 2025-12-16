# Self-Supervised Learning (SSL) - Tổng quan

## 1. Khái niệm và Nguyên lý

**Self-Supervised Learning (SSL)** là phương pháp học máy trong đó mô hình tự tạo nhãn từ dữ liệu chưa được gán nhãn, thay vì cần nhãn do con người cung cấp.

### Nguyên lý cốt lõi:

SSL hoạt động dựa trên **pretext tasks** (nhiệm vụ giả tạo) để học representations hữu ích:
- Mô hình học bằng cách giải quyết các nhiệm vụ được tự động tạo từ dữ liệu
- Representations học được sau đó được sử dụng cho các downstream tasks
- Giảm đáng kể nhu cầu dữ liệu có nhãn

### Các phương pháp chính:

1. **Contrastive Learning**: Học bằng cách phân biệt các mẫu tương tự và khác biệt
2. **Predictive Learning**: Dự đoán một phần dữ liệu từ phần còn lại
3. **Generative Learning**: Tái tạo dữ liệu gốc từ dữ liệu bị biến đổi

## 2. Phương pháp Toán học

### 2.1 Contrastive Learning

**Công thức InfoNCE Loss** (sử dụng trong SimCLR, MoCo):

```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

Trong đó:
- `z_i, z_j`: representations của cặp positive (augmented từ cùng một mẫu)
- `z_k`: representations của negative samples
- `sim(u,v)`: hàm similarity (thường dùng cosine similarity)
- `τ`: temperature parameter

**Cosine Similarity**:
```
sim(u,v) = (u·v) / (||u|| ||v||)
```

### 2.2 Masked Language Modeling (BERT)

**Hàm mục tiêu**:
```
L_MLM = -Σ log P(x_masked | x_context)
```

Mô hình dự đoán các token bị mask dựa trên context xung quanh.

### 2.3 Vision Transformer (MAE - Masked Autoencoder)

**Reconstruction Loss**:
```
L = ||x - f_θ(mask(x))||²
```

Trong đó:
- `x`: hình ảnh gốc
- `mask(x)`: hình ảnh với một số patches bị mask
- `f_θ`: mô hình encoder-decoder

### 2.4 Momentum Encoder (MoCo)

**Cập nhật momentum**:
```
θ_k ← m·θ_k + (1-m)·θ_q
```

Trong đó:
- `θ_q`: parameters của query encoder
- `θ_k`: parameters của key encoder
- `m`: momentum coefficient (thường là 0.999)

## 3. Các Giải thuật Tin học Chính

### 3.1 SimCLR (Simple Framework for Contrastive Learning)

**Thuật toán**:
```
1. Lấy minibatch N samples
2. Tạo 2N augmented samples (mỗi sample tạo 2 views)
3. Encode tất cả samples → representations
4. Tính contrastive loss cho các positive pairs
5. Backpropagation và cập nhật weights
```

**Augmentations phổ biến**:
- Random crop and resize
- Color distortion
- Gaussian blur
- Random flip

### 3.2 MoCo (Momentum Contrast)

**Thuật toán**:
```
1. Maintain một queue chứa K negative samples
2. Query encoder xử lý current batch
3. Key encoder (momentum update) xử lý samples cho queue
4. Tính contrastive loss với queue
5. Enqueue new keys, dequeue oldest keys
```

**Ưu điểm**: Queue cho phép số lượng negative samples lớn mà không cần batch size lớn.

### 3.3 BYOL (Bootstrap Your Own Latent)

**Thuật toán** (không cần negative samples):
```
1. Online network: encoder + projector + predictor
2. Target network: encoder + projector (momentum update)
3. Loss: MSE giữa prediction và target projection
4. Asymmetric architecture ngăn collapse
```

**Loss function**:
```
L = ||q_θ(z_θ) - sg(z'_ξ)||²₂
```
(sg = stop gradient)

### 3.4 MAE (Masked Autoencoder for Images)

**Thuật toán**:
```
1. Chia ảnh thành patches
2. Random mask 75% patches
3. Encoder xử lý visible patches
4. Decoder reconstruction từ encoded + mask tokens
5. Tính reconstruction loss trên masked patches
```

### 3.5 BERT (Bidirectional Encoder Representations)

**Pretraining tasks**:
- **MLM (Masked Language Modeling)**: Mask 15% tokens, dự đoán chúng
- **NSP (Next Sentence Prediction)**: Dự đoán câu B có theo sau câu A không

## 4. Tài liệu tham khảo 

### Papers cơ bản (PDF miễn phí)

1. **SimCLR** - "A Simple Framework for Contrastive Learning of Visual Representations"
   - PDF: https://arxiv.org/pdf/2002.05709.pdf
   - Code: https://github.com/google-research/simclr

2. **MoCo v3** - "An Empirical Study of Training Self-Supervised Vision Transformers"
   - PDF: https://arxiv.org/pdf/2104.02057.pdf
   - Code: https://github.com/facebookresearch/moco-v3

### Survey Paper (Tổng quan toàn diện)

5. **"Self-supervised Learning: Generative or Contrastive"**
   - PDF: https://arxiv.org/pdf/2006.08218.pdf
   - Review xuất sắc về toàn bộ SSL methods

### Tutorial và Implementation

6. **PyTorch Lightning Bolts**
   - Link: https://github.com/Lightning-AI/lightning-bolts
   - Implementations của SimCLR, BYOL, SwAV

7. **Lightly SSL Framework**
   - Link: https://github.com/lightly-ai/lightly
   - Framework dễ sử dụng cho SSL

## 6. Ứng dụng thực tế

### Computer Vision:
- Image classification với ít dữ liệu labeled
- Object detection
- Semantic segmentation
- Medical imaging

### NLP:
- Language understanding (BERT, RoBERTa)
- Text generation (GPT series)
- Machine translation
- Question answering

### Multi-modal:
- CLIP (Vision-Language)
- Speech recognition (wav2vec 2.0)

## 7. Tips Implementation

1. **Batch size**: Contrastive methods thường cần batch size lớn (256-8192)
2. **Augmentation**: Chọn augmentations mạnh nhưng không làm mất semantic meaning
3. **Training time**: SSL thường cần pretrain lâu (100-1000 epochs)
4. **Temperature τ**: Thường trong khoảng 0.07-0.5
5. **Learning rate**: Sử dụng cosine decay với warmup

## 8. Data Augmentation Pipeline (Example)

```python
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```