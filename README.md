# NTIRE 2026: Ambient Lighting Normalization (ALN)

## 1. Introduction

This repository contains our implementation for the **NTIRE 2026 Ambient Lighting Normalization Challenge**, conducted as part of the CVPR 2026 Workshop. The task focuses on correcting uneven illumination in real-world images caused by ambient lighting variations such as shadows, brightness imbalance, and color distortions.

Our goal is to generate illumination-normalized images that preserve structural details while improving visual consistency.

---

## 2. Method Overview

We adopt a deep learning based **two-stage illumination normalization network (ALN-Net)** designed to learn illumination-aware features and reconstruct visually consistent outputs.

* Architecture: Encoder–Decoder (Fully Convolutional)
* Stage 1 → Illumination correction
* Stage 2 → Refinement
* Training: Supervised learning using paired dataset
* Loss Functions:
  * MSE Loss
  * Charbonnier Loss
  * Color Consistency Loss
* Total training duration: **~120 epochs**

---

## 3. Installation

```
git clone <repo-url>
cd NTIRE_ALN
pip install -r requirements.txt
```

---

## 4. GPU Usage

* **GPU 0 → Training**
* **GPU 1 → Validation / Inference**

This setup allows training and evaluation to run independently.

---

## 5. Training

### 5.1 Start Training

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 5.2 Resume Training

```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --resume NewCheckpoints/last_checkpoint.pth
```

---

## 6. Dataset

We use the **official NTIRE 2026 Ambient Lighting Normalization dataset**.

Input:

* Shadow images (**IN_SH**)
* Color distorted images (**IN_CR**)

Target:

* Ground truth illumination-normalized images (**GT**)

Dataset structure:

```
C3_ALN_Color/
├── Train/
├── cl3an_val/
├── NTIRE26-cl3an-test-in/
```

---

## 7. Validation

Validation is performed on:

```
cl3an_val/
```

Run validation:

```
CUDA_VISIBLE_DEVICES=1 python inference.py
```

---

## 8. Inference Pipeline

During inference, the trained model processes the input image and generates an illumination-normalized output.

```
Input Image → ALN-Net → Normalized Output
```

The model is fully convolutional and supports variable image resolutions.

---

## 9. Output Location

Outputs are saved in:

```
runs/
```

Check outputs:

```
find runs/ -type f | grep -iE "\.png$|\.jpg$"
```

---

## 10. Submission

Prepare:

* Validation outputs
* Test outputs

Ensure:

* Correct file naming
* No missing images
* Proper folder structure

---

## 11. References

* Bao et al., Frequency-Prior Enhanced Ambient Lighting Normalization (CVPRW 2025)
* Vasluianu et al., Towards Image Ambient Lighting Normalization (ECCV 2024)

---

## 13. Acknowledgement

This work is inspired by recent NTIRE challenge solutions and research on ambient lighting normalization.
