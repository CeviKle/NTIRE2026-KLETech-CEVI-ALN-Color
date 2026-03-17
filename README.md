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
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-ALN-Color.git
cd NTIRE2026-KLETech-CEVI-ALN-Color
pip install -r requirements.txt
```

---

## 4. Pretrained Model Weights

Download the trained model weights from the following link:

**Model Weights:**  
https://drive.google.com/file/d/1hP7AXxvBXCk4w1nV3prxCH0D3mVcNv5n/view?usp=drive_link

After downloading, place the weights inside:

```
NewCheckpoints/
```

---

## 5. GPU Usage

* **GPU 0 → Training**
* **GPU 1 → Validation / Inference**

This setup allows training and evaluation to run independently.

---

## 6. Training

### 6.1 Start Training

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 6.2 Resume Training

```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --resume NewCheckpoints/last_checkpoint.pth
```

---

## 7. Dataset

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

## 8. Validation

Validation is performed on:

```
cl3an_val/
```

Run validation:

```
CUDA_VISIBLE_DEVICES=1 python inference.py
```

---

## 9. Inference Pipeline

During inference, the trained model processes the input image and generates an illumination-normalized output.

```
Input Image → ALN-Net → Normalized Output
```

The model is fully convolutional and supports variable image resolutions.

---

## 10. Results

Download the generated results used for submission:

**Results Download Link:**  
https://drive.google.com/file/d/1hOPNNr6sxB4vVn_NWBfvyS8BXDd10-3e/view?usp=drive_link

The results include all required outputs for evaluation.

---

## 11. Output Location

Outputs are saved in:

```
runs/
```

Check outputs:

```
find runs/ -type f | grep -iE "\.png$|\.jpg$"
```

---

## 12. Submission

Prepare:

* Validation outputs
* Test outputs

Ensure:

* Correct file naming
* No missing images
* Proper folder structure

---

## 13. References

* Bao et al., Frequency-Prior Enhanced Ambient Lighting Normalization (CVPRW 2025)
* Vasluianu et al., Towards Image Ambient Lighting Normalization (ECCV 2024)

---

## 14. Acknowledgement

This work is inspired by recent NTIRE challenge solutions and research on ambient lighting normalization.
