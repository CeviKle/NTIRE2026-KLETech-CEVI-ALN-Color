import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from train_dataset import ALNTrainDataset
from model import final_net


# =========================================
# Settings
# =========================================

TRAIN_DIR = "/NTIRE2026/C3_ALN_Color/Train"

EPOCHS = 120
BATCH_SIZE = 4
LR = 1e-4

CHECKPOINT_DIR = "NewCheckpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================
# PSNR
# =========================================

def calc_psnr(sr, hr):

    mse = F.mse_loss(sr, hr)

    if mse == 0:
        return 100

    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# =========================================
# Charbonnier Loss
# =========================================

class CharbonnierLoss(torch.nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


# =========================================
# Multiscale Loss
# =========================================

def multiscale_loss(pred, gt):

    loss = F.l1_loss(pred, gt)

    pred_half = F.interpolate(pred, scale_factor=0.5, mode="bilinear", align_corners=False)
    gt_half = F.interpolate(gt, scale_factor=0.5, mode="bilinear", align_corners=False)

    loss += F.l1_loss(pred_half, gt_half)

    pred_quarter = F.interpolate(pred, scale_factor=0.25, mode="bilinear", align_corners=False)
    gt_quarter = F.interpolate(gt, scale_factor=0.25, mode="bilinear", align_corners=False)

    loss += F.l1_loss(pred_quarter, gt_quarter)

    return loss


# =========================================
# Dataset
# =========================================

dataset = ALNTrainDataset(TRAIN_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# =========================================
# Model
# =========================================

model = final_net().to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

charbonnier = CharbonnierLoss()

scaler = GradScaler()


# =========================================
# Training
# =========================================

for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}")

    model.train()

    total_loss = 0
    total_psnr = 0

    for sh, cr, gt in loader:

        sh = sh.to(DEVICE)
        gt = gt.to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):

            # FIXED MODEL CALL
            stage1, stage2 = model(sh)

            loss_stage1 = charbonnier(stage1, gt)
            loss_stage2 = charbonnier(stage2, gt)

            ms_loss = multiscale_loss(stage2, gt)

            loss = 0.4 * loss_stage1 + 0.6 * ms_loss


        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        psnr = calc_psnr(stage2, gt)
        total_psnr += psnr.item()


    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)

    scheduler.step()

    print(f"Loss : {avg_loss:.4f}")
    print(f"PSNR : {avg_psnr:.2f}")


    # =========================================
    # Save model
    # =========================================

    save_path = os.path.join(CHECKPOINT_DIR, f"final_net_epoch{epoch}.pth")

    torch.save(model.state_dict(), save_path)

    print(f"[INFO] Saved checkpoint {save_path}")
