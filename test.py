import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import autocast

from model import final_net


# =====================================================
# Dataset
# =====================================================

class TestDataset(Dataset):
    def __init__(self, input_root):
        self.to_tensor = T.ToTensor()

        # Only load PNG images, ignore Thumbs.db
        self.images = sorted(
            [f for f in glob.glob(os.path.join(input_root, "*.png")) if "Thumbs" not in f]
        )

        print("Total test images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        name = os.path.basename(path)

        img = Image.open(path).convert("RGB")
        img = self.to_tensor(img)

        return img, name


# =====================================================
# Model Forward
# =====================================================

@torch.no_grad()
def run_model(model, x, device):
    with autocast(device_type=device.type if hasattr(device, "type") else "cuda"):
        _, out = model(x)
    return out


# =====================================================
# Inference
# =====================================================

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model
    model = final_net().to(device)
    checkpoint_path = "NewCheckpoints/final_net_epoch2.pth"
    print("Loading checkpoint:", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both dict checkpoint and direct state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # Dataset folder (flat)
    dataset = TestDataset("/NTIRE2026/C3_ALN_Color/NTIRE26-cl3an-test-in")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs("Tested_results", exist_ok=True)

    for count, (img, name) in enumerate(loader, 1):
        img = img.to(device)

        _, _, h, w = img.shape

        # pad to multiple of 32
        pad = 32
        h_pad = (pad - h % pad) % pad
        w_pad = (pad - w % pad) % pad
        img = F.pad(img, (0, w_pad, 0, h_pad), mode="reflect")

        out = run_model(model, img, device)

        # remove padding
        out = out[:, :, :h, :w]
        out = torch.clamp(out, 0, 1)

        out_img = (out.detach().cpu() * 255).round().byte()
        out_img = out_img.squeeze(0).permute(1, 2, 0).numpy()

        save_path = os.path.join("Tested_results", name[0])
        Image.fromarray(out_img).save(save_path)

        if count % 10 == 0:
            print(f"Processed {count} images")

    print("\nInference completed")
    print("Total images saved:", count)


if __name__ == "__main__":
    inference()