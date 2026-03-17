import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def list_images_recursive(root):
    """
    Recursively list images in all subfolders,
    ignore 'Thumbs.db' and backup files '*.png~',
    and sort numerically for proper alignment with GT.
    """
    files = []

    # Sort subfolders numerically
    subfolders = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=lambda x: int(x)
    )

    for sub in subfolders:
        sub_path = os.path.join(root, sub)

        imgs = [
            f for f in os.listdir(sub_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and not f.startswith("~")
            and f != "Thumbs.db"
        ]

        imgs = sorted(imgs)  # Sort files inside folder
        files.extend([os.path.join(sub_path, f) for f in imgs])

    return files


def list_images_flat(root):
    """
    List images in a flat folder like GT,
    ignore unwanted files, and sort numerically.
    """
    files = [
        f for f in os.listdir(root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and not f.startswith("~")
        and f != "Thumbs.db"
    ]

    # Sort numerically by filename (e.g., 1.png, 2.png, ...)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    return [os.path.join(root, f) for f in files]


class ALNTrainDataset(Dataset):
    def __init__(self, train_dir, crop_size=384):
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

        # Correct paths
        self.root_gt = os.path.join(train_dir, "GT")
        self.root_sh = os.path.join(train_dir, "IN_SH")
        self.root_cr = os.path.join(train_dir, "IN_CR")

        # Load images
        self.sh_images = list_images_recursive(self.root_sh)
        self.cr_images = list_images_recursive(self.root_cr)
        self.gt_images = list_images_flat(self.root_gt)

        # Ensure dataset lengths match
        min_len = min(len(self.sh_images), len(self.cr_images), len(self.gt_images))
        self.sh_images = self.sh_images[:min_len]
        self.cr_images = self.cr_images[:min_len]
        self.gt_images = self.gt_images[:min_len]

        print(f"[INFO] Training samples: {min_len}")

        # Data augmentation
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.sh_images)

    def random_crop(self, sh, cr, gt):
        w, h = sh.size
        cs = self.crop_size

        if w <= cs or h <= cs:
            return sh, cr, gt

        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)

        sh = sh.crop((x, y, x + cs, y + cs))
        cr = cr.crop((x, y, x + cs, y + cs))
        gt = gt.crop((x, y, x + cs, y + cs))

        return sh, cr, gt

    def __getitem__(self, idx):
        sh = Image.open(self.sh_images[idx]).convert("RGB")
        cr = Image.open(self.cr_images[idx]).convert("RGB")
        gt = Image.open(self.gt_images[idx]).convert("RGB")

        # Random crop
        sh, cr, gt = self.random_crop(sh, cr, gt)

        # Ensure same augmentation
        seed = random.randint(0, 99999)

        random.seed(seed)
        sh = self.augment(sh)

        random.seed(seed)
        cr = self.augment(cr)

        random.seed(seed)
        gt = self.augment(gt)

        sh = self.to_tensor(sh)
        cr = self.to_tensor(cr)
        gt = self.to_tensor(gt)

        return sh, cr, gt