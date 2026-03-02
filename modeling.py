import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms as T

torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.backends.mkldnn.enabled = True

# =========================
# Config
# =========================
IMAGES_ROOT = Path(r"D:\school\kbs\images")
LABELS_ROOT = Path(r"D:\school\kbs\labels")

NUM_TRAIN_FOLDERS = 2
NUM_VAL_FOLDERS = 1

IMAGE_SIZE = (256, 512)   # (H, W)
BATCH_SIZE = 4
NUM_WORKERS = 0        
EPOCHS = 5
LR = 1e-3                 # good when backbone frozen

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# =========================
# Class mapping
# =========================
CSV_ROWS = [
    (0, "Traffic Sign", [220, 220, 0]),
    (1, "Building", [70, 70, 70]),
    (2, "Fence", [190, 153, 153]),
    (3, "Other", [250, 170, 160]),
    (4, "Pedestrian", [220, 20, 60]),
    (5, "Pole", [153, 153, 153]),
    (6, "Road Line", [157, 234, 50]),
    (7, "Road", [128, 64, 128]),
    (8, "Sidewalk", [244, 35, 232]),
    (9, "Vegetation", [107, 142, 35]),
    (10, "Car", [0, 0, 142]),
    (11, "Wall", [102, 102, 156]),
    (12, "Unlabeled", [0, 0, 0]),
]

NUM_CLASSES = len(CSV_ROWS)
index_to_name = {i: n for i, n, _ in CSV_ROWS}
index_to_color = {i: tuple(c) for i, _, c in CSV_ROWS}
color_to_index = {tuple(c): i for i, _, c in CSV_ROWS}
UNLABELED_IDX = color_to_index[(0, 0, 0)]


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, pairs, image_size=(512, 1024)):
        self.pairs = pairs
        self.image_size = image_size

        self.img_tf = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.lab_resize = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

        # packed RGB -> class idx
        self.packed_map = {}
        for rgb, idx in color_to_index.items():
            pr = (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]
            self.packed_map[pr] = idx

        # CACHE: label path -> mask tensor (speeds CPU)
        self.mask_cache = {}

    def __len__(self):
        return len(self.pairs)

    def rgb_label_to_mask(self, label_pil: Image.Image) -> torch.Tensor:
        arr = np.array(label_pil, dtype=np.uint8)  # HxWx3
        packed = (
            (arr[..., 0].astype(np.int32) << 16)
            + (arr[..., 1].astype(np.int32) << 8)
            + arr[..., 2].astype(np.int32)
        )

        mask = np.full(packed.shape, UNLABELED_IDX, dtype=np.int64)
        for u in np.unique(packed):
            mask[packed == u] = self.packed_map.get(int(u), UNLABELED_IDX)

        return torch.from_numpy(mask)  # HxW long

    def __getitem__(self, i):
        img_path, lab_path = self.pairs[i]

        img = Image.open(img_path).convert("RGB")
        img_t = self.img_tf(img)

        key = str(lab_path)
        if key in self.mask_cache:
            mask = self.mask_cache[key]
        else:
            lab = Image.open(lab_path).convert("RGB")
            lab_r = self.lab_resize(lab)
            mask = self.rgb_label_to_mask(lab_r)
            self.mask_cache[key] = mask

        return img_t, mask


# =========================
# Helpers
# =========================
def list_subfolders(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def gather_pairs(images_root: Path, labels_root: Path, folder_names):
    pairs = []
    for fname in folder_names:
        img_dir = images_root / fname
        lab_dir = labels_root / fname

        label_map = {}
        for lp in lab_dir.rglob("*"):
            if lp.is_file() and lp.suffix.lower() in IMG_EXTS:
                rel = lp.relative_to(lab_dir)
                label_map[str(rel)] = lp

        for ip in img_dir.rglob("*"):
            if ip.is_file() and ip.suffix.lower() in IMG_EXTS:
                rel = ip.relative_to(img_dir)
                key = str(rel)
                if key in label_map:
                    pairs.append((ip, label_map[key]))
    return pairs


def make_model(num_classes: int):
    model = torchvision.models.segmentation.fcn_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model


def mask_to_color(mask_hw: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    for idx, color in index_to_color.items():
        out[mask_hw == idx] = np.array(color, dtype=np.uint8)
    return out


def format_seconds(s: float) -> str:
    s = max(0, int(s))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def save_side_by_side(img1: Image.Image, img2: Image.Image, out_path: str):
    w1, h1 = img1.size
    w2, h2 = img2.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))
    canvas.save(out_path)


@torch.no_grad()
def predict_label_color_image(model, device, image_path: str, image_size):
    model.eval()
    img_pil = Image.open(image_path).convert("RGB")

    img_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    inp = img_tf(img_pil).unsqueeze(0).to(device)
    out = model(inp)["out"]
    pred = out.argmax(dim=1)[0].cpu().numpy()
    colored = mask_to_color(pred)

    orig_resized = img_pil.resize((image_size[1], image_size[0]))
    return orig_resized, Image.fromarray(colored)


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)["out"]
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_metrics(model, device, loader, num_classes: int, ignore_index: int | None = None):
    """
    Returns:
      pixel_acc (float)
      mean_iou (float)
      per_class_iou (np.array shape [C])
    """
    model.eval()

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for imgs, gt in loader:
        imgs = imgs.to(device)
        gt = gt.to(device)  # BxHxW

        out = model(imgs)["out"]               # BxCxHxW
        pred = out.argmax(dim=1)               # BxHxW

        if ignore_index is not None:
            valid = (gt != ignore_index)
            gt = gt[valid]
            pred = pred[valid]
        else:
            gt = gt.reshape(-1)
            pred = pred.reshape(-1)

        k = (gt * num_classes + pred).to(torch.int64)
        bincount = torch.bincount(k, minlength=num_classes * num_classes)
        conf += bincount.reshape(num_classes, num_classes).cpu()

    correct = torch.diag(conf).sum().item()
    total = conf.sum().item()
    pixel_acc = correct / total if total > 0 else 0.0

    tp = torch.diag(conf).to(torch.float64)
    fp = conf.sum(dim=0).to(torch.float64) - tp
    fn = conf.sum(dim=1).to(torch.float64) - tp
    denom = tp + fp + fn

    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    mean_iou = iou.mean().item()

    return pixel_acc, mean_iou, iou.numpy()

def analyze_real_photo(model, device, photo_path, image_size, alpha=0.5):
    """
    Analyzes a single real-life photo and saves an overlay comparison.
    """
    model.eval()
    
    orig_pil = Image.open(photo_path).convert("RGB")
    w, h = orig_pil.size
    
    img_tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    inp = img_tf(orig_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp)["out"]
        pred = out.argmax(dim=1)[0].cpu().numpy()
    
    colored_mask = mask_to_color(pred)
    mask_pil = Image.fromarray(colored_mask).resize((w, h), resample=Image.NEAREST)
    
    blended = Image.blend(orig_pil, mask_pil, alpha=alpha)
    
    return orig_pil, mask_pil, blended


def main():
    assert IMAGES_ROOT.exists()
    assert LABELS_ROOT.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    img_names = {p.name for p in list_subfolders(IMAGES_ROOT)}
    lab_names = {p.name for p in list_subfolders(LABELS_ROOT)}
    common = sorted(list(img_names & lab_names))

    need = NUM_TRAIN_FOLDERS + NUM_VAL_FOLDERS
    assert len(common) >= need, f"Need {need} matching subfolders, found {len(common)}."

    train_folder_names = common[:NUM_TRAIN_FOLDERS]
    val_folder_names = common[NUM_TRAIN_FOLDERS:NUM_TRAIN_FOLDERS + NUM_VAL_FOLDERS]

    train_pairs = gather_pairs(IMAGES_ROOT, LABELS_ROOT, train_folder_names)
    val_pairs = gather_pairs(IMAGES_ROOT, LABELS_ROOT, val_folder_names)
    assert train_pairs and val_pairs

    print(f"Device: {device}")
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_ds = SegDataset(train_pairs, IMAGE_SIZE)
    val_ds = SegDataset(val_pairs, IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = make_model(NUM_CLASSES).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    overall_start = time.time()
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        tr_loss = train_one_epoch(model, device, train_loader, criterion, optimizer)

        pix_acc, miou, _ = evaluate_metrics(model, device, val_loader, NUM_CLASSES, ignore_index=None)

        epoch_sec = time.time() - epoch_start
        epoch_times.append(epoch_sec)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta_sec = (EPOCHS - epoch) * avg_epoch
        elapsed = time.time() - overall_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"epoch {format_seconds(epoch_sec)} | "
            f"elapsed {format_seconds(elapsed)} | "
            f"ETA {format_seconds(eta_sec)} | "
            f"train loss {tr_loss:.4f} | "
            f"val pixel-acc {pix_acc*100:.2f}% | "
            f"val mIoU {miou*100:.2f}%"
        )

    torch.save(model.state_dict(), "best_model.pth")
    print("Saved best_model.pth")

    real_photo_path = r"D:\school\kbs\Screenshot 2026-02-12 153315.png" # <--- CHANGE THIS
    
    if os.path.exists(real_photo_path):
        print(f"\nAnalyzing real-life photo: {real_photo_path}")
        orig, mask, overlay = analyze_real_photo(model, device, real_photo_path, IMAGE_SIZE)
        
        overlay.save("real_photo_analysis.png")
        
        total_w = orig.width + mask.width + overlay.width
        max_h = max(orig.height, mask.height, overlay.height)
        combined = Image.new("RGB", (total_w, max_h))
        combined.paste(orig, (0, 0))
        combined.paste(mask, (orig.width, 0))
        combined.paste(overlay, (orig.width + mask.width, 0))
        combined.save("full_analysis_comparison.png")
        
        print("Analysis complete! Check 'real_photo_analysis.png' and 'full_analysis_comparison.png'")
    else:
        print(f"Warning: Could not find photo at {real_photo_path}")

    demo_img = str(random.choice(val_pairs)[0])
    orig, pred_label = predict_label_color_image(model, device, demo_img, IMAGE_SIZE)

    pred_label.save("predicted_label.png")
    print("Saved predicted_label.png")

    save_side_by_side(orig, pred_label, "comparison.png")
    print("Saved comparison.png")
    print("Demo image:", demo_img)

    pix_acc, miou, per_class_iou = evaluate_metrics(model, device, val_loader, NUM_CLASSES, ignore_index=None)
    print("\nPer-class IoU:")
    for c in range(NUM_CLASSES):
        print(f"{c:02d} {index_to_name[c]:12s} : {per_class_iou[c]*100:.2f}%")
    print(f"\nFinal Val Pixel-Acc: {pix_acc*100:.2f}% | Final Val mIoU: {miou*100:.2f}%")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

