import os
from pathlib import Path
import time
import random
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("")

# =========================
# 0) Path
# =========================
EXCEL_DIR = r""
BASE_IMG_DIR = r""

# (stage: image subdir)
stage_to_subdir = {
    0:  r"H_pluvialis_0day\crops\plu",
    2:  r"H_pluvialis_2day\crops\plu",
    5:  r"H_pluvialis_5day\crops\plu",
    10: r"H_pluvialis_10day\crops\plu",
    25: r"H_pluvialis_25day\crops\plu",
}

stage_map = {
    "H_pluvialis_0day.xlsx": 0,
    "H_pluvialis_2day.xlsx": 2,
    "H_pluvialis_5day.xlsx": 5,
    "H_pluvialis_10day.xlsx": 10,
    "H_pluvialis_25day.xlsx": 25,
}

SEED = 42
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
NUM_WORKERS = 0  
PATIENCE = 5     
MODEL_DIR = Path("./_cnn_models"); MODEL_DIR.mkdir(exist_ok=True)
BEST_PATH = MODEL_DIR / "hpluvialis_resnet18_best.pt"

# =========================
# 1) Import
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ device={device} | torch={torch.__version__} | torchvision={torchvision.__version__}")

def build_img_path(filename: str, stage_val: int) -> str:
    sub = stage_to_subdir.get(int(stage_val))
    if sub is None: return None
    return os.path.join(BASE_IMG_DIR, sub, str(filename))

# =========================
# 2) Dataframe load
# =========================
dfs = []
for fn in os.listdir(EXCEL_DIR):
    if fn.endswith(".xlsx") and fn in stage_map:
        fp = os.path.join(EXCEL_DIR, fn)
        df = pd.read_excel(fp)
        if "Filename" not in df.columns:
            raise KeyError(f"")
        df["stage"] = stage_map[fn]
        dfs.append(df)

if not dfs:
    raise RuntimeError("")

df_all = pd.concat(dfs, ignore_index=True)

df_all["img_path"] = df_all.apply(lambda r: build_img_path(r["Filename"], r["stage"]), axis=1)
valid = df_all["img_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))
n_bad = (~valid).sum()
if n_bad:
    print(f"")
df_all = df_all.loc[valid].reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(df_all["stage"].values)  # class index
class_values = le.inverse_transform(np.arange(len(le.classes_)))  # [0,2,5,10,25]
class_names = [f"stage{int(v)}" for v in class_values]
num_classes = len(class_names)
print("🎯 Classes:", dict(zip(range(num_classes), class_names)))

# Stratified split
idx = np.arange(len(df_all))
train_idx, val_idx = train_test_split(
    idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
)
df_train = df_all.iloc[train_idx].reset_index(drop=True)
df_val   = df_all.iloc[val_idx].reset_index(drop=True)
y_train  = y[train_idx]
y_val    = y[val_idx]
print(f"📦 Train={len(df_train)} | Val={len(df_val)}")

# =========================
# 3) Dataset / Transform
# =========================
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

val_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

class ImgClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: np.ndarray, transform):
        self.df = df
        self.labels = labels
        self.t = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        p = self.df.at[i, "img_path"]
        y = self.labels[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0,0,0))
        x = self.t(img)
        return x, y

train_ds = ImgClsDataset(df_train, y_train, train_tf)
val_ds   = ImgClsDataset(df_val,   y_val,   val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))

# =========================
# 4) Modeling
# =========================
def build_model(num_classes: int):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(pretrained=True)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net

model = build_model(num_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

# =========================
# 5) Train/val
# =========================
def run_epoch(dataloader, train: bool):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    pbar = tqdm(dataloader, leave=False)
    for x, t in pbar:
        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logits = model(x)
            loss = criterion(logits, t)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_correct += (pred == t).sum().item()
            total += t.size(0)
            total_loss += loss.item() * t.size(0)
            all_preds.append(pred.detach().cpu())
            all_targets.append(t.detach().cpu())

        pbar.set_description(f"{'Train' if train else 'Val'} loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, acc, all_preds, all_targets

best_val_acc = -1.0
best_state = None
epochs_no_improve = 0
t0 = time.time()

for epoch in range(1, EPOCHS+1):
    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
    tr_loss, tr_acc, _, _ = run_epoch(train_dl, train=True)
    val_loss, val_acc, vpreds, vtargets = run_epoch(val_dl, train=False)
    scheduler.step()

    print(f"📉 Train  loss={tr_loss:.4f} | acc={tr_acc:.4f}")
    print(f"✅  Val   loss={val_loss:.4f} | acc={val_acc:.4f}")

    # 체크포인트 & 얼리스톱
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": best_val_acc,
            "val_loss": val_loss,
            "classes": class_names,
        }
        torch.save(best_state, BEST_PATH)
        print(f"💾 Best updated (acc={best_val_acc:.4f}) → {BEST_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⏳ No improvement ({epochs_no_improve}/{PATIENCE})")
        if epochs_no_improve >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")
print(f"🏁 Best Val Acc: {best_val_acc:.4f} (saved at {BEST_PATH})")

# =========================
# 6) Evaluation
# =========================
ckpt = torch.load(BEST_PATH, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for x, t in tqdm(val_dl, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(t.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

print("\n🧾 Classification Report (Val):")
print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(all_targets, all_preds)
print("\n📊 Confusion Matrix (Val):\n", cm)


def predict_paths(img_paths, batch_size=64):
    tfm = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    class _TmpDS(Dataset):
        def __init__(self, ps): self.ps=list(ps)
        def __len__(self): return len(self.ps)
        def __getitem__(self, i):
            p=self.ps[i]
            try: im=Image.open(p).convert("RGB")
            except Exception: im=Image.new("RGB",(224,224),(0,0,0))
            return tfm(im), p

    dl = DataLoader(_TmpDS(img_paths), batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
    preds, names, probs = [], [], []
    sm = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            x=x.to(device, non_blocking=True)
            logits = model(x)
            p = sm(logits)
            conf, idx = torch.max(p, dim=1)
            preds += idx.cpu().tolist()
            probs += conf.cpu().tolist()
    names = [class_names[i] for i in preds]
    return preds, names, probs

if __name__ == "__main__":
    print("\n✅ Pure CNN training finished. Best model saved at:", BEST_PATH)
