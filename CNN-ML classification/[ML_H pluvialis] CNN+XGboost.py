import os
from pathlib import Path
import time
import gc

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  

# Sklearn / ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# PyTorch / CNN
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# tqdm 진행률
try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("")

# =========================
# 0) Import
# =========================
EXCEL_DIR = r""
BASE_IMG_DIR = r""

stage_map = {
    "H_pluvialis_0day.xlsx": 0,
    "H_pluvialis_2day.xlsx": 2,
    "H_pluvialis_5day.xlsx": 5,
    "H_pluvialis_10day.xlsx": 10,
    "H_pluvialis_25day.xlsx": 25,
}

stage_to_subdir = {
    0:  r"H_pluvialis_0day\crops\plu",
    2:  r"H_pluvialis_2day\crops\plu",
    5:  r"H_pluvialis_5day\crops\plu",
    10: r"H_pluvialis_10day\crops\plu",
    25: r"H_pluvialis_25day\crops\plu",
}

PCA_COMPONENTS = 128          
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
NUM_WORKERS = 0              

CACHE_DIR = Path("./_cache"); CACHE_DIR.mkdir(exist_ok=True)
CNN_CACHE = CACHE_DIR / "cnn_feats_hpluvialis_resnet18.npy"

# SHAP 
ENABLE_SHAP = False and HAS_SHAP

# =========================
# 1) Sampling
# =========================
dfs = []
for fn in os.listdir(EXCEL_DIR):
    if fn.endswith(".xlsx") and fn in stage_map:
        fp = os.path.join(EXCEL_DIR, fn)
        df = pd.read_excel(fp)
        df["stage"] = stage_map[fn]
        if "Filename" not in df.columns:
            raise KeyError("")       
        dfs.append(df.sample(n=min(10000, len(df)), random_state=RANDOM_STATE))

combined_df = pd.concat(dfs, ignore_index=True)

X_num = combined_df.drop(columns=["Filename", "stage"], errors="ignore").copy()
need_cols = [
    "chl_a_pg_total", "Size(μm²)", "Major Axis", "Minor Axis",
    "Ellipse Mean Gray Scale", "chl_a_pg_DOF", "Eccentricity",
    "Astaxanthin", "Integrated Density", "Perimeter(μm)",
    "Equivalent Diameter(μm)", "Solidity", "Circularity", "Aspect Ratio", "Rectangularity",
    "Deformation Index", "Intensity Std", "Intensity CoV"
]
for c in need_cols:
    if c not in X_num.columns:
        raise KeyError(f"필수 컬럼 누락: {c}")

X_num["chl_per_area"]      = X_num["chl_a_pg_total"] / (X_num["Size(μm²)"] + 1e-6)
X_num["aspect_ratio"]      = X_num["Major Axis"] / (X_num["Minor Axis"] + 1e-6)
X_num["chl_per_gray"]      = X_num["chl_a_pg_total"] / (X_num["Ellipse Mean Gray Scale"] + 1e-6)
X_num["chl_ratio_dof"]     = X_num["chl_a_pg_DOF"] / (X_num["chl_a_pg_total"] + 1e-6)
X_num["eccentricity_squared"] = X_num["Eccentricity"] ** 2
X_num["chl_area_x_ratio"]  = X_num["chl_per_area"] * X_num["aspect_ratio"]
X_num["gray_x_chl_ratio"]  = X_num["Ellipse Mean Gray Scale"] * X_num["chl_ratio_dof"]

valid_idx = X_num.dropna().index
X_num = X_num.loc[valid_idx].reset_index(drop=True)
combined_df = combined_df.loc[valid_idx].reset_index(drop=True)

# =========================
# 2) Image path
# =========================
def build_img_path(row):
    sub = stage_to_subdir.get(int(row["stage"]))
    if sub is None:
        return None
    return os.path.join(BASE_IMG_DIR, sub, str(row["Filename"]))

combined_df["img_path"] = combined_df.apply(build_img_path, axis=1)

def _ok(p):
    return isinstance(p, str) and os.path.isfile(p)

valid = combined_df["img_path"].apply(_ok)
n_bad = (~valid).sum()
if n_bad:
    print(f"")
combined_df = combined_df.loc[valid].reset_index(drop=True)
X_num = X_num.loc[combined_df.index].reset_index(drop=True)

le = LabelEncoder()
y_raw = combined_df["stage"].values
y = le.fit_transform(y_raw)
class_names = [f"stage{int(s)}" for s in le.inverse_transform(np.arange(len(le.classes_)))]

# =========================
# 3) CNN embedding
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}, torch={torch.__version__}, torchvision={torchvision.__version__}")

try:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    tfm = weights.transforms()
    mean, std = tfm.mean, tfm.std
except Exception:
    backbone = models.resnet18(pretrained=True)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

backbone.fc = torch.nn.Identity()  
backbone.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

class ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths); self.t = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0,0,0))  
        return self.t(img)

def extract_cnn_features(paths, batch_size=64):
    ds = ImgDataset(paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
    feats = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs in tqdm(dl, total=len(dl), unit="batch",
                         desc=f"🔎 Extracting CNN features (batch={batch_size})",
                         dynamic_ncols=True, leave=True):
            if device.type == 'cuda':
                imgs = imgs.to(device, non_blocking=True)
            f = backbone(imgs)              # [B, 512]
            feats.append(torch.flatten(f, 1).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    print(f"⏱️ CNN feature extraction done: {feats.shape} in {time.perf_counter()-t0:.1f}s")
    return feats

if CNN_CACHE.exists():
    X_cnn = np.load(CNN_CACHE)
    if X_cnn.shape[0] != len(combined_df):
        print("")
        X_cnn = extract_cnn_features(combined_df["img_path"].tolist(), BATCH_SIZE)
        np.save(CNN_CACHE, X_cnn)
else:
    print("")
    X_cnn = extract_cnn_features(combined_df["img_path"].tolist(), BATCH_SIZE)
    np.save(CNN_CACHE, X_cnn)

del backbone
gc.collect()
try:
    torch.cuda.empty_cache()
except Exception:
    pass

print("✅ CNN feature shape:", X_cnn.shape)  # [N, 512]

if PCA_COMPONENTS:
    print(f"🔻 PCA: 512 → {PCA_COMPONENTS}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_cnn = pca.fit_transform(X_cnn)       # [N, PCA_COMPONENTS]

# =========================
# 4) Feature scaling
# =========================
scaler_tab = StandardScaler()
X_tab = scaler_tab.fit_transform(X_num.values)        
X_all = np.hstack([X_tab, X_cnn])                     

idx_all = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx_all, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

datasets = {
    "Tabular only":  (X_tab[train_idx],  X_tab[test_idx]),
    "Tabular + CNN": (X_all[train_idx],  X_all[test_idx]),
}
y_train = y[train_idx]
y_test  = y[test_idx]

# =========================
# 5) XGBoost tuning
# =========================
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

def make_xgb():    
    return xgb.XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=len(np.unique(y)),
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        random_state=RANDOM_STATE
    )

print("\n🎯 Tuning XGBoost on [Tabular + CNN] ...")
xgb_model = make_xgb()
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20, cv=3, scoring='accuracy',
    verbose=2, random_state=RANDOM_STATE, n_jobs=-1
)

X_tr_all, X_te_all = datasets["Tabular + CNN"]
random_search.fit(X_tr_all, y_train)
best_params = random_search.best_params_
print("🏆 Best Params (Tab+CNN):", best_params)
print("🎯 Best CV Accuracy (Tab+CNN):", random_search.best_score_)

best_model_tabcnn = make_xgb().set_params(**best_params)
best_model_tabcnn.fit(X_tr_all, y_train)
y_pred_tabcnn = best_model_tabcnn.predict(X_te_all)
print("🎯 Test Accuracy (Tab+CNN, XGBoost):", accuracy_score(y_test, y_pred_tabcnn))

cm = confusion_matrix(y_test, y_pred_tabcnn)
target_names = [f"stage{int(s)}" for s in le.inverse_transform(np.arange(len(le.classes_)))]
print("\n📊 Confusion Matrix (Tab+CNN, XGB):\n", cm)
print("\n🧾 Classification Report (Tab+CNN, XGB):\n",
      classification_report(y_test, y_pred_tabcnn, target_names=target_names, zero_division=0))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (Tabular + CNN → XGBoost)")
plt.tight_layout(); plt.show()

# =========================
# 6) ML evaluation
# =========================
from collections import OrderedDict

RESULTS_XLSX = r""
Path(os.path.dirname(RESULTS_XLSX)).mkdir(parents=True, exist_ok=True)

def make_7_models():
    return OrderedDict({
        "XGBoost":              make_xgb().set_params(**best_params),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Random Forest":        RandomForestClassifier(random_state=RANDOM_STATE),
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)":            SVC(kernel="rbf", probability=False, random_state=RANDOM_STATE),
        "K-Nearest Neighbors":  KNeighborsClassifier(),
        "Decision Tree":        DecisionTreeClassifier(random_state=RANDOM_STATE),
    })

def run_models_and_metrics(X_tr, X_te, y_tr, y_te, scenario_name):
    models = make_7_models()
    target_names_all = [f"stage{int(s)}" for s in le.inverse_transform(np.arange(len(le.classes_)))]
    summary_rows, per_class_rows = [], []

    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        rep = classification_report(
            y_te, y_pred,
            target_names=target_names_all,
            output_dict=True, zero_division=0
        )

        summary_rows.append({
            "Model": name,
            "accuracy": round(rep["accuracy"], 4),
            "macro_precision": round(rep["macro avg"]["precision"], 4),
            "macro_recall": round(rep["macro avg"]["recall"], 4),
            "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        })

        for cls in target_names_all:
            row = rep.get(cls, {})
            per_class_rows.append({
                "Scenario": scenario_name,
                "Model": name,
                "Class": cls,
                "precision": round(row.get("precision", 0.0), 4),
                "recall": round(row.get("recall", 0.0), 4),
                "f1": round(row.get("f1-score", 0.0), 4),
                "support": int(row.get("support", 0)),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows)

(X_tr_all, X_te_all) = datasets["Tabular + CNN"]
(X_tr_tab, X_te_tab) = datasets["Tabular only"]

sum_tabcnn, pc_tabcnn = run_models_and_metrics(X_tr_all, X_te_all, y_train, y_test, "Tab+CNN")
sum_tabonly, pc_tabonly = run_models_and_metrics(X_tr_tab, X_te_tab, y_train, y_test, "TabOnly")

with pd.ExcelWriter(RESULTS_XLSX, engine="xlsxwriter") as w:
    sum_tabcnn.to_excel(w, sheet_name="Tab+CNN_summary", index=False)
    sum_tabonly.to_excel(w, sheet_name="TabOnly_summary", index=False)
    pc_tabcnn.to_excel(w, sheet_name="Tab+CNN_per_class", index=False)
    pc_tabonly.to_excel(w, sheet_name="TabOnly_per_class", index=False)

print(f"")

# =========================
# 7) SHAP
# =========================
tab_feat_names = list(X_num.columns)
cnn_feat_names = [f"cnn_{i}" for i in range(X_cnn.shape[1])]
feature_names = tab_feat_names + cnn_feat_names

booster = best_model_tabcnn.get_booster()
imp_gain = booster.get_score(importance_type='gain')

def f2name(k):
    if k.startswith('f') and k[1:].isdigit():
        idx = int(k[1:])
        return feature_names[idx] if idx < len(feature_names) else k
    return k

imp_df = pd.DataFrame([(f2name(k), v) for k, v in imp_gain.items()],
                      columns=['feature', 'gain']).sort_values('gain', ascending=False)
print("\n[Top 20 XGBoost Feature Importances by GAIN] (Tab+CNN)")
print(imp_df.head(20).to_string(index=False))

plt.figure(figsize=(8, 6))
topk = min(20, len(imp_df))
plt.barh(imp_df['feature'].head(topk)[::-1], imp_df['gain'].head(topk)[::-1])
plt.xlabel('Gain'); plt.title('Top Feature Importances (XGBoost gain, Tab+CNN)')
plt.tight_layout(); plt.show()

if ENABLE_SHAP and HAS_SHAP:
    nsample = min(2000, len(X_te_all))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_te_all), nsample, replace=False)
    X_shap = pd.DataFrame(X_te_all, columns=feature_names).iloc[idx]
    explainer = shap.TreeExplainer(best_model_tabcnn)
    shap_values = explainer.shap_values(X_shap)

    def mean_abs_shap(shap_values, X_frame):
        if isinstance(shap_values, list):
            mats = [np.abs(s) for s in shap_values]
            mean_abs = np.mean(np.stack(mats, axis=0), axis=0)
        elif getattr(shap_values, "ndim", 0) == 3:
            mean_abs = np.mean(np.abs(shap_values), axis=1)
        else:
            mean_abs = np.abs(shap_values)
        return pd.Series(mean_abs.mean(axis=0), index=X_frame.columns).sort_values(ascending=False)

    shap_rank = mean_abs_shap(shap_values, X_shap)
    print("\n[Top 20 Global Importance by mean(|SHAP|)] (Tab+CNN)")
    print(shap_rank.head(20).to_string())

    plt.figure(figsize=(8, 6))
    plt.barh(shap_rank.index[:topk][::-1], shap_rank.values[:topk][::-1])
    plt.title('Global Feature Importance by mean(|SHAP|) (Tab+CNN)')
    plt.tight_layout(); plt.show()

# =========================
# 8) Classification
# =========================
from typing import List, Dict

def _resolve_img_root(img_subdir: str) -> str:
    return img_subdir if os.path.isabs(img_subdir) else os.path.join(BASE_IMG_DIR, img_subdir)

def _build_img_path(row, img_root: str) -> str:
    return os.path.join(img_root, str(row["Filename"]))

def run_inference_one(excel_path: str, img_subdir: str, save_path: str):
    print(f"\n================ Inference for: {excel_path} ================")
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    df_new = pd.read_excel(excel_path)
    if "Filename" not in df_new.columns:
        raise KeyError("")

    def add_derived_features_inplace(df):
        if "chl_a_pg_total" in df.columns and "Size(μm²)" in df.columns:
            df["chl_per_area"] = df["chl_a_pg_total"] / (df["Size(μm²)"] + 1e-6)
        if "Major Axis" in df.columns and "Minor Axis" in df.columns:
            df["aspect_ratio"] = df["Major Axis"] / (df["Minor Axis"] + 1e-6)

    new_num = df_new.drop(columns=["Filename"], errors="ignore").copy()
    add_derived_features_inplace(new_num)

    tab_cols_train = list(X_num.columns)
    missing = [c for c in tab_cols_train if c not in new_num.columns]
    if missing:
        raise KeyError(f"")

    X_new_tab = new_num.reindex(columns=tab_cols_train)

    valid_idx_new = X_new_tab.dropna().index
    if len(valid_idx_new) < len(X_new_tab):
        print(f"")
    X_new_tab = X_new_tab.loc[valid_idx_new].reset_index(drop=True)
    df_new = df_new.loc[valid_idx_new].reset_index(drop=True)

    img_root = _resolve_img_root(img_subdir)
    df_new["img_path"] = df_new.apply(lambda r: _build_img_path(r, img_root), axis=1)
    valid_img = df_new["img_path"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))
    n_bad_new = (~valid_img).sum()
    if n_bad_new:
        print(f"")
    df_new = df_new.loc[valid_img].reset_index(drop=True)
    X_new_tab = X_new_tab.loc[df_new.index].reset_index(drop=True)

    if len(df_new) == 0:
        print("")
        return

    class _TmpDS(Dataset):
        def __init__(self, paths, t): self.paths = list(paths); self.t = t
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            p = self.paths[i]
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), (0,0,0))
            return self.t(img)

    def _extract(paths):
        dl = DataLoader(_TmpDS(paths, transform), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
        feats = []
        with torch.no_grad():
            for imgs in tqdm(dl, total=len(dl), unit="batch", desc="🔎 Extracting CNN features (NEW)", dynamic_ncols=True):
                if device.type == 'cuda':
                    imgs = imgs.to(device, non_blocking=True)
                f = backbone(imgs)
                feats.append(torch.flatten(f, 1).cpu().numpy())
        return np.concatenate(feats, axis=0)

    print("")
    X_new_cnn_raw = _extract(df_new["img_path"].tolist())

    if PCA_COMPONENTS:
        try:
            X_new_cnn = pca.transform(X_new_cnn_raw)
        except NameError:
            raise RuntimeError("")
    else:
        X_new_cnn = X_new_cnn_raw

    X_new_tab_scaled = scaler_tab.transform(X_new_tab.values)
    X_new_all = np.hstack([X_new_tab_scaled, X_new_cnn])
    print("🧱 NEW Final feature shape (Tab+CNN):", X_new_all.shape)

    proba = best_model_tabcnn.predict_proba(X_new_all)
    pred_idx = np.argmax(proba, axis=1)
    pred_stage_value = le.inverse_transform(pred_idx)   # 0/2/5/10/25
    pred_stage_name  = [f"stage{int(s)}" for s in pred_stage_value]
    pred_conf        = proba[np.arange(len(proba)), pred_idx]

    proba_df = pd.DataFrame(proba, columns=class_names)
    out_df = pd.DataFrame({
        "Filename": df_new["Filename"].values,
        "pred_stage_value": pred_stage_value,
        "pred_stage_name": pred_stage_name,
        "confidence": pred_conf,
        "img_path": df_new["img_path"].values
    })
    out_df = pd.concat([out_df, proba_df], axis=1)

    stage_order = [0, 2, 5, 10, 25]
    counts = out_df["pred_stage_value"].value_counts().reindex(stage_order, fill_value=0)
    perc   = (counts / counts.sum() * 100).round(2)
    summary_hard = (
        pd.DataFrame({"stage_value": stage_order, "count": counts.values, "percent": perc.values})
        .assign(stage_name=lambda x: x["stage_value"].map(lambda v: f"stage{int(v)}"))
        [["stage_value", "stage_name", "count", "percent"]]
    )

    soft_means = (proba.mean(axis=0) * 100).round(2)
    soft_summary = pd.DataFrame({"class_name": class_names, "mean_prob_percent": soft_means})
    soft_summary["stage_value"] = soft_summary["class_name"].str.extract(r'(\d+)').astype(int)
    soft_summary = soft_summary.sort_values("stage_value")[["stage_value", "class_name", "mean_prob_percent"]]

    print("\n==== Hard Prediction % by stage (0,2,5,10,25) ====")
    print(summary_hard.to_string(index=False))
    print("\n==== Mean predicted probability (%) by stage ====")
    print(soft_summary.to_string(index=False))

    with pd.ExcelWriter(save_path, engine="xlsxwriter") as w:
        out_df.to_excel(w, sheet_name="predictions", index=False)
        summary_hard.to_excel(w, sheet_name="summary_hard_ordered", index=False)
        soft_summary.to_excel(w, sheet_name="summary_soft_meanprob", index=False)


BATCH_INFER_LIST: List[Dict[str, str]] = [
    {
        "excel_path": r"",
        "img_subdir": r"",
        "save_path":  r"",
    },
    # ... 2/5/10/25 

