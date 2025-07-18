import torch, json, tqdm, os
from models.clip_attention_extractor import CLIPAttentionExtractor
from models.anomaly_detector           import AttentionPatchAD
from utils.data_utils                  import mvtec_loader

device   = "cuda" if torch.cuda.is_available() else "cpu"
extract  = CLIPAttentionExtractor("ViT-B/32", device)
detector = AttentionPatchAD(extract, k=25)

# ---- 1. fit on normal train split ----------------------------------------
train_loader = mvtec_loader(split='train')
detector.fit(train_loader)

# ---- 2. evaluate ---------------------------------------------------------
test_loader  = mvtec_loader(split='test')
scores, labels = [], []
for img, lbl in tqdm.tqdm(test_loader):
    s, w = detector.score(img.to(device))
    scores.append(s);       
    labels+=lbl.tolist()
    # optional: save overlay
    extract.heatmap_overlay(img, w, save=f"D:\\master\\summer 25\\subjects\\high-level computer vision\\test project\\overlay\\{len(scores)}.png")
    

# ---- 3. AUROC ------------------------------------------------------------
# from sklearn import metrics
# auroc = metrics.roc_auc_score(labels[:len(scores)], scores, multi_class='ovr')
# print(f"AUROC: {auroc:.3f}")
