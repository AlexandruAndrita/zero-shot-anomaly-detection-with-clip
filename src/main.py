import torch, tqdm
from models.clip_attention_extractor import CLIPAttentionExtractor
from models.anomaly_detector import AttentionPatchAD
from utils.data_utils import mvtec_train_loader, mvtec_test_loader

PATH = "D:\\master\\summer 25\\subjects\\high-level computer vision\\test project\\overlay\\"
MODEL_NAME = "ViT-B/32"

device = "cuda" if torch.cuda.is_available() else "cpu"
extract = CLIPAttentionExtractor(MODEL_NAME, device)
detector = AttentionPatchAD(extract, k=25)

train_loader = mvtec_train_loader(split='train')
detector.fit(train_loader)

test_loader  = mvtec_test_loader(split='test')
scores, labels = [], []
for img, lbl in tqdm.tqdm(test_loader):
    s, w = detector.score(img.to(device))
    scores.append(s);       
    labels+=lbl.tolist()
    extract.heatmap_overlay(img, w, save=f"{PATH}{len(scores)}.png")

# needs to be sorted out - debug first
# ---- 3. AUROC ------------------------------------------------------------
# from sklearn import metrics
# auroc = metrics.roc_auc_score(labels[:len(scores)], scores, multi_class='ovr')
# print(f"AUROC: {auroc:.3f}")
