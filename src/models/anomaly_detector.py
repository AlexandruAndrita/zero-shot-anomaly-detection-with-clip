import torch, torch.nn.functional as F
from .clip_attention_extractor import CLIPAttentionExtractor

class AttentionPatchAD:
    """
    Simple memory-bank detector:
    1. collect mean μ and covariance Σ of ALL normal patch features
    2. during test, Mahalanobis distance of top-k patches = anomaly score
    """
    def __init__(self, extractor: CLIPAttentionExtractor, k=25):
        self.ext = extractor
        self.k   = k
        self.mu  = None
        self.inv = None                                # Σ⁻¹

    # ------------- stage 1: fit --------------------------------------------
    def fit(self, dataloader):
        feats = []
        for img, _ in dataloader:                      # only normals
            img = img.to(self.ext.device)
            f   = self.ext.img2patch_features(img)     # [B,N,C]
            feats.append(f.reshape(-1, f.size(-1)))
        feats = torch.cat(feats, 0)                    # [M,C]
        self.mu  = feats.mean(0, keepdim=True)
        Σ        = torch.cov(feats.T) + 1e-5*torch.eye(feats.shape[1], device=feats.device)
        self.inv = torch.linalg.inv(Σ)

    # ------------- stage 2: score ------------------------------------------
    def score(self, img: torch.Tensor) -> float:
        idx, w  = self.ext.topk_patches(img, self.k)   # indices
        patch_f = self.ext.img2patch_features(img)[0]  # [N,C]
        sel_f   = patch_f[idx[0]]                     # [k,C]
        δ       = sel_f - self.mu                     # broadcast
        m_dist  = (δ @ self.inv * δ).sum(-1).sqrt()   # [k]
        return m_dist.mean().item(), w                # scalar score
