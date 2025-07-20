import torch
import torch.nn.functional as F
import clip
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import uuid

class CLIPAttentionExtractor:
    """
    Enhanced CLIP attention extractor for self-supervised anomaly detection
    """
    def __init__(self, model_name: str = "ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        v = self.model.visual
        self.P = v.conv1.kernel_size[0]  # patch size
        self.R = v.input_resolution
        self.N = (self.R // self.P) ** 2  # number of patches
        self.H = v.transformer.width // 64  # number of heads
        self.L = len(v.transformer.resblocks)  # number of layers
        
        self._attn: Dict[str, torch.Tensor] = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for transformer blocks"""
        for idx, block in enumerate(self.model.visual.transformer.resblocks):
            block.attn.register_forward_hook(self._hook(idx))
    
    def _hook(self, idx):
        """Forward hook to capture attention weights"""
        name = f"layer_{idx}"
        def fn(module, inp, out):
            x = inp[0]  # [seq_len, batch, embed_dim]
            
            # Get QKV projections
            qkv = module.in_proj_weight @ x.transpose(0, 1).reshape(-1, x.size(-1)).T
            qkv = qkv.T.reshape(x.size(1), x.size(0), 3, self.H, -1)
            qkv = qkv.permute(2, 1, 3, 0, 4)  # [3, seq, heads, batch, head_dim]
            
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights
            scale = (q.size(-1)) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            # Store attention weights [batch, heads, seq, seq]
            self._attn[name] = attn.permute(2, 1, 0, 3).detach()
        
        return fn
    
    @torch.no_grad()
    def get_agg_attn(self, img: torch.Tensor, layer: int = -1, mode="mean") -> torch.Tensor:
        """
        Returns aggregated CLS→patch weights [B, N] with enhanced aggregation modes
        """
        self._attn.clear()
        _ = self.model.encode_image(img)
        m = self._attn[f"layer_{layer%self.L}"]  # [B,H,seq,seq]
        cls2patch = m[:, :, 0, 1:]  # drop CLS→CLS
        
        if mode == "mean": 
            cls2patch = cls2patch.mean(1)
        elif mode == "max": 
            cls2patch = cls2patch.max(1).values
        elif mode == "sum": 
            cls2patch = cls2patch.sum(1)
        elif mode == "entropy_weighted":  # NEW: attention entropy weighting
            # Weight by attention entropy (lower entropy = more focused)
            entropy = -(cls2patch * torch.log(cls2patch + 1e-8)).sum(-1)
            weights = F.softmax(-entropy, dim=1).unsqueeze(-1)
            cls2patch = (cls2patch * weights).sum(1)
        
        return cls2patch
    
    def adaptive_patch_selection(self, img: torch.Tensor, k=25, threshold=0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced patch selection with adaptive thresholding
        """
        attn_weights = self.get_agg_attn(img, mode="mean")
        
        # Adaptive k based on attention distribution
        attention_std = attn_weights.std(dim=1, keepdim=True)
        attention_mean = attn_weights.mean(dim=1, keepdim=True)
        
        # Select patches above mean + threshold * std
        adaptive_mask = attn_weights > (attention_mean + threshold * attention_std)
        
        
        # Get top-k indices
        try:
            adaptive_k = min(k, adaptive_mask.sum(dim=1).max().item())
            adaptive_k = max(adaptive_k, 5)  # Minimum 5 patches
            idx = attn_weights.topk(adaptive_k, dim=1).indices
        except Exception as e:
            print(f"Error occurred while getting top-k indices: {e}")
            idx = torch.zeros((attn_weights.size(0), adaptive_k), device=attn_weights.device).long()

        return idx, attn_weights
    
    def topk_patches(self, img: torch.Tensor, k=10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns indices of k most-attended patches and the full weight map.
        """
        w = self.get_agg_attn(img, mode="entropy_weighted")
        actual_patches = w.shape[1]
        actual_correct_k = min(k, actual_patches) 
        idx = w.topk(actual_correct_k, dim=1).indices
        return idx, w
    
    def img2patch_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Returns ViT patch embeddings [B, N, C] before the class token.
        """
        with torch.no_grad():
            v = self.model.visual
            x = v.conv1(img.to(torch.float16))  # [B,C,H/P,W/P]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # flatten
            x = x.permute(0, 2, 1) + v.positional_embedding[1:]  # skip CLS
            return x  # [B,N,C]
    
    def heatmap_overlay(self, img_tensor, w, save=None):
        """
        Overlays attention heat-map on original image. img_tensor shape BxCxHxW.
        """
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        if w.ndim == 2:
            w = w[0]
        
        # Ensure w has the right number of elements
        expected_patches = self.N
        if w.numel() != expected_patches:
            if w.numel() < expected_patches:
                padding = torch.zeros(expected_patches - w.numel(), device=w.device)
                w = torch.cat([w, padding])
            else:
                w = w[:expected_patches]

        grid_size = int(np.sqrt(expected_patches))
        grid = w.view(grid_size, grid_size).cpu().numpy()
        heat = cv2.applyColorMap((grid*255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (self.R, self.R))
        im = img_tensor.cpu().permute(1,2,0).numpy()
        im = ((im*0.5+0.5)*255).astype(np.uint8)  # de-norm
        out = cv2.addWeighted(im, 0.6, heat, 0.4, 0)
        if save: 
            cv2.imwrite(save, out[:, :, ::-1])  # BGR→RGB
        
        return out
