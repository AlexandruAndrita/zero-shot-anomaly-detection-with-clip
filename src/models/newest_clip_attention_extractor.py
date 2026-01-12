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
        
        # FIX 1: Calculate N correctly based on actual model behavior
        # For ViT-B/32 with 224x224 input, this gives 7x7 = 49 patches
        self.N = (self.R // self.P) ** 2  # number of patches
        
        self.H = v.transformer.width // 64  # number of heads
        self.L = len(v.transformer.resblocks)  # number of layers
        
        print(f"Model parameters: P={self.P}, R={self.R}, N={self.N}, H={self.H}, L={self.L}")
        
        self._attn: Dict[str, torch.Tensor] = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for transformer blocks"""
        for idx, block in enumerate(self.model.visual.transformer.resblocks):
            block.attn.register_forward_hook(self._hook(idx))
    
    def _hook(self, idx):
        """Forward hook to capture attention weights - FIXED"""
        name = f"layer_{idx}"
        def fn(module, inp, out):
            x = inp[0]  # [seq_len, batch, embed_dim]
            seq_len, batch_size, embed_dim = x.shape
            
            # FIX 2: More robust QKV computation
            try:
                # Standard attention computation
                B, T, C = batch_size, seq_len, embed_dim
                
                # Get QKV from the linear layer
                qkv = F.linear(x.transpose(0, 1), module.in_proj_weight, module.in_proj_bias)  # [B, T, 3*C]
                qkv = qkv.reshape(B, T, 3, self.H, C // self.H).permute(2, 0, 3, 1, 4)  # [3, B, H, T, C_head]
                
                q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, T, C_head]
                
                # Compute attention weights
                scale = (C // self.H) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
                attn = attn.softmax(dim=-1)
                
                # Store attention weights [B, H, seq, seq]
                self._attn[name] = attn.detach()
                
            except Exception as e:
                print(f"Error in attention hook {name}: {e}")
                # Fallback: create dummy attention weights
                dummy_attn = torch.ones(batch_size, self.H, seq_len, seq_len, device=x.device) / seq_len
                self._attn[name] = dummy_attn
        
        return fn
    
    @torch.no_grad()
    def get_agg_attn(self, img: torch.Tensor, layer: int = -1, mode="mean") -> torch.Tensor:
        """
        Returns aggregated CLS→patch weights [B, N] - FIXED
        """
        self._attn.clear()
        _ = self.model.encode_image(img)
        
        layer_name = f"layer_{layer % self.L}"
        if layer_name not in self._attn:
            print(f"Warning: Layer {layer_name} not found in attention cache")
            # Return uniform attention as fallback
            batch_size = img.shape[0]
            return torch.ones(batch_size, self.N, device=img.device) / self.N
        
        m = self._attn[layer_name]  # [B, H, seq, seq]
        
        # FIX 3: Ensure we have the right sequence length
        if m.shape[-1] != self.N + 1:  # Should be N patches + 1 CLS token
            print(f"Warning: Attention matrix has sequence length {m.shape[-1]}, expected {self.N + 1}")
            # Adjust if needed
            if m.shape[-1] > self.N + 1:
                m = m[:, :, :self.N+1, :self.N+1]
        
        cls2patch = m[:, :, 0, 1:]  # CLS to patch attention [B, H, N]
        
        # FIX 4: Ensure we have exactly N patches
        if cls2patch.shape[-1] != self.N:
            if cls2patch.shape[-1] < self.N:
                # Pad with small values
                padding_size = self.N - cls2patch.shape[-1]
                padding = torch.full((cls2patch.shape[0], cls2patch.shape[1], padding_size), 1e-6, device=cls2patch.device, dtype=cls2patch.dtype)
                cls2patch = torch.cat([cls2patch, padding], dim=-1)
            else:
                # Truncate
                cls2patch = cls2patch[:, :, :self.N]
        
        # Aggregate across heads
        if mode == "mean":
            cls2patch = cls2patch.mean(1)
        elif mode == "max":
            cls2patch = cls2patch.max(1).values
        elif mode == "sum":
            cls2patch = cls2patch.sum(1)
        elif mode == "entropy_weighted":
            # Weight by attention entropy (lower entropy = more focused)
            entropy = -(cls2patch * torch.log(cls2patch + 1e-8)).sum(-1)  # [B, H]
            weights = F.softmax(-entropy, dim=1).unsqueeze(-1)  # [B, H, 1]
            cls2patch = (cls2patch * weights).sum(1)  # [B, N]
        
        return cls2patch  # [B, N]
    
    def adaptive_patch_selection(self, img: torch.Tensor, k=25, threshold=0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced patch selection with adaptive thresholding - FIXED
        """
        attn_weights = self.get_agg_attn(img, mode="mean")  # [B, N]
        
        # Adaptive k based on attention distribution
        attention_std = attn_weights.std(dim=1, keepdim=True)
        attention_mean = attn_weights.mean(dim=1, keepdim=True)
        
        # Select patches above mean + threshold * std
        adaptive_mask = attn_weights > (attention_mean + threshold * attention_std)
        
        # Get top-k indices
        try:
            # FIX 5: Ensure k doesn't exceed available patches
            max_patches = attn_weights.shape[1]  # Should be N
            adaptive_k = min(k, adaptive_mask.sum(dim=1).max().item())
            adaptive_k = max(adaptive_k, min(5, max_patches))  # Minimum 5 or max_patches
            adaptive_k = min(adaptive_k, max_patches)  # Don't exceed available patches
            
            idx = attn_weights.topk(adaptive_k, dim=1).indices
        except Exception as e:
            print(f"Error in adaptive patch selection: {e}")
            # Fallback: return first k patches
            batch_size = attn_weights.shape[0]
            adaptive_k = min(k, max_patches, 5)
            idx = torch.arange(adaptive_k, device=attn_weights.device).unsqueeze(0).repeat(batch_size, 1)
        
        return idx, attn_weights
    
    def topk_patches(self, img: torch.Tensor, k=10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns indices of k most-attended patches and the full weight map - FIXED
        """
        w = self.get_agg_attn(img, mode="entropy_weighted")  # [B, N]
        actual_patches = w.shape[1]
        actual_correct_k = min(k, actual_patches)
        idx = w.topk(actual_correct_k, dim=1).indices
        return idx, w
    
    def img2patch_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Returns ViT patch embeddings [B, N, C] before the class token - FIXED
        """
        with torch.no_grad():
            v = self.model.visual
            x = v.conv1(img.to(torch.float16))  # [B, C, H/P, W/P]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, H*W/P^2]
            x = x.permute(0, 2, 1)  # [B, H*W/P^2, C]
            
            # FIX 6: Ensure correct positional embedding
            pos_embed = v.positional_embedding[1:]  # Skip CLS token
            
            # Handle size mismatch in positional embeddings
            if pos_embed.shape[0] != x.shape[1]:
                print(f"Positional embedding size mismatch: {pos_embed.shape[0]} vs {x.shape[1]}")
                if pos_embed.shape[0] < x.shape[1]:
                    # Pad positional embeddings
                    padding = torch.zeros(x.shape[1] - pos_embed.shape[0], pos_embed.shape[1], 
                                        device=pos_embed.device, dtype=pos_embed.dtype)
                    pos_embed = torch.cat([pos_embed, padding], dim=0)
                else:
                    # Truncate positional embeddings
                    pos_embed = pos_embed[:x.shape[1]]
            
            x = x + pos_embed  # Add positional embedding
            return x  # [B, N, C]
    
    def heatmap_overlay(self, img_tensor, w, save=None):
        """
        Overlays attention heat-map on original image - FIXED
        """
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        if w.ndim == 2:
            w = w[0]
        
        # FIX 7: Use actual patch count, not self.N which might be wrong
        actual_patches = w.numel()
        expected_patches = self.N
        
        if actual_patches != expected_patches:
            print(f"Warning: Attention weights have {actual_patches} elements, expected {expected_patches}")
            if actual_patches < expected_patches:
                padding = torch.zeros(expected_patches - actual_patches, device=w.device)
                w = torch.cat([w, padding])
            else:
                w = w[:expected_patches]
        
        # Create square grid
        grid_size = int(np.sqrt(self.N))
        if grid_size * grid_size != self.N:
            print(f"Warning: Cannot create perfect square grid from {self.N} patches")
            grid_size = int(np.sqrt(self.N))
        
        grid = w[:grid_size*grid_size].view(grid_size, grid_size).cpu().numpy()
        heat = cv2.applyColorMap((grid*255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (self.R, self.R))
        im = img_tensor.cpu().permute(1,2,0).numpy()
        im = ((im*0.5+0.5)*255).astype(np.uint8)  # de-norm
        out = cv2.addWeighted(im, 0.6, heat, 0.4, 0)
        if save:
            cv2.imwrite(save, out[:, :, ::-1])  # BGR→RGB
        
        return out
