import torch
import torch.nn.functional as F
import numpy as np
from .new_clip_attention_extractor import CLIPAttentionExtractor
from typing import Tuple, Optional

class SelfSupervisedAttentionPatchAD:
    """
    Self-supervised attention-guided patch anomaly detector without text prompts
    Implements Axis 1 (Attention-Guided) + Axis 3 (Self-supervised)
    """
    def __init__(self, extractor: CLIPAttentionExtractor, k=25, use_adaptive=True, 
                 contrastive_weight=0.3):
        self.ext = extractor
        self.k = k
        self.use_adaptive = use_adaptive
        self.contrastive_weight = contrastive_weight
        
        # Statistical parameters for memory bank
        self.mu = None
        self.inv = None
        self.normal_features = []
        
        # Self-supervised parameters
        self.feature_memory = None
        self.attention_memory = None
        
    def _create_contrastive_pairs(self, features, attention_weights):
        """
        Create self-supervised contrastive pairs from normal features
        """
        batch_size, num_patches, feat_dim = features.shape
        
        # Create positive pairs (similar patches within same image)
        pos_pairs = []
        neg_pairs = []
        
        for b in range(batch_size):
            feat = features[b]  # [N, C]
            attn = attention_weights[b]  # [N]
            
            # Get top attended patches as anchors
            top_indices = attn.topk(min(10, num_patches)).indices
            
            # Create positive pairs (spatially close patches)
            for i in range(len(top_indices)):
                anchor_idx = top_indices[i]
                # Find spatially close patches (simplified as nearby indices)
                for j in range(max(0, anchor_idx-2), min(num_patches, anchor_idx+3)):
                    if j != anchor_idx:
                        pos_pairs.append((feat[anchor_idx], feat[j]))
                        
                # Create negative pairs (distant patches)
                for _ in range(2):  # 2 negatives per positive
                    neg_idx = torch.randint(0, num_patches, (1,)).item()
                    if abs(neg_idx - anchor_idx) > 5:  # distant patches
                        neg_pairs.append((feat[anchor_idx], feat[neg_idx]))
        
        return pos_pairs, neg_pairs
    
    def _contrastive_loss(self, pos_pairs, neg_pairs, temperature=0.1):
        """
        Compute contrastive loss for self-supervised learning
        """
        if not pos_pairs or not neg_pairs:
            return 0.0
            
        pos_sims = []
        neg_sims = []
        
        for anchor, pos in pos_pairs:
            sim = F.cosine_similarity(anchor.unsqueeze(0), pos.unsqueeze(0))
            pos_sims.append(sim / temperature)
            
        for anchor, neg in neg_pairs:
            sim = F.cosine_similarity(anchor.unsqueeze(0), neg.unsqueeze(0))
            neg_sims.append(sim / temperature)
        
        if pos_sims and neg_sims:
            pos_loss = -torch.log(torch.sigmoid(torch.stack(pos_sims))).mean()
            neg_loss = -torch.log(torch.sigmoid(-torch.stack(neg_sims))).mean()
            return pos_loss + neg_loss
        
        return 0.0
    
    def fit(self, dataloader):
        """
        Enhanced fitting with self-supervised contrastive learning
        """
        all_features = []
        all_attention_weights = []
        contrastive_losses = []
        
        print("Extracting features and computing self-supervised learning...")
        
        for batch_idx, (img, _) in enumerate(dataloader):
            img = img.to(self.ext.device)
            
            # Get patch features and attention weights
            patch_features = self.ext.img2patch_features(img)  # [B, N, C]
            
            if self.use_adaptive:
                _, attention_weights = self.ext.adaptive_patch_selection(img, self.k)
            else:
                _, attention_weights = self.ext.topk_patches(img, self.k)
            
            # Self-supervised contrastive learning
            pos_pairs, neg_pairs = self._create_contrastive_pairs(patch_features, attention_weights)
            cont_loss = self._contrastive_loss(pos_pairs, neg_pairs)
            
            if cont_loss > 0:
                contrastive_losses.append(cont_loss)
            
            # Store for memory bank
            all_features.append(patch_features.reshape(-1, patch_features.size(-1)))
            all_attention_weights.append(attention_weights.reshape(-1))
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}, contrastive loss: {cont_loss:.4f}")
        
        # Combine all features
        all_features = torch.cat(all_features, 0)  # [M, C]
        all_attention_weights = torch.cat(all_attention_weights, 0)  # [M]
        
        print(f"Total features collected: {all_features.shape[0]}")
        print(f"Average contrastive loss: {np.mean(contrastive_losses) if contrastive_losses else 0:.4f}")
        
        # Create attention-weighted statistics (Axis 3: Self-supervised)
        # Weight features by attention importance
        attention_weights_norm = all_attention_weights / (all_attention_weights.sum() + 1e-8)
        weights = attention_weights_norm.unsqueeze(-1)
        
        # Weighted mean and covariance
        self.mu = (all_features * weights).sum(0, keepdim=True)
        
        # Compute weighted covariance matrix
        centered_features = all_features - self.mu
        weighted_cov = (centered_features.T @ (centered_features * weights))
        
        # Add regularization and compute inverse
        reg_cov = weighted_cov + 1e-5 * torch.eye(all_features.shape[1], device=all_features.device)
        self.inv = torch.linalg.inv(reg_cov)
        
        # Store feature memory for additional self-supervised comparison
        self.feature_memory = all_features
        self.attention_memory = all_attention_weights
        
        print(f"Memory bank created with {self.feature_memory.shape[0]} features")
        print("Self-supervised training completed!")
    
    def score(self, img: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Enhanced scoring combining statistical anomaly detection with self-supervised features
        """
        # Get attention-guided patches (Axis 1)
        if self.use_adaptive:
            idx, attention_weights = self.ext.adaptive_patch_selection(img, self.k)
        else:
            idx, attention_weights = self.ext.topk_patches(img, self.k)
        
        # Extract patch features
        patch_features = self.ext.img2patch_features(img)[0]  # [N, C]
        selected_features = patch_features[idx[0]]  # [k, C]
        
        # Statistical anomaly score (Mahalanobis distance)
        delta = selected_features - self.mu
        mahal_distances = (delta @ self.inv * delta).sum(-1).sqrt()
        statistical_score = mahal_distances.mean()
        
        # Self-supervised anomaly score (Axis 3)
        # Compare with memory bank using attention-weighted similarity
        if self.feature_memory is not None:
            similarities = []
            for feat in selected_features:
                # Compute similarity with memory bank
                sims = F.cosine_similarity(feat.unsqueeze(0), self.feature_memory, dim=1)
                # Weight by attention importance from memory
                weighted_sims = sims * self.attention_memory
                similarities.append(weighted_sims.max().item())
            
            # Lower similarity means higher anomaly
            self_supervised_score = 1.0 - (sum(similarities) / len(similarities))
        else:
            self_supervised_score = 0.0
        
        # Combine scores
        final_score = (1 - self.contrastive_weight) * statistical_score + \
                      self.contrastive_weight * self_supervised_score
        
        return final_score.item(), attention_weights

# Backward compatibility
AttentionPatchAD = SelfSupervisedAttentionPatchAD
