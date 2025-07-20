import torch
import tqdm
import numpy as np
from sklearn import metrics
import os
import json
from torchvision import transforms
import cv2
# from models.new_clip_attention_extractor import CLIPAttentionExtractor
from models.newest_clip_attention_extractor import CLIPAttentionExtractor
from models.new_anomaly_detector import SelfSupervisedAttentionPatchAD
from utils.new_data_utils import mvtec_train_loader, mvtec_test_loader, load_random_images

# Configuration
CONFIG = {
    "PATH": "overlay/",
    "MODEL_NAME": "ViT-B/32", 
    "K": 25,
    "USE_ADAPTIVE": True,
    "CONTRASTIVE_WEIGHT": 0.3,
    "SAVE_VISUALIZATIONS": True,
    "DATA_ROOT": "data"  # Update this to your MVTec dataset path
}

# TEST_CONFIG = {
#     "RANDOM_DIR": "data/random_input",
#     "OUTPUT_JSON": "overlay/random_results.json",
#     "OUTPUT_HEATMAP_DIR": "overlay/heatmaps/",
#     "MODEL_NAME": "ViT-B/32",
#     "K": 25,
#     "USE_ADAPTIVE": True,
#     "CONTRASTIVE_WEIGHT": 0.3,
#     "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
# }


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4814, 0.4578, 0.4082),
                        (0.2686, 0.2613, 0.2758))
])

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("="*60)
    print("AXES 1 + 3: Attention-Guided Self-Supervised Anomaly Detection")
    print("="*60)
    
    # Initialize models
    print("Initializing CLIP attention extractor...")
    extractor = CLIPAttentionExtractor(CONFIG["MODEL_NAME"], device)

    print("Initializing self-supervised anomaly detector...")
    detector = SelfSupervisedAttentionPatchAD(
        extractor, 
        k=CONFIG["K"], 
        use_adaptive=CONFIG["USE_ADAPTIVE"],
        contrastive_weight=CONFIG["CONTRASTIVE_WEIGHT"]
    )
    
    # Create output directory
    os.makedirs(CONFIG["PATH"], exist_ok=True)
    
    # Training phase (Self-supervised learning on normal images only)
    print("\n" + "="*60)
    print("TRAINING PHASE - Self-Supervised Learning")
    print("="*60)
    print("Loading training data (normal images only)...")
    train_loader = mvtec_train_loader(root=CONFIG["DATA_ROOT"], split='train')
    
    print(f"Training self-supervised anomaly detector with contrastive learning...")
    detector.fit(train_loader)
    
    # Testing phase
    print("\n" + "="*60)
    print("TESTING PHASE - Anomaly Detection")
    print("="*60)
    print("Loading test data...")
    test_loader = mvtec_test_loader(root=CONFIG["DATA_ROOT"], split='test')
    
    print("Running inference with attention-guided patch analysis...")
    scores, labels = [], []
    
    for batch_idx, (img, lbl) in enumerate(tqdm.tqdm(test_loader)):
        img = img.to(device)
        
        # Get anomaly score using Axes 1 + 3
        anomaly_score, attention_weights = detector.score(img)
        scores.append(anomaly_score)
        labels.extend(lbl.tolist())
        
        # Save visualization if enabled
        if CONFIG["SAVE_VISUALIZATIONS"] and batch_idx < 50:  # Save first 50 samples
            is_anomaly = "anomaly" if lbl[0] > 0 else "normal"
            save_path = f"{CONFIG['PATH']}sample_{batch_idx:03d}_{is_anomaly}_score_{anomaly_score:.3f}.png"
            extractor.heatmap_overlay(img, attention_weights, save=save_path)
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Ensure equal lengths
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]
    
    # Convert to binary labels (assuming 0=normal, >0=anomalous)
    binary_labels = [1 if l > 0 else 0 for l in labels]
    
    # Compute AUROC
    if len(set(binary_labels)) > 1:  # Check if we have both classes
        auroc = metrics.roc_auc_score(binary_labels, scores)
        
        # Compute threshold and other metrics
        fpr, tpr, thresholds = metrics.roc_curve(binary_labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        pred_labels = [1 if s > optimal_threshold else 0 for s in scores]
        accuracy = metrics.accuracy_score(binary_labels, pred_labels)
        precision = metrics.precision_score(binary_labels, pred_labels, zero_division=0)
        recall = metrics.recall_score(binary_labels, pred_labels, zero_division=0)
        f1 = metrics.f1_score(binary_labels, pred_labels, zero_division=0)
        
        # Print results
        print(f"AUROC: {auroc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
    else:
        print("Warning: Only one class found in labels, cannot compute AUROC")
        auroc = accuracy = precision = recall = f1 = optimal_threshold = 0.0
    
    print(f"Total Samples: {len(scores)}")
    print(f"Normal Samples: {binary_labels.count(0)}")
    print(f"Anomaly Samples: {binary_labels.count(1)}")
    print("="*60)
    
    # Save results
    results = {
        "method": "Attention-Guided Self-Supervised (Axes 1+3)",
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "optimal_threshold": float(optimal_threshold),
        "total_samples": len(scores),
        "normal_samples": binary_labels.count(0),
        "anomaly_samples": binary_labels.count(1),
        "config": CONFIG
    }
    
    with open(f"{CONFIG['PATH']}results_axes1_3.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {CONFIG['PATH']}results_axes1_3.json")
    print(f"Visualizations saved to {CONFIG['PATH']}")
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main()

