import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class ModelIntegrator:
    """
    Integrates with your existing CLIP-based anomaly detection model
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.transform = self._get_transform()
        
        # These will be set based on your actual model implementation
        self.feature_bank = None
        self.attention_extractor = None
        
    def _get_transform(self):
        """Standard CLIP preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def load_model(self):
        """Load your trained anomaly detection model"""
        # This should be adapted to your actual model loading code
        # from your anomaly_detector.py
        try:
            # Example: assuming you have a saved state dict
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                # Load your model state here
                print(f"Model loaded from {self.model_path}")
            else:
                print("No model path provided, using default initialization")
                # Initialize your model here
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Initialize with default parameters
            
    def extract_attention_features(self, image_tensor):
        """
        Extract attention-guided features using your CLIP attention mechanism
        This should integrate with your clip_attention_extractor.py
        """
        with torch.no_grad():
            # Your attention extraction logic here
            # This is a placeholder - replace with your actual implementation
            features = torch.randn(512)  # Replace with actual feature extraction
            return features
    
    def compute_anomaly_score(self, image_path: str) -> float:
        """
        Compute anomaly score for a single image
        Integrates with your k-NN based scoring from anomaly_detector.py
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features using attention-guided approach (Axis 1)
            features = self.extract_attention_features(image_tensor)
            
            # Compute k-NN distance to feature bank (Axis 3 - Self-supervised refining)
            if self.feature_bank is not None:
                distances = torch.cdist(features.unsqueeze(0), self.feature_bank)
                # Use minimum distance or k-NN average as anomaly score
                anomaly_score = torch.min(distances).item()
            else:
                # Fallback scoring method
                anomaly_score = torch.norm(features).item()
                
            return anomaly_score
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0.0

class TestEvaluator:
    """
    Comprehensive test evaluation system for zero-shot anomaly detection
    """
    
    def __init__(self, test_data_path: str, output_dir: str = "evaluation_results"):
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"evaluation_{self.timestamp}"
        
        # Create output directories
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir = self.eval_dir / "visual_comparison"
        self.visual_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = []
        self.model_integrator = None
        
    def setup_model(self, model_path: str = None):
        """Initialize and load the anomaly detection model"""
        self.model_integrator = ModelIntegrator(model_path)
        self.model_integrator.load_model()
        
    def load_test_data(self) -> Dict[str, List[str]]:
        """
        Load test images from good/ and anomaly/ folders
        Returns dict with 'good' and 'anomaly' image paths
        """
        test_images = {'good': [], 'anomaly': []}
        
        # Load good images
        good_dir = self.test_data_path / "good"
        if good_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                test_images['good'].extend(list(good_dir.glob(ext)))
                
        # Load anomaly images  
        anomaly_dir = self.test_data_path / "anomaly"
        if anomaly_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                test_images['anomaly'].extend(list(anomaly_dir.glob(ext)))
                
        print(f"Loaded {len(test_images['good'])} good images")
        print(f"Loaded {len(test_images['anomaly'])} anomaly images")
        
        return test_images
    
    def predict_batch(self, image_paths: List[Path], true_labels: List[int]) -> Tuple[List[float], List[int]]:
        """
        Predict anomaly scores for a batch of images
        """
        scores = []
        predictions = []
        
        for img_path in image_paths:
            score = self.model_integrator.compute_anomaly_score(str(img_path))
            scores.append(score)
            
        # Convert scores to binary predictions (you may need to adjust threshold)
        threshold = np.median(scores)  # Simple threshold - can be optimized
        predictions = [1 if score > threshold else 0 for score in scores]
        
        return scores, predictions
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], 
                         y_scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Advanced metrics
        metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)  # Same as recall
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC and PR metrics
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
            
        # Confusion matrix components
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def create_visual_comparison(self, image_paths: List[Path], 
                               true_labels: List[int], pred_labels: List[int]):
        """
        Create organized visual comparison folders
        """
        # Create comparison directories
        tp_dir = self.visual_dir / "true_positives"
        tn_dir = self.visual_dir / "true_negatives" 
        fp_dir = self.visual_dir / "false_positives"
        fn_dir = self.visual_dir / "false_negatives"
        all_dir = self.visual_dir / "all_predictions"
        
        for dir_path in [tp_dir, tn_dir, fp_dir, fn_dir, all_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Copy images to appropriate folders
        for img_path, true_label, pred_label in zip(image_paths, true_labels, pred_labels):
            img_name = f"{true_label}_{pred_label}_{img_path.name}"
            
            # Copy to all_predictions folder
            shutil.copy2(img_path, all_dir / img_name)
            
            # Copy to specific category folder
            if true_label == 1 and pred_label == 1:  # True Positive
                shutil.copy2(img_path, tp_dir / img_path.name)
            elif true_label == 0 and pred_label == 0:  # True Negative
                shutil.copy2(img_path, tn_dir / img_path.name)
            elif true_label == 0 and pred_label == 1:  # False Positive
                shutil.copy2(img_path, fp_dir / img_path.name)
            elif true_label == 1 and pred_label == 0:  # False Negative
                shutil.copy2(img_path, fn_dir / img_path.name)
    
    def create_evaluation_plots(self, y_true: List[int], y_scores: List[float]):
        """Generate comprehensive evaluation plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        if len(set(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
            
            axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Score Distribution
        normal_scores = [score for score, label in zip(y_scores, y_true) if label == 0]
        anomaly_scores = [score for score, label in zip(y_scores, y_true) if label == 1]
        
        axes[1, 0].hist(normal_scores, alpha=0.7, label='Normal', bins=30, color='blue')
        axes[1, 0].hist(anomaly_scores, alpha=0.7, label='Anomaly', bins=30, color='red')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        
        # Confusion Matrix
        y_pred = [1 if score > np.median(y_scores) else 0 for score in y_scores]
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'comprehensive_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, metrics: Dict[str, float], 
                               image_paths: List[Path], y_true: List[int], 
                               y_pred: List[int], y_scores: List[float]):
        """Generate detailed evaluation report"""
        
        # Create detailed results DataFrame
        detailed_results = pd.DataFrame({
            'image_path': [str(p) for p in image_paths],
            'true_label': y_true,
            'predicted_label': y_pred,
            'anomaly_score': y_scores,
            'correct_prediction': [t == p for t, p in zip(y_true, y_pred)]
        })
        
        # Save detailed results
        detailed_results.to_csv(self.eval_dir / 'detailed_results.csv', index=False)
        
        # Save metrics as JSON
        with open(self.eval_dir / 'comprehensive_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Generate markdown summary report
        report_content = f"""# Zero-Shot Anomaly Detection Evaluation Report

## Evaluation Summary
- **Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Images**: {len(image_paths)}
- **Normal Images**: {sum(1 for label in y_true if label == 0)}
- **Anomaly Images**: {sum(1 for label in y_true if label == 1)}

## Performance Metrics

### Classification Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall (Sensitivity)**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **Specificity**: {metrics['specificity']:.4f}

### Advanced Metrics
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **PR-AUC**: {metrics['pr_auc']:.4f}

### Confusion Matrix
|               | Predicted Normal | Predicted Anomaly |
|---------------|------------------|-------------------|
| **Actual Normal**  | {metrics['true_negatives']} | {metrics['false_positives']} |
| **Actual Anomaly** | {metrics['false_negatives']} | {metrics['true_positives']} |

## Model Performance Analysis

### Strengths
"""
        
        if metrics['accuracy'] > 0.8:
            report_content += "- **High Overall Accuracy**: The model demonstrates strong classification performance.\n"
        if metrics['roc_auc'] > 0.8:
            report_content += "- **Excellent ROC-AUC**: Strong discrimination between normal and anomalous samples.\n"
        if metrics['precision'] > 0.8:
            report_content += "- **High Precision**: Low false positive rate, reliable anomaly flagging.\n"
        if metrics['recall'] > 0.8:
            report_content += "- **High Recall**: Effective detection of actual anomalies.\n"
            
        report_content += "\n### Areas for Improvement\n"
        
        if metrics['false_positives'] > metrics['true_positives']:
            report_content += "- **False Positive Rate**: Consider adjusting the anomaly threshold to reduce false alarms.\n"
        if metrics['false_negatives'] > metrics['true_negatives'] * 0.1:
            report_content += "- **Missed Anomalies**: Model may benefit from additional attention mechanism tuning.\n"
        if metrics['roc_auc'] < 0.7:
            report_content += "- **ROC Performance**: Consider feature bank expansion or k-NN parameter optimization.\n"
            
        report_content += f"""
## Integration with Project Axes

### Axis 1: Attention-Guided Patch Analysis
The evaluation shows how well the attention mechanism focuses on relevant regions for anomaly detection.

### Axis 3: Self-Supervised Refining  
The k-NN based scoring demonstrates effectiveness of label-free anomaly detection approach.

## Files Generated
- `detailed_results.csv`: Per-image results with scores
- `comprehensive_metrics.json`: All metrics in JSON format
- `comprehensive_evaluation_plots.png`: Visualization plots
- Visual comparison folders with categorized images

## Usage Notes
This evaluation framework integrates with your existing CLIP-based anomaly detection pipeline, specifically designed for your attention-guided and self-supervised refining approaches.
"""
        
        with open(self.eval_dir / 'summary_report.md', 'w') as f:
            f.write(report_content)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Main evaluation pipeline
        """
        print("Starting comprehensive evaluation...")
        
        # Load test data
        test_images = self.load_test_data()
        
        if not test_images['good'] and not test_images['anomaly']:
            raise ValueError("No test images found. Check your test data path.")
            
        # Prepare data for evaluation
        all_image_paths = test_images['good'] + test_images['anomaly']
        all_true_labels = [0] * len(test_images['good']) + [1] * len(test_images['anomaly'])
        
        print(f"Running inference on {len(all_image_paths)} images...")
        
        # Get predictions
        y_scores, y_pred = self.predict_batch(all_image_paths, all_true_labels)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_true_labels, y_pred, y_scores)
        
        print("Generating visual comparisons...")
        # Create visual comparison
        self.create_visual_comparison(all_image_paths, all_true_labels, y_pred)
        
        print("Creating evaluation plots...")
        # Create plots
        self.create_evaluation_plots(all_true_labels, y_scores)
        
        print("Generating detailed report...")
        # Generate report
        self.generate_detailed_report(metrics, all_image_paths, all_true_labels, y_pred, y_scores)
        
        print(f"Evaluation complete! Results saved to: {self.eval_dir}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'scores': y_scores,
            'output_directory': str(self.eval_dir)
        }

def run_evaluation_with_existing_model(model_path: str = None, 
                                     test_data_path: str = "data/test",
                                     output_dir: str = "evaluation_results"):
    """
    Convenience function to run evaluation with existing model
    """
    evaluator = TestEvaluator(test_data_path, output_dir)
    evaluator.setup_model(model_path)
    return evaluator.run_evaluation()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Evaluation for Zero-Shot Anomaly Detection')
    parser.add_argument('--test-path', default='data/screw/test', help='Path to test data directory')
    parser.add_argument('--model-path', default=None, help='Path to trained model')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation_with_existing_model(
            model_path=args.model_path,
            test_data_path=args.test_path,
            output_dir=args.output_dir
        )
        print("Evaluation completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
