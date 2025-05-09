import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from huggingface_hub import login

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "MahmoodLab/UNI2-h"  # Hugging Face model path

print(f"Using device: {DEVICE}")

# Login to HuggingFace (if needed)
HF_TOKEN = "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"  # Replace with your token
login(token=HF_TOKEN)

# Load metadata
train_meta = pd.read_csv('train_meta.csv')
val_meta = pd.read_csv('val_meta.csv')
test_meta = pd.read_csv('test_meta.csv')

print(f"Training samples: {len(train_meta)}")
print(f"Validation samples: {len(val_meta)}")
print(f"Test samples: {len(test_meta)}")

class FeatureExtractor(nn.Module):
    """
    Feature extractor using UNI2-h backbone with one layer unfrozen
    """
    def __init__(self, model_name="hf_hub:MahmoodLab/UNI2-h", **timm_kwargs):
        super(FeatureExtractor, self).__init__()
        
        # Initialize the UNI2-h model from HuggingFace using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            **timm_kwargs
        )
        
        # Get output features dimension
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        else:
            self.feature_dim = 1536  # Default for UNI2-h
            
        print(f"Feature extractor output dimension: {self.feature_dim}")
        
        # First freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last transformer block
        if hasattr(self.backbone, 'blocks'):
            num_blocks = len(self.backbone.blocks)
            last_block_idx = num_blocks - 1
            
            # Unfreeze only the last block
            for param in self.backbone.blocks[last_block_idx].parameters():
                param.requires_grad = True
            
            print(f"Unfrozen last transformer block (block {last_block_idx})")
        
        # Always unfreeze the positional embedding and patch embedding
        if hasattr(self.backbone, 'pos_embed'):
            self.backbone.pos_embed.requires_grad = True
            print("Positional embedding unfrozen")
        
        if hasattr(self.backbone, 'patch_embed'):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = True
            print("Patch embedding unfrozen")
    
    def forward(self, x):
        """
        Extract features from input images
        
        Args:
            x: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            features: Extracted features
        """
        features = self.backbone(x)
        return features

class PathologyDataset(Dataset):
    def __init__(self, meta_df, split):
        """
        Dataset for pathology images without augmentation
        
        Args:
            meta_df: DataFrame with image_id and label
            split: 'train', 'val', or 'test'
        """
        self.meta_df = meta_df
        self.split = split
        self.has_labels = 'label' in meta_df.columns
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        image_id = row['image_id']
        
        # Load image
        try:
            file_path = f"{self.split}/{image_id}.pkl"
            img_tensor = torch.load(file_path)
            
            # Remove batch dimension if present
            if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)  # (3, 224, 224)
            
            # Ensure we have a 3D tensor
            if img_tensor.ndim != 3:
                print(f"Warning: Unusual tensor shape {img_tensor.shape} for {image_id}")
                img_tensor = torch.zeros((3, 224, 224))
            
        except Exception as e:
            print(f"Error loading {image_id}: {e}")
            img_tensor = torch.zeros((3, 224, 224))
        
        # Create return dictionary
        result = {
            'image_id': image_id,
            'image': img_tensor
        }
        
        # Add label if available
        if self.has_labels:
            label = row['label']
            result['label'] = torch.tensor(label, dtype=torch.long)
        
        return result

def extract_features(feature_extractor, dataloader, device):
    """
    Extract features using the feature extractor
    
    Args:
        feature_extractor: Feature extraction model
        dataloader: DataLoader for images
        device: Device to run extraction on
        
    Returns:
        features: Extracted features
        labels: Corresponding labels (if available)
        image_ids: Corresponding image IDs
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    all_image_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Get data
            images = batch['image'].to(device)
            
            # Extract features
            features = feature_extractor(images).cpu().numpy()
            
            # Store features
            all_features.append(features)
            
            # Store labels if available
            if 'label' in batch:
                all_labels.append(batch['label'].numpy())
            
            # Store image IDs
            all_image_ids.extend(batch['image_id'])
    
    # Concatenate all features and labels
    all_features = np.vstack(all_features)
    
    if all_labels:
        all_labels = np.concatenate(all_labels)
    else:
        all_labels = None
    
    return all_features, all_labels, all_image_ids

def train_ml_classifiers(X_train, y_train, X_val, y_val, save_dir='ml_models'):
    """
    Train XGBoost, Random Forest, and SVM on extracted features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        save_dir: Directory to save models and results
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    # 1. Train XGBoost
    print("\nTraining XGBoost...")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define parameters - adjusted for compatibility
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 5,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'scale_pos_weight': 5,
        'seed': 42
    }
    
    # Create evaluation list
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    # Train model
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=True
    )
    
    # Evaluate XGBoost
    y_pred_xgb = (xgb_model.predict(dval) > 0.5).astype(int)
    y_prob_xgb = xgb_model.predict(dval)
    
    # Calculate metrics
    accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
    precision_xgb = precision_score(y_val, y_pred_xgb, zero_division=0)
    recall_xgb = recall_score(y_val, y_pred_xgb, zero_division=0)
    f1_xgb = f1_score(y_val, y_pred_xgb, zero_division=0)
    cm_xgb = confusion_matrix(y_val, y_pred_xgb)
    
    # Store results
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_xgb,
        'precision': precision_xgb,
        'recall': recall_xgb,
        'f1': f1_xgb,
        'confusion_matrix': cm_xgb,
        'prob': y_prob_xgb
    }
    
    # Print results
    print(f"\nXGBoost Results:")
    print(f"Accuracy: {accuracy_xgb:.4f}")
    print(f"Precision: {precision_xgb:.4f}")
    print(f"Recall: {recall_xgb:.4f}")
    print(f"F1 Score: {f1_xgb:.4f}")
    print("Confusion Matrix:")
    print(cm_xgb)
    
    # 2. Train Random Forest
    print("\nTraining Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_val)
    y_prob_rf = rf_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy_rf = accuracy_score(y_val, y_pred_rf)
    precision_rf = precision_score(y_val, y_pred_rf, zero_division=0)
    recall_rf = recall_score(y_val, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_val, y_pred_rf, zero_division=0)
    cm_rf = confusion_matrix(y_val, y_pred_rf)
    
    # Store results
    results['RandomForest'] = {
        'model': rf_model,
        'accuracy': accuracy_rf,
        'precision': precision_rf,
        'recall': recall_rf,
        'f1': f1_rf,
        'confusion_matrix': cm_rf,
        'prob': y_prob_rf
    }
    
    # Print results
    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_rf:.4f}")
    print(f"Precision: {precision_rf:.4f}")
    print(f"Recall: {recall_rf:.4f}")
    print(f"F1 Score: {f1_rf:.4f}")
    print("Confusion Matrix:")
    print(cm_rf)
    
    # 3. Train SVM
    print("\nTraining SVM...")
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    svm_model = SVC(
        C=10,
        kernel='linear',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate SVM
    y_pred_svm = svm_model.predict(X_val_scaled)
    y_prob_svm = svm_model.predict_proba(X_val_scaled)[:, 1]
    
    # Calculate metrics
    accuracy_svm = accuracy_score(y_val, y_pred_svm)
    precision_svm = precision_score(y_val, y_pred_svm, zero_division=0)
    recall_svm = recall_score(y_val, y_pred_svm, zero_division=0)
    f1_svm = f1_score(y_val, y_pred_svm, zero_division=0)
    cm_svm = confusion_matrix(y_val, y_pred_svm)
    
    # Store results
    results['SVM'] = {
        'model': svm_model,
        'scaler': scaler,
        'accuracy': accuracy_svm,
        'precision': precision_svm,
        'recall': recall_svm,
        'f1': f1_svm,
        'confusion_matrix': cm_svm,
        'prob': y_prob_svm
    }
    
    # Print results
    print(f"\nSVM Results:")
    print(f"Accuracy: {accuracy_svm:.4f}")
    print(f"Precision: {precision_svm:.4f}")
    print(f"Recall: {recall_svm:.4f}")
    print(f"F1 Score: {f1_svm:.4f}")
    print("Confusion Matrix:")
    print(cm_svm)
    
    # Save models
    with open(os.path.join(save_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open(os.path.join(save_dir, 'randomforest_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(os.path.join(save_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump((svm_model, scaler), f)
    
    # Create comparison plot
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    precisions = [results[model]['precision'] for model in models]
    recalls = [results[model]['recall'] for model in models]
    f1_scores = [results[model]['f1'] for model in models]
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    plt.bar(x - 0.5*width, precisions, width, label='Precision')
    plt.bar(x + 0.5*width, recalls, width, label='Recall')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1_Score': f1_scores
    })
    results_df.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False)
    
    # Determine best model
    best_idx = np.argmax(f1_scores)
    best_model_name = models[best_idx]
    best_model_info = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} with F1 score: {f1_scores[best_idx]:.4f}")
    
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    
    for model_name in models:
        model_info = results[model_name]
        fpr, tpr, _ = roc_curve(y_val, model_info['prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    
    return results

def main():
    # Define UNI2-h model parameters
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 16,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    
    # Create feature extractor with one layer unfrozen
    feature_extractor = FeatureExtractor(model_name="hf_hub:MahmoodLab/UNI2-h", **timm_kwargs)
    feature_extractor = feature_extractor.to(DEVICE)
    
    # Create datasets
    train_dataset = PathologyDataset(train_meta, 'train')
    val_dataset = PathologyDataset(val_meta, 'val')
    test_dataset = PathologyDataset(test_meta, 'test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # No need to shuffle when extracting features
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Extract features
    print("\nExtracting training features...")
    X_train, y_train, train_ids = extract_features(feature_extractor, train_loader, DEVICE)
    
    print("\nExtracting validation features...")
    X_val, y_val, val_ids = extract_features(feature_extractor, val_loader, DEVICE)
    
    print("\nExtracting test features...")
    X_test, y_test, test_ids = extract_features(feature_extractor, test_loader, DEVICE)
    
    # Print feature shapes
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Save extracted features
    os.makedirs('features', exist_ok=True)
    
    np.save('features/X_train.npy', X_train)
    np.save('features/y_train.npy', y_train)
    np.save('features/X_val.npy', X_val)
    np.save('features/y_val.npy', y_val)
    np.save('features/X_test.npy', X_test)
    
    if y_test is not None:
        np.save('features/y_test.npy', y_test)
    
    # Save image IDs
    with open('features/train_ids.pkl', 'wb') as f:
        pickle.dump(train_ids, f)
    
    with open('features/val_ids.pkl', 'wb') as f:
        pickle.dump(val_ids, f)
    
    with open('features/test_ids.pkl', 'wb') as f:
        pickle.dump(test_ids, f)
    
    print("Features saved to 'features' directory")
    
    # Train the three ML classifiers
    ml_results = train_ml_classifiers(X_train, y_train, X_val, y_val)
    
    # Evaluate best model on test set if labels are available
    if y_test is not None:
        print("\nEvaluating models on test set...")
        
        # XGBoost evaluation
        xgb_model = ml_results['XGBoost']['model']
        dtest = xgb.DMatrix(X_test, label=y_test)
        y_pred_xgb = (xgb_model.predict(dtest) > 0.5).astype(int)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
        
        print(f"XGBoost Test - Accuracy: {accuracy_xgb:.4f}, F1: {f1_xgb:.4f}")
        
        # Random Forest evaluation
        rf_model = ml_results['RandomForest']['model']
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
        
        print(f"Random Forest Test - Accuracy: {accuracy_rf:.4f}, F1: {f1_rf:.4f}")
        
        # SVM evaluation
        svm_model = ml_results['SVM']['model']
        scaler = ml_results['SVM']['scaler']
        X_test_scaled = scaler.transform(X_test)
        y_pred_svm = svm_model.predict(X_test_scaled)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        f1_svm = f1_score(y_test, y_pred_svm, zero_division=0)
        
        print(f"SVM Test - Accuracy: {accuracy_svm:.4f}, F1: {f1_svm:.4f}")
    else:
        # Generate predictions for unlabeled test set using all models
        xgb_model = ml_results['XGBoost']['model']
        rf_model = ml_results['RandomForest']['model']
        svm_model = ml_results['SVM']['model']
        scaler = ml_results['SVM']['scaler']
        
        # XGBoost predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_xgb = (xgb_model.predict(dtest) > 0.5).astype(int)
        y_prob_xgb = xgb_model.predict(dtest)
        
        # Random Forest predictions
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        
        # SVM predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred_svm = svm_model.predict(X_test_scaled)
        y_prob_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
        
        # Create submission DataFrames
        submission_xgb = pd.DataFrame({
            'image_id': test_ids,
            'prediction': y_pred_xgb,
            'probability': y_prob_xgb
        })
        
        submission_rf = pd.DataFrame({
            'image_id': test_ids,
            'prediction': y_pred_rf,
            'probability': y_prob_rf
        })
        
        submission_svm = pd.DataFrame({
            'image_id': test_ids,
            'prediction': y_pred_svm,
            'probability': y_prob_svm
        })
        
        # Save submissions
        submission_xgb.to_csv('xgboost_predictions.csv', index=False)
        submission_rf.to_csv('randomforest_predictions.csv', index=False)
        submission_svm.to_csv('svm_predictions.csv', index=False)
        
        print("\nTest predictions saved to CSV files")

if __name__ == "__main__":
    main()
