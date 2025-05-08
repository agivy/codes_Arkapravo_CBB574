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
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from huggingface_hub import login
from prettytable import PrettyTable
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
HF_TOKEN = "token"  # Replace with your token
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
    Train multiple ML classifiers on extracted features
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        save_dir: Directory to save models and results
        
    Returns:
        dict: Results and models
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    # Scale features for models that are sensitive to feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler for later use
    with open(os.path.join(save_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Define all classifiers to train
    classifiers = {
        'XGBoost': {
            'train_func': train_xgboost,
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'error',
                'max_depth': 5,
                'eta': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'scale_pos_weight': 5,
                'seed': 42
            },
            'scaled': False
        },
        'RandomForest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'scaled': False
        },
        'SVM': {
            'model': SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'scaled': True
        },
        'KNN': {
            'model': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski',
                p=2,  # Euclidean distance
                n_jobs=-1
            ),
            'scaled': True
        },
        'NaiveBayes': {
            'model': GaussianNB(),
            'scaled': True
        },
        'LogisticRegression': {
            'model': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'scaled': True
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'scaled': False
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'scaled': False
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(
                max_depth=5,
                class_weight='balanced',
                random_state=42
            ),
            'scaled': False
        },
        'LDA': {
            'model': LinearDiscriminantAnalysis(),
            'scaled': True
        }
    }
    
    # Initialize metric collectors
    all_metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Specificity': [],
        'NPV': [],  # Negative Predictive Value
        'AUC': []
    }
    
    # Train XGBoost separately due to its different API
    print("\nTraining XGBoost...")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, classifiers['XGBoost']['params'])
    results['XGBoost'] = {
        'model': xgb_model,
        **xgb_metrics
    }
    
    # # Add XGBoost metrics to the collector
    # for key in all_metrics.keys():
    #     if key == 'Model':
    #         all_metrics[key].append('XGBoost')
    #     elif key in xgb_metrics:
    #         all_metrics[key].append(xgb_metrics[key])
    
    # Add XGBoost metrics to the collector
    all_metrics['Model'].append('XGBoost')
    all_metrics['Accuracy'].append(xgb_metrics['accuracy'])
    all_metrics['Precision'].append(xgb_metrics['precision'])
    all_metrics['Recall'].append(xgb_metrics['recall'])
    all_metrics['F1 Score'].append(xgb_metrics['f1'])
    all_metrics['Specificity'].append(xgb_metrics['specificity'])
    all_metrics['NPV'].append(xgb_metrics['npv'])
    all_metrics['AUC'].append(xgb_metrics['auc'])
    
    # Train all other classifiers
    for name, clf_info in classifiers.items():
        if name == 'XGBoost':
            continue  # Already trained
        
        print(f"\nTraining {name}...")
        
        # Get the appropriate training data (scaled or not)
        X_train_use = X_train_scaled if clf_info['scaled'] else X_train
        X_val_use = X_val_scaled if clf_info['scaled'] else X_val
        
        # Train and evaluate
        model = clf_info['model']
        model.fit(X_train_use, y_train)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val_use)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_val_use)
            y_prob = y_pred  # For models without probability output
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred, y_prob)
        
        # Store results
        results[name] = {
            'model': model,
            **metrics
        }
        
        # If the model uses scaled features, store the scaler
        if clf_info['scaled']:
            results[name]['scaler'] = scaler
        
        # Add metrics to the collector
        for key in all_metrics.keys():
            if key == 'Model':
                all_metrics[key].append(name)
            elif key in metrics:
                all_metrics[key].append(metrics[key])
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Save model
        with open(os.path.join(save_dir, f'{name.lower()}_model.pkl'), 'wb') as f:
            if clf_info['scaled']:
                pickle.dump((model, scaler), f)
            else:
                pickle.dump(model, f)
    
    # Create a formatted table for easy comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Sort by F1 Score descending
    metrics_df = metrics_df.sort_values(by='F1 Score', ascending=False)
    
    # Print as a formatted table
    table = PrettyTable()
    table.field_names = metrics_df.columns.tolist()
    for _, row in metrics_df.iterrows():
        formatted_row = [row['Model']]
        for col in metrics_df.columns[1:]:  # Skip 'Model'
            formatted_row.append(f"{row[col]:.4f}")
        table.add_row(formatted_row)
    
    print(table)
    print("="*80)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(save_dir, 'all_model_metrics.csv'), index=False)
    
    # Create comparison plot
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(metrics_df))
    width = 0.15  # Narrower bars to fit all metrics
    
    plt.bar(x - 2*width, metrics_df['Accuracy'], width, label='Accuracy')
    plt.bar(x - width, metrics_df['Precision'], width, label='Precision')
    plt.bar(x, metrics_df['Recall'], width, label='Recall')
    plt.bar(x + width, metrics_df['F1 Score'], width, label='F1 Score')
    plt.bar(x + 2*width, metrics_df['AUC'], width, label='AUC')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        if 'prob' in result and hasattr(result['prob'], '__iter__'):
            fpr, tpr, _ = roc_curve(y_val, result['prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
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
    
    # Create heatmap of model performance
    plt.figure(figsize=(12, 8))
    metrics_heatmap = metrics_df.drop(columns=['Model']).copy()
    metrics_heatmap.index = metrics_df['Model']
    
    sns.heatmap(metrics_heatmap, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Model Performance Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'))
    
    # Determine best model
    best_model_name = metrics_df.iloc[0]['Model']  # Top model after sorting by F1
    best_model_f1 = metrics_df.iloc[0]['F1 Score']
    
    print(f"\nBest model: {best_model_name} with F1 score: {best_model_f1:.4f}")
    
    return results

def train_xgboost(X_train, y_train, X_val, y_val, params):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters
        
    Returns:
        tuple: (model, metrics)
    """
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Create evaluation list
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=True
    )
    
    # Evaluate XGBoost
    y_prob = model.predict(dval)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_prob)
    
    return model, metrics

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Probability predictions (optional)
        
    Returns:
        dict: Metrics
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate AUC if probabilities provided
    auc_score = 0
    if y_prob is not None and hasattr(y_prob, '__iter__'):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'auc': auc_score,
        'specificity': specificity,
        'npv': npv,
        'prob': y_prob
    }

def evaluate_models_on_test(results, X_test, y_test=None, save_dir='ml_models'):
    """
    Evaluate all trained models on test set
    
    Args:
        results: Dictionary of trained models and their results
        X_test: Test features
        y_test: Test labels (optional)
        save_dir: Directory to save results
        
    Returns:
        pd.DataFrame: Test metrics for all models
    """
    if y_test is None:
        print("\nNo test labels provided. Generating predictions only...")
        return generate_test_predictions(results, X_test, save_dir)
    
    print("\nEvaluating all models on test set...")
    
    # Initialize metrics collector
    test_metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Specificity': [],
        'NPV': [],
        'AUC': []
    }
    
    # Evaluate each model
    for name, result in results.items():
        model = result['model']
        
        # Handle different model types
        if name == 'XGBoost':
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_prob = model.predict(dtest)
            y_pred = (y_prob > 0.5).astype(int)
        elif name == 'SVM' or 'scaler' in result:
            # Model uses scaled features
            scaler = result['scaler']
            X_test_scaled = scaler.transform(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test_scaled)
                y_prob = y_pred  # For models without probability output
        else:
            # Model uses unscaled features
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_prob = y_pred  # For models without probability output
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Add metrics to collector
        test_metrics['Model'].append(name)
        test_metrics['Accuracy'].append(metrics['accuracy'])
        test_metrics['Precision'].append(metrics['precision'])
        test_metrics['Recall'].append(metrics['recall'])
        test_metrics['F1 Score'].append(metrics['f1'])
        test_metrics['Specificity'].append(metrics['specificity'])
        test_metrics['NPV'].append(metrics['npv'])
        test_metrics['AUC'].append(metrics['auc'])
        
        # Print detailed report
        print(f"\n{name} Test Results:")
        print("="*50)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        print(f"AUC:         {metrics['auc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"NPV:         {metrics['npv']:.4f}")
        
        # Print confusion matrix
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives: {tp}, False Positives: {fp}")
        print(f"True Negatives: {tn}, False Negatives: {fn}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        print("="*50)
    
    # Create DataFrame for test metrics
    test_df = pd.DataFrame(test_metrics)
    
    # Sort by F1 Score
    test_df = test_df.sort_values(by='F1 Score', ascending=False)
    
    # Save test metrics to CSV
    test_df.to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)
    
    # Print formatted table
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE COMPARISON")
    print("="*80)
    
    table = PrettyTable()
    table.field_names = test_df.columns.tolist()
    for _, row in test_df.iterrows():
        formatted_row = [row['Model']]
        for col in test_df.columns[1:]:  # Skip 'Model'
            formatted_row.append(f"{row[col]:.4f}")
        table.add_row(formatted_row)
    
    print(table)
    print("="*80)
    
    # Create comparison plot for test results
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(test_df))
    width = 0.15
    
    plt.bar(x - 2*width, test_df['Accuracy'], width, label='Accuracy')
    plt.bar(x - width, test_df['Precision'], width, label='Precision')
    plt.bar(x, test_df['Recall'], width, label='Recall')
    plt.bar(x + width, test_df['F1 Score'], width, label='F1 Score')
    plt.bar(x + 2*width, test_df['AUC'], width, label='AUC')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance on Test Set')
    plt.xticks(x, test_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results_comparison.png'))
    
    # Create heatmap of test performance
    plt.figure(figsize=(12, 8))
    metrics_heatmap = test_df.drop(columns=['Model']).copy()
    metrics_heatmap.index = test_df['Model']
    
    sns.heatmap(metrics_heatmap, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Test Set Performance Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_metrics_heatmap.png'))
    
    return test_df

def generate_test_predictions(results, X_test, save_dir='ml_models'):
    """
    Generate predictions for unlabeled test set
    
    Args:
        results: Dictionary of trained models and their results
        X_test: Test features
        save_dir: Directory to save predictions
        
    Returns:
        dict: DataFrames with predictions for each model
    """
    print("\nGenerating predictions for test set...")
    predictions = {}
    
    # Load image IDs
    with open('features/test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    # Generate predictions for each model
    for name, result in results.items():
        model = result['model']
        
        # Handle different model types
        if name == 'XGBoost':
            dtest = xgb.DMatrix(X_test)
            y_prob = model.predict(dtest)
            y_pred = (y_prob > 0.5).astype(int)
        elif name == 'SVM' or 'scaler' in result:
            # Model uses scaled features
            scaler = result['scaler']
            X_test_scaled = scaler.transform(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test_scaled)
                y_prob = y_pred
        else:
            # Model uses unscaled features
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_prob = y_pred
        
        # Continuation of generate_test_predictions function
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'image_id': test_ids,
            'prediction': y_pred,
            'probability': y_prob
        })
        
        # Save predictions
        predictions_df.to_csv(os.path.join(save_dir, f'{name.lower()}_predictions.csv'), index=False)
        
        # Store DataFrame
        predictions[name] = predictions_df
        
        print(f"Generated and saved predictions for {name}")
    
    # Create ensemble prediction (majority voting)
    ensemble_pred = np.zeros(len(test_ids))
    ensemble_prob = np.zeros(len(test_ids))
    
    for name, result in results.items():
        # Get predictions for this model
        model_pred = predictions[name]['prediction'].values
        
        # Add to ensemble prediction (voting)
        ensemble_pred += model_pred
        
        # If probabilities available, add to ensemble probability
        if hasattr(predictions[name]['probability'], '__iter__'):
            ensemble_prob += predictions[name]['probability'].values
    
    # Average the probabilities
    ensemble_prob /= len(results)
    
    # Majority vote for final prediction
    ensemble_pred = (ensemble_pred > (len(results) / 2)).astype(int)
    
    # Create DataFrame for ensemble predictions
    ensemble_df = pd.DataFrame({
        'image_id': test_ids,
        'prediction': ensemble_pred,
        'probability': ensemble_prob
    })
    
    # Save ensemble predictions
    ensemble_df.to_csv(os.path.join(save_dir, 'ensemble_majority_vote_predictions.csv'), index=False)
    
    print("Generated and saved ensemble predictions (majority voting)")
    
    # Add ensemble to predictions dictionary
    predictions['Ensemble_Majority_Vote'] = ensemble_df
    
    return predictions

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
    
    # Check if features already exist
    if os.path.exists('features/X_train.npy'):
        print("Loading pre-extracted features...")
        X_train = np.load('features/X_train.npy')
        y_train = np.load('features/y_train.npy')
        X_val = np.load('features/X_val.npy')
        y_val = np.load('features/y_val.npy')
        X_test = np.load('features/X_test.npy')
        
        # Load test labels if available
        if os.path.exists('features/y_test.npy'):
            y_test = np.load('features/y_test.npy')
        else:
            y_test = None
            
        # Load image IDs
        with open('features/train_ids.pkl', 'rb') as f:
            train_ids = pickle.load(f)
        
        with open('features/val_ids.pkl', 'rb') as f:
            val_ids = pickle.load(f)
        
        with open('features/test_ids.pkl', 'rb') as f:
            test_ids = pickle.load(f)
            
        print(f"Loaded features - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    else:
        # Create feature extractor
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
    
    # Train all ML classifiers
    save_dir = 'expanded_ml_models'
    ml_results = train_ml_classifiers(X_train, y_train, X_val, y_val, save_dir=save_dir)
    
    # Evaluate on test set or generate predictions
    if y_test is not None:
        test_metrics = evaluate_models_on_test(ml_results, X_test, y_test, save_dir=save_dir)
        
        # Print best model on test set
        best_test_model = test_metrics.iloc[0]['Model']
        best_test_f1 = test_metrics.iloc[0]['F1 Score']
        print(f"\nBest model on test set: {best_test_model} with F1 score: {best_test_f1:.4f}")
    else:
        # Generate predictions for unlabeled test set
        predictions = generate_test_predictions(ml_results, X_test, save_dir=save_dir)
        print("\nAll predictions saved to CSV files")

if __name__ == "__main__":
    main()