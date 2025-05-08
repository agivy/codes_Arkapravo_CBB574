import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import pickle
from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import login
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define constants
BATCH_SIZE = 16
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "MahmoodLab/UNI2-h"  # Hugging Face model path

print(f"Using device: {DEVICE}")

# Login to HuggingFace
HF_TOKEN = "token"  # Replace with your token
login(token=HF_TOKEN)

# Load metadata
train_meta = pd.read_csv('train_meta.csv')
val_meta = pd.read_csv('val_meta.csv')
test_meta = pd.read_csv('test_meta.csv')

print(f"Training samples: {len(train_meta)}")
print(f"Validation samples: {len(val_meta)}")
print(f"Test samples: {len(test_meta)}")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torchvision import transforms
import random
from PIL import Image
import cv2
import os
import pandas as pd

# Custom TensorAugmentation class specifically for histopathology images
import random
import torch
from torchvision import transforms

class HistoAugmentation:
    def __init__(self, p=0.5, rotation_range=15, flip=True, crop_scale=(0.8, 1.0), 
                 normalize=True, mean=None, std=None):
        """
        Biologically meaningful augmentation for histopathology images
        
        Args:
            p: Probability of applying each augmentation
            rotation_range: Range for random rotation (degrees)
            flip: Whether to apply random flips
            crop_scale: Range of random crop scale (simulates different magnifications)
            normalize: Whether to normalize the images
            mean: Mean for normalization (default None means no normalization)
            std: Standard deviation for normalization (default None means no normalization)
        """
        self.p = p
        self.rotation_range = rotation_range
        self.flip = flip
        self.crop_scale = crop_scale
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        # Ensure tensor is on CPU for transformations
        device = tensor.device
        tensor = tensor.cpu()
        
        # Random rotation - simulates slide orientation differences
        if self.rotation_range > 0 and random.random() < self.p:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            # Convert to PIL for rotation
            img = transforms.ToPILImage()(tensor)
            img = transforms.functional.rotate(img, angle)
            tensor = transforms.ToTensor()(img)
        
        # Random horizontal flip - tissue has no inherent orientation
        if self.flip and random.random() < self.p:
            tensor = tensor.flip(dims=[2])
        
        # Random vertical flip - tissue has no inherent orientation
        if self.flip and random.random() < self.p:
            tensor = tensor.flip(dims=[1])
        
        # Normalize if requested
        if self.normalize:
            if self.mean is None or self.std is None:
                # Default: Normalize to [0, 1] range (optional for minimal difference)
                mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
                std = torch.std(tensor, dim=(1, 2), keepdim=True)
            else:
                mean = torch.tensor(self.mean).view(3, 1, 1)
                std = torch.tensor(self.std).view(3, 1, 1)

            tensor = (tensor - mean) / std
        
        # Return tensor to original device if needed
        return tensor.to(device)

class ClassBalancedPathologyDataset(Dataset):
    def __init__(self, meta_df, split, transform=None, target_size=10000):
        """
        Class-balanced dataset for pathology images using simple augmentation
        
        Args:
            meta_df: DataFrame with image_id and label
            split: 'train', 'val', or 'test'
            transform: Base transforms for all images
            target_size: Target total dataset size after augmentation
        """
        self.meta_df = meta_df
        self.split = split
        self.transform = transform
        self.has_labels = 'label' in meta_df.columns
        self.target_size = target_size
        
        # Analyze class distribution
        if self.has_labels:
            self.original_samples = []
            self.class_counts = meta_df['label'].value_counts().to_dict()
            self.class_indices = {}
            self.augmented_indices = []
            self.all_images = []
            self.all_labels = []
            
            # Identify tumor class (assuming binary classification: 0=normal, 1=tumor)
            self.tumor_class = min(self.class_counts, key=self.class_counts.get)
            self.normal_class = 1 if self.tumor_class == 0 else 0
            
            # Calculate augmentation factors for each class
            total_original = len(meta_df)
            
            # Get class indices
            for cls in self.class_counts.keys():
                self.class_indices[cls] = meta_df[meta_df['label'] == cls].index.tolist()
            
            # Determine augmentation needed to balance classes
            tumor_count = self.class_counts[self.tumor_class]
            normal_count = self.class_counts[self.normal_class]
            
            # Target an equal class distribution
            target_per_class = target_size // 2
            
            # Calculate needed total tumor and normal samples
            total_target_tumor = target_per_class
            total_target_normal = target_per_class
            
            # Calculate needed augmented samples
            needed_tumor_aug = total_target_tumor - tumor_count
            needed_normal_aug = total_target_normal - normal_count
            
            # Calculate augmentations per sample (rounded up to ensure we reach targets)
            tumor_augmentations_per_sample = int(np.ceil(needed_tumor_aug / tumor_count))
            normal_augmentations_per_sample = int(np.ceil(needed_normal_aug / normal_count))
            
            # Class ratios (65-35 split based on your description)
            class_ratio = normal_count / tumor_count
            print(f"Original class imbalance ratio: {class_ratio:.2f}:1")
            print(f"Original samples: {total_original}")
            print(f"Target size: {target_size}")
            print(f"Normal class: {normal_count} samples")
            print(f"Tumor class: {tumor_count} samples")
            print(f"Needed tumor augmentations: {needed_tumor_aug}")
            print(f"Needed normal augmentations: {needed_normal_aug}")
            print(f"Augmentations per tumor sample: {tumor_augmentations_per_sample}")
            print(f"Augmentations per normal sample: {normal_augmentations_per_sample}")
            
            # Predefined augmentation parameters - biologically meaningful transformations
            # Each tuple is (rotation angle, horizontal flip, vertical flip)
            augmentation_params = [
                (90, False, False),
                (180, False, False),
                (270, False, False),
                (0, True, False),
                (0, False, True),
                (0, True, True),
                (90, True, False),
                (90, False, True),
                (270, True, False),
                (270, False, True)
            ]
            
            # Build dataset index
            # First add all original samples
            for i in range(len(meta_df)):
                row = meta_df.iloc[i]
                self.original_samples.append({
                    'index': i,
                    'image_id': row['image_id'],
                    'label': row['label'],
                    'aug_idx': 0,
                    'rotation': 0,
                    'flip_h': False,
                    'flip_v': False
                })
                
                self.all_images.append(self.original_samples[-1])
                self.all_labels.append(row['label'])
            
            # Add augmented tumor samples
            tumor_aug_count = 0
            for i in self.class_indices[self.tumor_class]:
                row = meta_df.iloc[i]
                
                # Add up to augmentations_per_sample for each tumor image
                for aug_idx in range(1, tumor_augmentations_per_sample + 1):
                    if tumor_aug_count >= needed_tumor_aug:
                        break
                        
                    # Use augmentation parameters in order, cycling if needed
                    param_idx = (aug_idx - 1) % len(augmentation_params)
                    rotation, flip_h, flip_v = augmentation_params[param_idx]
                    
                    self.augmented_indices.append({
                        'index': i,
                        'image_id': row['image_id'],
                        'label': row['label'],
                        'aug_idx': aug_idx,
                        'rotation': rotation,
                        'flip_h': flip_h,
                        'flip_v': flip_v
                    })
                    
                    self.all_images.append(self.augmented_indices[-1])
                    self.all_labels.append(row['label'])
                    tumor_aug_count += 1
                    
                    if tumor_aug_count >= needed_tumor_aug:
                        break
            
            # Add augmented normal samples
            normal_aug_count = 0
            for i in self.class_indices[self.normal_class]:
                row = meta_df.iloc[i]
                
                # Add up to augmentations_per_sample for each normal image
                for aug_idx in range(1, normal_augmentations_per_sample + 1):
                    if normal_aug_count >= needed_normal_aug:
                        break
                        
                    # Use augmentation parameters in order, cycling if needed
                    param_idx = (aug_idx - 1) % len(augmentation_params)
                    rotation, flip_h, flip_v = augmentation_params[param_idx]
                    
                    self.augmented_indices.append({
                        'index': i,
                        'image_id': row['image_id'],
                        'label': row['label'],
                        'aug_idx': aug_idx,
                        'rotation': rotation,
                        'flip_h': flip_h,
                        'flip_v': flip_v
                    })
                    
                    self.all_images.append(self.augmented_indices[-1])
                    self.all_labels.append(row['label'])
                    normal_aug_count += 1
                    
                    if normal_aug_count >= needed_normal_aug:
                        break
            
            print(f"Total dataset size after augmentation: {len(self.all_images)}")
            
            # Display final class distribution
            final_counts = {cls: self.all_labels.count(cls) for cls in self.class_counts.keys()}
            print("Final class distribution:")
            for cls, count in final_counts.items():
                class_name = "Tumor" if cls == self.tumor_class else "Normal"
                print(f"Class {cls} ({class_name}): {count} samples ({count/len(self.all_images)*100:.1f}%)")
        else:
            # For test set without labels
            self.all_images = [{'index': i, 'image_id': meta_df.iloc[i]['image_id'], 'aug_idx': 0,
                               'rotation': 0, 'flip_h': False, 'flip_v': False} 
                              for i in range(len(meta_df))]
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        # Get sample info
        sample = self.all_images[idx]
        orig_idx = sample['index']
        aug_idx = sample['aug_idx']
        image_id = sample['image_id']
        rotation = sample['rotation']
        flip_h = sample['flip_h']
        flip_v = sample['flip_v']
        
        # Load original image
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
        
        # Apply augmentation for non-original samples
        if aug_idx > 0:
            try:
                # Apply rotation if needed
                if rotation != 0:
                    # Convert to PIL for rotation
                    img = transforms.ToPILImage()(img_tensor)
                    img = transforms.functional.rotate(img, rotation)
                    img_tensor = transforms.ToTensor()(img)
                
                # Apply horizontal flip if needed
                if flip_h:
                    img_tensor = img_tensor.flip(dims=[2])
                
                # Apply vertical flip if needed
                if flip_v:
                    img_tensor = img_tensor.flip(dims=[1])
                
            except Exception as e:
                print(f"Augmentation error for {image_id}: {e}")
        
        # Apply base transform to all images
        if self.transform is not None:
            try:
                img_tensor = self.transform(img_tensor)
            except Exception as e:
                print(f"Transform error for {image_id}: {e}")
        
        # Create return dictionary
        result = {
            'image': img_tensor,
            'image_id': image_id
        }
        
        # Add label if available
        if self.has_labels:
            result['label'] = torch.tensor(sample['label'], dtype=torch.long)
        
        # Add index for reference
        result['idx'] = torch.tensor(idx, dtype=torch.long)
        
        return result

# Function to load images (original from your code)
def load_image(image_id, split):
    """
    Load an image file using torch.load
    
    Args:
        image_id: ID of the image
        split: 'train', 'val', or 'test' folder
        
    Returns:
        tensor: PyTorch tensor of the image or None on error
    """
    file_path = f"{split}/{image_id}.pkl"
    
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    
    try:
        # Load using torch.load
        img = torch.load(file_path)
        
        # Remove batch dimension if present
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)  # (3, 224, 224)
            
        return img
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Original PathologyDataset (keeping for validation/test)
class PathologyDataset(Dataset):
    def __init__(self, meta_df, split, transform=None):
        """
        Dataset for pathology images
        
        Args:
            meta_df: DataFrame with image_id and label
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply to images
        """
        self.meta_df = meta_df
        self.split = split
        self.transform = transform
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
        
        # Apply transforms if provided
        if self.transform is not None:
            try:
                img_tensor = self.transform(img_tensor)
            except Exception as e:
                print(f"Transform error for {image_id}: {e}")
        
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

# Define base transforms for normalization based on image statistics (mean and std)
base_transforms = transforms.Compose([
    transforms.Lambda(lambda x: (x - torch.mean(x, dim=(1, 2), keepdim=True)) / torch.std(x, dim=(1, 2), keepdim=True))
])

# Load metadata
train_meta = pd.read_csv('train_meta.csv')
val_meta = pd.read_csv('val_meta.csv')
test_meta = pd.read_csv('test_meta.csv')

# Display class distribution
print("\nOriginal class distribution:")
print("Training set:")
train_class_counts = train_meta['label'].value_counts()
print(train_class_counts)
print("\nValidation set:")
val_class_counts = val_meta['label'].value_counts()
print(val_class_counts)

# # Create balanced datasets with augmentation
# train_dataset = ClassBalancedPathologyDataset(
#     train_meta, 
#     'train', 
#     transform=base_transforms,
#     target_size=500000
# )

# val_dataset = ClassBalancedPathologyDataset(
#     val_meta, 
#     'val', 
#     transform=base_transforms,
#     target_size=500
# )

train_dataset = PathologyDataset(train_meta, 'train', transform=base_transforms)
val_dataset = PathologyDataset(val_meta, 'val', transform=base_transforms)
test_dataset = PathologyDataset(test_meta, 'test', transform=base_transforms)

# # Set batch size and workers
# BATCH_SIZE = 1
# NUM_WORKERS = 4

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # Already balanced through augmentation
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

print("Successfully created balanced dataset with histopathology-appropriate augmentations")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

def calculate_dataset_statistics(train_loader, val_loader):
    """
    Calculate and print the mean and standard deviation statistics for train and validation datasets
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Initialize containers for storing all pixel values
    train_pixels = []
    val_pixels = []
    
    print("Calculating training set statistics...")
    for batch in train_loader:
        # Get images from batch
        images = batch['image']  # Should be tensor of shape (batch_size, channels, height, width)
        
        # Reshape to (batch_size, channels, height*width)
        b, c, h, w = images.shape
        images = images.reshape(b, c, -1)
        
        # Add to list
        train_pixels.append(images)
    
    # Concatenate all batches
    if train_pixels:
        train_pixels = torch.cat(train_pixels, dim=0)
        
        # Calculate statistics per channel
        train_mean = torch.mean(train_pixels, dim=(0, 2))
        train_std = torch.std(train_pixels, dim=(0, 2))
    
    print("Calculating validation set statistics...")
    for batch in val_loader:
        # Get images from batch
        images = batch['image']
        
        # Reshape
        b, c, h, w = images.shape
        images = images.reshape(b, c, -1)
        
        # Add to list
        val_pixels.append(images)
    
    # Concatenate all batches
    if val_pixels:
        val_pixels = torch.cat(val_pixels, dim=0)
        
        # Calculate statistics per channel
        val_mean = torch.mean(val_pixels, dim=(0, 2))
        val_std = torch.std(val_pixels, dim=(0, 2))
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    print("Training Dataset:")
    print(f"Mean per channel: {train_mean.tolist()}")
    print(f"Std per channel: {train_std.tolist()}")
    print("\nValidation Dataset:")
    print(f"Mean per channel: {val_mean.tolist()}")
    print(f"Std per channel: {val_std.tolist()}")
    print("-" * 40)
    
    # Print distribution difference
    mean_diff = torch.abs(train_mean - val_mean).mean().item()
    std_diff = torch.abs(train_std - val_std).mean().item()
    print(f"Mean difference between train and val: {mean_diff:.4f}")
    print(f"Std difference between train and val: {std_diff:.4f}")
    
    return {
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    }

import timm
import torch.nn as nn
import torch
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.scale = None
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        device = x.device
        
        if self.scale is None:
            self.scale = torch.sqrt(torch.FloatTensor([x.size(-1)]).to(device))
        else:
            self.scale = self.scale.to(device)

        query = self.query(x)
        key = self.key(x)
        
        attention_scores = torch.matmul(query.unsqueeze(1), key.unsqueeze(2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        context = attention_weights * x.unsqueeze(1)
        context = context.squeeze(1)
        
        output = self.projection(context)
        
        return output + x  # Residual connection

# Custom layers for reshaping tensors
class UnsqueezeLayer(nn.Module):
    def __init__(self, dim):
        super(UnsqueezeLayer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.unsqueeze(self.dim)

class SqueezeLayer(nn.Module):
    def __init__(self, dim):
        super(SqueezeLayer, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.squeeze(self.dim)

# Custom wrapper for MultiheadAttention to make it work in nn.Sequential
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        # x shape: [batch_size, 1, embed_dim]
        # For self-attention, query, key, and value are all the same
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# Custom Permute Layer
class PermuteLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(PermuteLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
    def forward(self, x):
        return x.permute(self.dim1, self.dim2, self.dim3)

# class TumorClassifier(nn.Module):
#     def __init__(self, num_classes=1, img_size=224, patch_size=16, 
#                  unfreeze_layers=0, dropout_rate=0.95):
#         """
#         Tumor classifier using a pretrained ViT-Small model with selective layer unfreezing
        
#         Args:
#             num_classes: Number of output classes (1 for binary classification)
#             img_size: Input image size
#             patch_size: Size of patches for ViT
#             unfreeze_layers: Number of transformer blocks to unfreeze from the end
#                              0 means freeze all, -1 means unfreeze all
#             dropout_rate: Dropout rate for regularization
#         """
#         super(TumorClassifier, self).__init__()
        
#         # Initialize a pretrained ViT-Small model
#         self.backbone = timm.create_model(
#             'vit_small_patch16_224', 
#             pretrained=True,
#             img_size=img_size,
#             patch_size=patch_size,
#             num_classes=1  # Remove classifier head
#         )
        
#         # # Get output features dimension from ViT-Small
#         hidden_size = self.backbone.num_features  # 384 for ViT-Small
        
#         # First freeze all backbone parameters
#         for param in self.backbone.parameters():
#             param.requires_grad = False
            
#         # Selectively unfreeze the specified number of transformer blocks from the end
#         if unfreeze_layers > 0:
#             # Assuming backbone.blocks is the list of transformer blocks
#             num_blocks = len(self.backbone.blocks)
#             start_idx = max(0, num_blocks - unfreeze_layers)
            
#             # Unfreeze only the last few blocks
#             for i in range(start_idx, num_blocks):
#                 for param in self.backbone.blocks[i].parameters():
#                     param.requires_grad = True
#                 print(f"Unfrozen block {i}")
        
#         # Unfreeze all layers if unfreeze_layers is -1
#         elif unfreeze_layers == -1:
#             for param in self.backbone.parameters():
#                 param.requires_grad = True
#             print("All backbone layers unfrozen")
        
#         # Always unfreeze the positional embedding and patch embedding
#         # This can help adapt the model to your specific image characteristics
#         if hasattr(self.backbone, 'pos_embed'):
#             self.backbone.pos_embed.requires_grad = True
#             print("Positional embedding unfrozen")
        
#         if hasattr(self.backbone, 'patch_embed'):
#             self.backbone.patch_embed.requires_grad = True
#             print("Patch embedding unfrozen")
        
#         # Classification head with strong regularization
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_size, num_classes)
#         )
#         #     nn.ReLU(),
#         #     nn.BatchNorm1d(512),
#         #     nn.Dropout(dropout_rate),
#         #     nn.Linear(512, 256),
#         #     nn.ReLU(),
#         #     nn.BatchNorm1d(256),
#         #     nn.Dropout(dropout_rate),
#         #     nn.Linear(256, num_classes)
#         # )
    
#     def forward(self, x):
#         """
#         Forward pass
        
#         Args:
#             x: Image tensor of shape (batch_size, channels, height, width)
            
#         Returns:
#             logits: Classification logits
#         """
#         # Get features from backbone
#         features = self.backbone(x)
        
#         # Classification
#         # logits = self.classifier(features)
#         return features.squeeze(-1)
        
#         # return logits.squeeze(-1)  # Remove last dimension for binary classification
        
# class TumorClassifier(nn.Module):
#     def __init__(self, num_classes=1, dropout_rate=0.5, input_size=224):
#         super(TumorClassifier, self).__init__()
        
#         # Calculate flattened input size
#         flattened_size = 3 * input_size * input_size  # 3 channels * height * width
        
#         # Create a small MLP backbone
#         self.backbone = nn.Sequential(
#             nn.Flatten(),  # Flatten the input image
#             nn.Linear(flattened_size, 1024),
#             nn.ReLU(),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(dropout_rate),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         # Pass through the MLP backbone
#         features = self.backbone(x)
#         return features.squeeze(-1)

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=1, unfreeze_layers=1, dropout_rate=0.7, 
                 model_name="hf_hub:MahmoodLab/UNI2-h", freeze_backbone=True, **timm_kwargs):
        """
        Tumor classifier using a pretrained UNI2-h model with selective layer unfreezing
        
        Args:
            num_classes: Number of output classes (1 for binary classification)
            unfreeze_layers: Number of transformer blocks to unfreeze from the end
                            0 means freeze all, -1 means unfreeze all
            dropout_rate: Dropout rate for regularization
            model_name: Model name or path to load from HuggingFace or timm
            freeze_backbone: Whether to freeze the backbone initially
            **timm_kwargs: Additional arguments to pass to timm.create_model
        """
        super(TumorClassifier, self).__init__()
        
        # Initialize the UNI2-h model from HuggingFace using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            **timm_kwargs
        )
        
        # Get output features dimension from the model
        if hasattr(self.backbone, 'num_features'):
            hidden_size = self.backbone.num_features
        elif hasattr(self.backbone, 'embed_dim'):
            hidden_size = self.backbone.embed_dim
        else:
            hidden_size = 1536  # Default for UNI2-h
            
        print(f"Model feature dimension: {hidden_size}")
        
        # Freeze/unfreeze backbone parameters based on the freeze_backbone parameter
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Selectively unfreeze the specified number of transformer blocks from the end
            if unfreeze_layers > 0 and hasattr(self.backbone, 'blocks'):
                num_blocks = len(self.backbone.blocks)
                start_idx = max(0, num_blocks - unfreeze_layers)
                
                # Unfreeze only the last few blocks
                for i in range(start_idx, num_blocks):
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = True
                    print(f"Unfrozen block {i}")
            
            # Unfreeze all layers if unfreeze_layers is -1
            elif unfreeze_layers == -1:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                print("All backbone layers unfrozen")
            
            # Always unfreeze the positional embedding and patch embedding
            if hasattr(self.backbone, 'pos_embed'):
                self.backbone.pos_embed.requires_grad = True
                print("Positional embedding unfrozen")
            
            if hasattr(self.backbone, 'patch_embed'):
                self.backbone.patch_embed.requires_grad = True
                print("Patch embedding unfrozen")
        else:
            # If freeze_backbone is False, unfreeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("All backbone parameters are trainable")
        
        # # Add a classification head with strong regularization
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(hidden_size, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(256, num_classes)
        # )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            logits: Classification logits
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits.squeeze(-1)  # Remove last dimension for binary classification
        

# def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
#     """Train for one epoch"""
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []
    
#     progress_bar = tqdm(dataloader, desc="Training")
    
#     for batch in progress_bar:
#         # Get data
#         images = batch['image'].to(device)
#         labels = batch['label'].float().to(device)  # Use float for BCEWithLogitsLoss
        
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(images)  # Changed from model(pixel_values=images)
#         loss = criterion(outputs, labels)
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         # Update metrics
#         running_loss += loss.item()
        
#         # Get predictions
#         preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
#         all_preds.extend(preds)
#         all_labels.extend(labels.cpu().numpy())
        
#         # Update progress bar
#         progress_bar.set_postfix({"loss": loss.item()})
    
#     # Calculate metrics
#     epoch_loss = running_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, zero_division=0)
#     recall = recall_score(all_labels, all_preds, zero_division=0)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)
    
#     return epoch_loss, accuracy, precision, recall, f1

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    """Train for one epoch
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        scheduler: Learning rate scheduler
        device: Device to run training on (cuda/cpu)
        
    Returns:
        tuple: (epoch_loss, accuracy, precision, recall, f1)
    """
    # Set model to training mode
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Iterate through batches
    for batch in progress_bar:
        # Get data and move to device
        images = batch['image'].to(device)
        labels = batch['label'].float().to(device)  # Use float for BCEWithLogitsLoss
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        running_loss += loss.item()
        
        # Calculate predictions
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar with current loss
        progress_bar.set_postfix({"loss": loss.item()})
    
    # Calculate final metrics
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get data
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device)
            
            # Forward pass - Changed to pass images directly
            outputs = model(images)  # Changed from model(pixel_values=images)
            loss = criterion(outputs, labels)
            
            # Update metrics
            running_loss += loss.item()
            
            # Get predictions
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, accuracy, precision, recall, f1, cm

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance in binary classification
        
        Args:
            alpha: Weighting factor for the rare class (typically the positive class)
            gamma: Focusing parameter that reduces the loss contribution from easy examples
            reduction: 'mean' or 'sum' - how to reduce the loss over the batch
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            loss: Computed focal loss
        """
        # First compute standard BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Get probabilities
        pt = torch.exp(-bce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting for class imbalance
        if self.alpha is not None:
            # For positive samples: use alpha, for negative: use (1-alpha)
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_weight * focal_weight
            
        # Apply the weights to the BCE loss
        loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_model(model, train_loader, val_loader, num_epochs=10, 
                learning_rate=2e-5, weight_decay=0.01, save_dir='models'):
    """Train and evaluate model"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function - binary cross entropy with logits for binary classification
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_tumor_classifier.pth')
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    counter = 0 #early stopping
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        if counter == 100:
            break
        
        # Train
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_cm = validate(
            model, val_loader, criterion, device
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(val_cm)

        
        # Save the best model
        if (val_f1 > best_val_f1):
            best_val_f1 = val_f1
            # best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
            }, best_model_path)
            print(f"Saved best model with F1 score: {val_f1:.4f}, accuracy: {val_acc:.4f}")
        else:
            counter += 1
    
    # Plot training curves
    epochs = range(1, num_epochs + 1)
    
    # Create a 2x2 grid of plots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, 'b-', label='Training F1')
    plt.plot(epochs, val_f1s, 'r-', label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_precisions, 'b--', label='Training Precision')
    plt.plot(epochs, val_precisions, 'r--', label='Validation Precision')
    plt.plot(epochs, train_recalls, 'b-', label='Training Recall')
    plt.plot(epochs, val_recalls, 'r-', label='Validation Recall')
    plt.title('Training and Validation Precision/Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.show()
    
    # Save all metrics to CSV for further analysis
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accs,
        'val_accuracy': val_accs,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'train_precision': train_precisions,
        'val_precision': val_precisions,
        'train_recall': train_recalls,
        'val_recall': val_recalls
    })
    metrics_df.to_csv(os.path.join(save_dir, 'training_metrics.csv'), index=False)
    
    print(f"\nTraining complete. Best validation F1 score: {best_val_f1:.4f}")
    print(f"Training metrics saved to {os.path.join(save_dir, 'training_metrics.png')}")
    print(f"Detailed metrics data saved to {os.path.join(save_dir, 'training_metrics.csv')}")
    
    return best_model_path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def visualize_embeddings_tsne(model, dataloader, device, perplexity=30, n_iter=1000, 
                              max_samples=1000, save_path='tsne_visualization.png'):
    """
    Generate t-SNE visualization for model embeddings
    
    Args:
        model: The trained model
        dataloader: DataLoader to extract features from
        device: Device to run the model on
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        max_samples: Maximum number of samples to use for visualization
        save_path: Path to save the visualization
        
    Returns:
        None (saves visualization to file)
    """
    # Create modified forward hook to get embeddings
    embeddings = []
    labels = []
    image_ids = []
    
    # Create a modified model that only returns the features before classification
    class EmbeddingExtractor(nn.Module):
        def __init__(self, original_model):
            super(EmbeddingExtractor, self).__init__()
            self.backbone = original_model.backbone
            
        def forward(self, x):
            return self.backbone(x)
    
    # Create the embedding extractor
    embedding_model = EmbeddingExtractor(model).to(device)
    embedding_model.eval()
    
    # Collect embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader):
            # Get data
            images = batch['image'].to(device)
            batch_labels = batch['label'].cpu().numpy() if 'label' in batch else np.zeros(len(images))
            batch_image_ids = batch['image_id']
            
            # Extract features
            batch_embeddings = embedding_model(images).cpu().numpy()
            
            # Append to storage
            embeddings.append(batch_embeddings)
            labels.append(batch_labels)
            image_ids.extend(batch_image_ids)
            
            # Count samples
            sample_count += len(images)
            if sample_count >= max_samples:
                break
    
    # Concatenate all embeddings and labels
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    print(f"Collected {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Apply t-SNE dimensionality reduction
    print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for each class
    class_colors = ['blue', 'red']  # For binary classification: 0=normal, 1=tumor
    
    # Create plot
    for class_idx in np.unique(labels):
        # Get indices for this class
        class_indices = np.where(labels == class_idx)[0]
        
        # Plot this class
        plt.scatter(
            embeddings_2d[class_indices, 0],
            embeddings_2d[class_indices, 1],
            c=class_colors[int(class_idx)],
            label=f"Class {int(class_idx)}" if class_idx in [0, 1] else "Unknown",
            alpha=0.7,
            s=50
        )
    
    # Add labels and legend
    plt.title(f"t-SNE Visualization of Model Embeddings (perplexity={perplexity})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"t-SNE visualization saved to {save_path}")
    
    # Create additional visualizations with different perplexity values
    for p in [5, 15, 50]:
        if p == perplexity:
            continue
            
        print(f"Running t-SNE with alternate perplexity={p}...")
        tsne_alt = TSNE(n_components=2, perplexity=p, n_iter=n_iter, random_state=42)
        embeddings_2d_alt = tsne_alt.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        for class_idx in np.unique(labels):
            class_indices = np.where(labels == class_idx)[0]
            plt.scatter(
                embeddings_2d_alt[class_indices, 0],
                embeddings_2d_alt[class_indices, 1],
                c=class_colors[int(class_idx)],
                label=f"Class {int(class_idx)}" if class_idx in [0, 1] else "Unknown",
                alpha=0.7,
                s=50
            )
        
        plt.title(f"t-SNE Visualization of Model Embeddings (perplexity={p})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path_alt = save_path.replace('.png', f'_perp{p}.png')
        plt.tight_layout()
        plt.savefig(save_path_alt, dpi=300)
        plt.close()
        
        print(f"Alternative t-SNE visualization saved to {save_path_alt}")
    
    # Create a heatmap visualization to see if there are any patterns in feature space
    plt.figure(figsize=(12, 10))
    heatmap = plt.hist2d(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        bins=50,
        cmap='viridis'
    )
    plt.colorbar(heatmap[3], label="Sample Density")
    plt.title("t-SNE Embedding Density Heatmap")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    save_path_heatmap = save_path.replace('.png', '_heatmap.png')
    plt.tight_layout()
    plt.savefig(save_path_heatmap, dpi=300)
    plt.close()
    
    print(f"Heatmap visualization saved to {save_path_heatmap}")

# Define required parameters for UNI2-h
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
# Create model
model = TumorClassifier(model_name="hf_hub:MahmoodLab/UNI2-h", freeze_backbone=False, **timm_kwargs)
# model = TumorClassifier()

# Calculate and print statistics
stats = calculate_dataset_statistics(train_loader, val_loader)

# You can use these statistics to normalize your data consistently
train_mean, train_std, val_mean, val_std = stats['train_mean'], stats['train_std'], stats['val_mean'], stats['val_std']

print(f"Train mean: {train_mean}, Train std: {train_std}\n Val mean: {val_mean}, Val std: {val_std}")

# Train the model
best_model_path = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=30,  # Start with smaller number for testing
    learning_rate=1e-4,
    weight_decay=0.1,
    save_dir='tumor_models_backbone_train'
)

model.eval()
save_dir='tumor_models_uni'
device = DEVICE

# visualize_embeddings_tsne(
#     model=model,
#     dataloader=val_loader,  # Use validation set to see how well features separate
#     device=device,
#     perplexity=30,  # Try different values like 5, 15, 30, 50
#     save_path=os.path.join(save_dir, 'vit_embeddings_tsne.png')
# )