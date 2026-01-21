import random
import requests
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import unittest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import TensorDataset


class CIFAR100Filtered(Dataset): #wrapper on the pytorch dataset
    """
    CIFAR-100 dataset wrapper with preprocessing and train/val split support.
    
    Args:
        root (str): Directory to store/load CIFAR-100 data
        split (str): Either "train" or "val" to specify which split to use
        transform (callable, optional): Transform to apply to images. If None, uses default.
    
    Attributes:
        dataset: The underlying torchvision CIFAR100 dataset
    """
    
    def __init__(self, root="./data", split="train", transform=None):
        # Validate that split is either "train" or "val"
        # Use assert to check this condition
        assert split in ("train", "val"), "split must be in train and validation"
        
        # If transform is None, create a default transform that:
        # 1. Resizes images to 224x224 (use transforms.Resize)
        # 2. Converts to tensor (use transforms.ToTensor)
        # 3. Normalizes with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Use transforms.Compose to chain these together
        if transform is None:
            '''
            transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ])
            '''
            transform = transforms.Compose([
                            transforms.Resize((224, 224)), # CIFAR100 has 32x32 img sizes, convert to 224x224 to be compatible MobileNet's input img size
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean = [0.485, 0.456, 0.406], #mean of each channel that was used to train MobileNet
                                std = [0.229, 0.224, 0.225]  #std of each channel that was used to train MobileNet
                            )
                        ])
            
        
        # Load CIFAR-100 using datasets.CIFAR100
        # Set train=True for split="train", train=False for split="val"
        # Remember to set download=True and pass the transform
        self.dataset = datasets.CIFAR100(root=root, train=(split == "train"), download=True, transform=transform)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        # Return the length of the underlying dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (image, label) where image is a transformed tensor and label is an integer
        """
        # Index into self.dataset and return the (image, label) tuple
        return self.dataset[idx]



class ImageEncoder(nn.Module):
    """
    MobileNetV3-based image encoder with trainable projection head.
    
    This model consists of two parts:
    1. Frozen pretrained MobileNetV3 backbone (feature extractor)
    2. Trainable projection head (maps features to embedding space)
    
    Args:
        proj_dim (int): Dimension of the output projection embeddings
        device (str): Device to place the model on ("cuda" or "cpu")
    
    Attributes:
        backbone: Frozen MobileNetV3 feature extractor (output: 576-dim)
        projection: Trainable MLP that projects to proj_dim
    """
    
    def __init__(self, proj_dim=64, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        
        # Load pretrained MobileNetV3-Small
        # Use models.mobilenet_v3_small with DEFAULT weights
        # Extract all layers except the final classifier using list(base.children())[:-1]
        # Wrap in nn.Sequential, move to device, and set to eval mode
        try:
            # torchvision >= 0.13 style weights enum
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            # Fallback: load without explicit enum (older torchvision)
            base = models.mobilenet_v3_small(pretrained=True)
        backbone = nn.Sequential(*list(base.children())[:-1]) #allows us to build a model as a plain stack of layers with each layer having one input tensor and one output tensor. #base.children() accesses all the base layers in the default MobileNetV3, we are excluding the last output layer, to include the projection head (that maps the features into an embedding space same as the SkipGram model's embedding space.
        self.backbone = backbone.to(self.device)
        self.backbone.eval() #removes dropout, batch normalization, gradients or any other techniques used for training, because we are using the frozen backbone model, not training it
        
        # Freeze the backbone parameters
        # Loop through self.backbone.parameters() and set requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False #we are NOT altering the existing graidents (trained gradients) of the backbone model (no training)
        # Create trainable projection head
        # Architecture: Linear(576 -> 512) -> BatchNorm1d(512) -> ReLU -> Linear(512 -> proj_dim)
        # Use nn.Sequential to chain the layers
        # Move to device using .to(device)
        dim_in = 576
        dim_h  = 512
        dim_out = proj_dim

        # self.projection = nn.Sequential(
            # nn.Linear(dim_in, dim_h), # layers used to map the output feature vectors from the backbone to a feature vector embedding that could live in the SkipGram's embedding space after completing Batch Normalization 
            # nn.BatchNorm1d(dim_h),
            # nn.ReLU(inplace=True),
            # nn.Linear(dim_h, proj_dim)
        # ).to(self.device)
        
        # self.projection = nn.Sequential(
            # nn.Linear(576, dim_h),
            # nn.BatchNorm1d(dim_h),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Linear(dim_h, proj_dim)
        # ).to(self.device)
        
        self.projection = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(dim_h, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(dim_h, dim_out),
        ).to(self.device)
            
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (backbone_features, projected_embeddings)
                - backbone_features: Raw features from MobileNet (batch_size, 576)
                - projected_embeddings: Projected embeddings (batch_size, proj_dim)
        """
        # Extract features using the frozen backbone
        # Use torch.no_grad() context to save memory
        # Flatten the output to shape (batch_size, 576) using .flatten(1)
        x = x.to(self.device)
        with torch.no_grad():
            features = self.backbone(x).flatten(1) #reshape the raw output feature vectors into a 1d vector 
        
        # Project features through the trainable projection head
        # Pass the flattened features through self.projection
        out = self.projection(features)
        
        # Return both the backbone features and projected embeddings as a tuple
        return features, out

def filter_dataset_indices(dataset, valid_labels):
    """Return indices of samples with labels in valid_labels set."""
    return [i for i, label in enumerate(dataset.dataset.targets) if label in valid_labels]

def create_data_splits(indices, val_ratio=0.2, seed=42):
    """Split indices into train/val sets."""
    np.random.seed(seed)
    indices = np.array(indices)
    np.random.shuffle(indices)
    split_idx = int((1 - val_ratio) * len(indices))
    return indices[:split_idx].tolist(), indices[split_idx:].tolist()

def create_dataloaders(train_idx, val_idx, test_idx, batch_sizes):
    """Create train, val, and test dataloaders."""
    datasets = {
        'train': Subset(CIFAR100Filtered(split="train"), train_idx),
        'val': Subset(CIFAR100Filtered(split="train"), val_idx), 
        'test': Subset(CIFAR100Filtered(split="val"), test_idx)
    }
    return {k: DataLoader(v, batch_size=batch_sizes['train' if k == 'train' else 'eval'], 
                         shuffle=(k == 'train'), num_workers=2) for k, v in datasets.items()}


def compute_contrastive_loss(visual_proj, text_emb, temperature):
    """
    Compute symmetric InfoNCE (contrastive) loss for vision-language alignment.
    
    This loss encourages matching pairs (image, correct_text) to have high similarity
    while pushing apart non-matching pairs (image, wrong_text).
    
    Args:
        visual_proj (torch.Tensor): Projected visual embeddings, shape (batch_size, proj_dim)
        text_emb (torch.Tensor): Text embeddings for the batch, shape (batch_size, proj_dim)
        temperature (float): Temperature parameter to scale logits (typically 0.07)
    
    Returns:
        torch.Tensor: Scalar loss value (symmetric InfoNCE loss)
    
    Mathematical formulation:
        1. Normalize both embeddings to unit vectors
        2. Compute similarity matrix: S = (visual @ text.T) / temperature
        3. Apply cross-entropy loss treating diagonal as correct matches
        4. Average image-to-text and text-to-image losses for symmetry
    """
    
    # Normalize visual_proj to unit vectors (L2 normalization)
    # Using F.normalize with p=2 and dim=1
    normalized_visual = F.normalize(visual_proj, p=2, dim=1) # Shape: (batch_size, proj_dim)
    
    # Normalize text_emb to unit vectors (L2 normalization)
    # Using F.normalize with p=2 and dim=1
    normalized_text = F.normalize(text_emb, p=2, dim=1) # Shape: (batch_size, proj_dim)

    
    # Compute similarity matrix (logits)
    # Matrix multiply: normalized_visual @ normalized_text.T
    # Divide by temperature to scale the logits
    # Using torch.matmul for matrix multiplication
    logits = torch.matmul(normalized_visual, normalized_text.T) / temperature #all the logits along the diagonal of our matrix are considred positive, and the rest are negatives
    
    # Create ground truth labels
    # For a batch of N samples, correct matches are on the diagonal
    # Labels should be [0, 1, 2, ..., N-1]
    # Using torch.arange to create labels on the same device as visual_proj
    device = visual_proj.device
    batch_size = visual_proj.shape[0]
    labels = torch.arange(batch_size, device=device)
    
    # Compute image-to-text loss
    # Treat each row as logits for which text matches this image
    # Using F.cross_entropy(logits, labels)
    i2t_loss = F.cross_entropy(logits, labels)
    
    # Compute text-to-image loss
    # Treat each column as logits for which image matches this text
    # Transpose the logits matrix and use F.cross_entropy(logits.T, labels)
    t2i_loss = F.cross_entropy(logits.T, labels)
    
    # Return symmetric loss
    # Average the two losses: (i2t_loss + t2i_loss) / 2
    loss = (i2t_loss + t2i_loss) / 2.0
    return loss



def run_epoch(model, dataloader, text_emb, class_words, label_to_word, optimizer, temperature, device, mode='train'):
    """
    Run one epoch of training or evaluation for contrastive vision-language learning.
    
    Handles a single pass over the dataset, computes loss and mean similarity,
    and updates the model if in training mode.
    
    Args:
        model (nn.Module): Image encoder model.
        dataloader (DataLoader): DataLoader providing (image, label) batches.
        text_emb (torch.Tensor): Embeddings for all class words, shape (num_classes, proj_dim).
        class_words (list): List of word strings for each class label index.
        label_to_word (dict): Maps integer label to the corresponding word string.
        optimizer (torch.optim.Optimizer or None): Optimizer for model parameters (set to None in eval mode).
        temperature (float): Contrastive loss temperature parameter.
        device (str or torch.device): Device for computation.
        mode (str): 'train' or 'eval' (evaluation).
    
    Returns:
        tuple: (mean_loss, mean_similarity)
            mean_loss: Average loss over the epoch.
            mean_similarity: Average cosine similarity between visual and aligned text embeddings.
    """

    if mode == 'train':
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    total_loss = 0.0
    total_sim = 0.0
    total_examples = 0

    device = torch.device(device)

    # ensure text_emb_tensor exists
    if isinstance(text_emb, np.ndarray):
        text_emb_tensor = torch.from_numpy(text_emb)
    else:
        text_emb_tensor = text_emb
    text_emb_tensor = text_emb_tensor.to(device)

    with grad_context:
        for images, labels in tqdm(dataloader, desc=f"{mode} epoch"):

            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # forward
            features, visual_proj = model(images)

            # map labels → class words → indices
            batch_text_idx = []
            for l in labels:
                word = label_to_word[int(l)]
                idx = class_words.index(word)
                batch_text_idx.append(idx)

            batch_text_idx = torch.tensor(batch_text_idx,
                                          dtype=torch.long,
                                          device=device)

            batch_text_emb = text_emb_tensor[batch_text_idx]

            # contrastive loss
            loss = compute_contrastive_loss(visual_proj,
                                            batch_text_emb,
                                            temperature)

            if mode == 'train' and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size

            # ---- fix: proj → visual_proj ----
            norm_proj = F.normalize(visual_proj, p=2, dim=1)
            norm_text = F.normalize(batch_text_emb, p=2, dim=1)

            sims = (norm_proj * norm_text).sum(dim=1)

            total_sim += sims.sum().item()
            total_examples += batch_size

    mean_loss = total_loss / total_examples
    mean_similarity = total_sim / total_examples

    return mean_loss, mean_similarity


def train_with_early_stopping(model, dataloaders, text_emb, class_words, label_to_word, config, device):
    """
    Train model with early stopping based on validation similarity.
    
    Trains the model for multiple epochs, monitoring validation performance and stopping
    early if no improvement is seen for a specified number of epochs (patience).
    
    Args:
        model (nn.Module): Image encoder model to train.
        dataloaders (dict): Dictionary with keys 'train' and 'val', each containing a DataLoader.
        text_emb (torch.Tensor): Text embeddings for all classes, shape (num_classes, proj_dim).
        class_words (list): List of class word strings.
        label_to_word (dict): Maps integer labels to word strings.
        config (dict): Training configuration containing:
            - 'lr': Learning rate
            - 'weight_decay': Weight decay for optimizer
            - 'epochs': Maximum number of epochs
            - 'temperature': Temperature for contrastive loss
            - 'patience': Early stopping patience (epochs without improvement)
            - 'save_path': Path to save best model checkpoint
        device (str or torch.device): Device for computation.
    
    Returns:
        tuple: (history, best_epoch, best_val_sim, best_val_loss)
            history: Dictionary tracking 'train_loss', 'val_loss', 'val_similarity', 'learning_rate'
            best_epoch: Epoch number with best validation similarity
            best_val_sim: Best validation similarity achieved
            best_val_loss: Validation loss at best epoch
    """
    
    # Create optimizer for trainable parameters (model.projection.parameters())
    # Use torch.optim.AdamW with lr and weight_decay from config
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=config.get('lr', 1e-3),
                                  weight_decay=config.get('weight_decay', 0.0))

    # Create learning rate scheduler
    # Use torch.optim.lr_scheduler.CosineAnnealingLR with T_max=config['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.get('epochs', 1)))

    # Initialize tracking variables
    best_val_sim = -float('inf') # we want to maximize similarity
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    history = defaultdict(list) #to track metrics

    # Print training header
    print(f"\n{'='*70}\nTraining (max {config['epochs']} epochs, patience={config['patience']})\n{'='*70}")
    
    # Training loop
    try:
        for epoch in range(1, config['epochs'] + 1):
            train_loss, train_sim = run_epoch(model, dataloaders['train'], text_emb, class_words, label_to_word,
                                              optimizer, config['temperature'], device, mode='train')
            val_loss, val_sim = run_epoch(model, dataloaders['val'], text_emb, class_words, label_to_word,
                                          None, config['temperature'], device, mode='eval')

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_similarity'].append(val_sim)
            history['learning_rate'].append(current_lr)

            print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f}, Train sim: {train_sim:.4f} | Val loss: {val_loss:.4f}, Val sim: {val_sim:.4f} | LR: {current_lr:.6f}")

            # If val_sim > best_val_sim: save checkpoint and reset patience
            if val_sim > best_val_sim:
                best_val_sim = val_sim
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save checkpoint using torch.save
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_similarity': val_sim,
                    'class_words': class_words,
                    'text_embeddings': text_emb.cpu() if isinstance(text_emb, torch.Tensor) else text_emb,
                    'history': dict(history),
                    'projection_state': model.projection.state_dict()
                }
                save_path = config.get('save_path', 'best_model.pth')
                try:
                    torch.save(ckpt, save_path)
                    print(f"  ✓ New best model saved to {save_path} (epoch {epoch}, val_sim={val_sim:.4f})")
                except Exception as e:
                    print(f"  ✗ Failed to save checkpoint: {e}")

            else:
                patience_counter += 1
                print(f"  - No improvement (patience {patience_counter}/{config['patience']})")

            # Early stopping
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping after epoch {epoch} (no improvement for {config['patience']} epochs).")
                break

    except Exception as e:
        print(f"\nTraining crashed with error: {str(e)}")
        print(f"Attempting to continue with best saved model...")

    # Return history (as dict), best_epoch, best_val_sim, best_val_loss
    return dict(history), best_epoch, best_val_sim, best_val_loss

def collect_embeddings(model, dataloader, device):
    """Collect all embeddings and labels from dataset."""
    model.eval()
    all_visual, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
            images = images.to(device)
            _, visual_proj = model(images)
            all_visual.append(F.normalize(visual_proj, p=2, dim=1).cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_visual, dim=0).numpy(), all_labels

def compute_alignment_metrics(visual_emb, labels, text_emb, class_words, label_to_word):
    """Compute comprehensive alignment metrics in one pass."""
    # Per-class statistics
    class_sims = defaultdict(list)
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            sim = np.dot(visual_emb[i], text_emb[class_words.index(word)])
            class_sims[word].append(sim)
    
    stats = sorted([{
        'word': word, 'mean': np.mean(sims), 'std': np.std(sims), 
        'min': np.min(sims), 'max': np.max(sims), 'count': len(sims)
    } for word, sims in class_sims.items()], key=lambda x: x['mean'], reverse=True)
    
    # Retrieval metrics
    sim_matrix = cosine_similarity(visual_emb, text_emb)
    i2t_recalls = {k: 0 for k in [1, 5, 10]}
    t2i_recalls = {k: 0 for k in [1, 5, 10]}
    
    # Image-to-text retrieval
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            correct_idx = class_words.index(word)
            ranking = np.argsort(-sim_matrix[i])
            for k in i2t_recalls:
                if correct_idx in ranking[:k]: i2t_recalls[k] += 1
    
    # Text-to-image retrieval  
    for class_idx, word in enumerate(class_words):
        class_img_idx = [i for i, l in enumerate(labels) if label_to_word[l] == word]
        if class_img_idx:
            ranking = np.argsort(-sim_matrix[:, class_idx])
            for k in t2i_recalls:
                if any(idx in ranking[:k] for idx in class_img_idx): t2i_recalls[k] += 1
    
    return stats, i2t_recalls, t2i_recalls, sim_matrix

def print_analysis_results(stats, i2t_recalls, t2i_recalls, n_samples, n_classes):
    """Print comprehensive analysis results."""
    print("\nPer-Class Similarity Analysis:")
    print("-" * 70)
    for title, data in [("Top 10 Best Aligned Classes:", stats[:10]), 
                        ("Bottom 10 Worst Aligned Classes:", stats[-10:])]:
        print(f"\n{title}")
        for i, s in enumerate(data, 1):
            print(f"{i:2d}. {s['word']:15s} | Mean: {s['mean']:.4f} ± {s['std']:.4f}")
    
    print("\nRetrieval Performance:")
    print("-" * 70)
    for name, recalls, total in [("Image-to-Text", i2t_recalls, n_samples), 
                                 ("Text-to-Image", t2i_recalls, n_classes)]:
        print(f"\n{name} Retrieval (Recall@K):")
        for k, count in recalls.items():
            print(f"  Recall@{k:2d}: {count/total*100:.2f}% ({count}/{total})")

def print_example_retrievals(sim_matrix, labels, class_words, label_to_word, n_examples=5):
    """Print text-based retrieval examples."""
    print("\nExample Image-to-Text Retrievals:")
    print("-" * 70)
    
    display_idx = np.random.choice(len(labels), size=n_examples, replace=False)
    
    for idx in display_idx:
        label = labels[idx]
        true_word = label_to_word[label]
        
        sims = sim_matrix[idx]
        top_5_idx = np.argsort(-sims)[:5]
        top_5_words = [class_words[i] for i in top_5_idx]
        top_5_sims = [sims[i] for i in top_5_idx]
        
        correct_sim = sims[class_words.index(true_word)]
        correct_rank = np.where(np.argsort(-sims) == class_words.index(true_word))[0][0] + 1
        
        print(f"\nTest Image #{idx}:")
        print(f"  True class: '{true_word}' (similarity: {correct_sim:.4f}, rank: {correct_rank})")
        print(f"  Top 5 predictions:")
        for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
            marker = "✓" if word == true_word else " "
            print(f"    {rank}. {marker} {word:15s} (similarity: {sim:.4f})")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_visualizations(sim_matrix, labels, class_words, label_to_word, test_indices, images=None, names=None, predictions=None):
    """Create all visualizations in one coordinated function."""
    # OOD analysis if provided
    if images and names and predictions:
        print(f"\nCreating OOD visualization for {len(images)} images...")
        n_imgs = len(images)
        n_cols = min(4, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        for i, (img, name, pred) in enumerate(zip(images, names, predictions)):
            axes[i].imshow(img); axes[i].axis('off')
            pred_text = f"{name.upper()}\n\nTop matches:\n"
            for rank, (word, sim) in enumerate(zip(pred['words'][:5], pred['sims'][:5]), 1):
                pred_text += f"{rank}. {word} ({sim:.3f})\n"
            axes[i].set_title(pred_text, fontsize=11, ha='center', color='darkblue', fontweight='bold', pad=12)
        
        for j in range(i+1, len(axes)): 
            axes[j].axis('off'); axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('ood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("\nCreating confusion matrix...")
        n_classes = len(class_words)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i, label in enumerate(labels):
            if (word := label_to_word[label]) in class_words:
                true_idx = class_words.index(word)
                pred_idx = np.argmax(sim_matrix[i])
                conf_matrix[true_idx, pred_idx] += 1
        
        conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, xticklabels=class_words, yticklabels=class_words,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Probability'}, square=True)
        ax.set_xlabel('Predicted Class'); ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Retrieval examples
        print("\nCreating retrieval examples...")
        test_raw = CIFAR100Filtered(split="val", transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for plot_idx, ax in enumerate(axes.flatten()):
            if plot_idx >= 12: break
            ex_idx = random.randint(0, len(labels)-1)
            original_idx = test_indices[ex_idx]
            img, label = test_raw[original_idx]
            
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            
            true_word = label_to_word[label]
            sims = sim_matrix[ex_idx]
            top_5_idx = np.argsort(-sims)[:5]
            top_5_words = [class_words[i] for i in top_5_idx]
            top_5_sims = [sims[i] for i in top_5_idx]

            # Build prediction text
            pred_text = f"GT: {true_word}\n"
            for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
                marker = "✓" if word == true_word else "✗"
                pred_text += f"{rank}. {marker} {word}: {sim:.2f}\n"

            # --- COLOR LOGIC CHANGE ---
            if top_5_words[0] == true_word:
                title_color = "green"          # correct top-1
            elif true_word in top_5_words:
                title_color = "#CC8A00"        # amber
            else:
                title_color = "red"            # incorrect
            # --------------------------

            ax.set_title(
                pred_text, fontsize=9, ha='left',
                fontfamily='monospace',
                color=title_color, fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig('retrieval_examples.png', dpi=300, bbox_inches='tight')
        plt.show()

def process_ood_images(model, image_urls, text_emb, class_words, device):
    """Download and process OOD images in one function."""
    print(f"\nDownloading {len(image_urls)} OOD test images...")
    images, names, headers = [], [], {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'}
    
    for desc, url in image_urls.items():
        try:
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                images.append(img.resize((224, 224), Image.BILINEAR))
                names.append(desc)
                print(f"  ✓ Downloaded: {desc}")
        except Exception as e:
            print(f"  ✗ Error downloading {desc}: {str(e)[:50]}")
    
    if not images: return [], [], []
    
    print(f"\nProcessing {len(images)} OOD images...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    model.eval()
    with torch.no_grad():
        ood_emb = []
        for img in images:
            img_tensor = normalize(to_tensor(img).unsqueeze(0).to(device))
            _, visual_proj = model(img_tensor)
            ood_emb.append(F.normalize(visual_proj, p=2, dim=1).cpu().numpy()[0])
    
    ood_emb = np.array(ood_emb)
    predictions = []
    for emb in ood_emb:
        sims = cosine_similarity(emb.reshape(1, -1), text_emb)[0]
        top_5_idx = np.argsort(-sims)[:5]
        predictions.append({
            'words': [class_words[j] for j in top_5_idx],
            'sims': [sims[j] for j in top_5_idx]
        })
    
    return images, names, predictions

def print_final_report(config, test_loss, test_sim, i2t_recalls, t2i_recalls, class_stats,
                      n_train, n_val, n_test, n_classes, batch_size, has_ood, history=None, 
                      best_epoch=None, best_val_sim=None, best_val_loss=None, n_vocab_total=None):
    """Print comprehensive final summary report."""
    n_samples = n_test
    
    print(f"""
Training Configuration:
   ├─ Model: MobileNetV3-Small with projection head
   ├─ Embedding dimension: {config['proj_dim']}
   ├─ Training samples: {n_train:,}, Validation samples: {n_val:,}, Test samples: {n_test:,}
   ├─ Number of training classes: {n_classes}
   {f'├─ Total vocabulary size: {n_vocab_total} words' if n_vocab_total else ''}
   {f'├─ Total epochs trained: {len(history["train_loss"]) if history else 0}, Best epoch: {best_epoch if best_epoch else "N/A"}' if history else ''}
   └─ Early stopping patience: {config['patience']} 

Performance Metrics:
   {f"├─ Best Val Similarity: {best_val_sim:.4f}" if best_val_sim else ""}
   ├─ Test Similarity: {test_sim:.4f}, Test Loss: {test_loss:.4f}
   ├─ Random baseline loss: ~{np.log(batch_size):.2f}
   │
   ├─ Image→Text Recall@1: {i2t_recalls[1]/n_samples*100:.2f}%
   ├─ Image→Text Recall@5: {i2t_recalls[5]/n_samples*100:.2f}%
   ├─ Image→Text Recall@10: {i2t_recalls[10]/n_samples*100:.2f}%
   │
   ├─ Text→Image Recall@1: {t2i_recalls[1]/n_classes*100:.2f}%
   ├─ Text→Image Recall@5: {t2i_recalls[5]/n_classes*100:.2f}%
   └─ Text→Image Recall@10: {t2i_recalls[10]/n_classes*100:.2f}%

Embedding Space Alignment:
   ├─ Mean per-class similarity: {np.mean([s['mean'] for s in class_stats]):.4f} ± {np.std([s['mean'] for s in class_stats]):.4f}
   ├─ Best aligned class: '{class_stats[0]['word']}' ({class_stats[0]['mean']:.4f})
   └─ Worst aligned class: '{class_stats[-1]['word']}' ({class_stats[-1]['mean']:.4f})

Key Insights:
   • The model {'successfully learns' if test_sim > 0.5 else 'attempts to learn'} visual-text alignment
   • {'High' if test_sim > 0.7 else 'Moderate' if test_sim > 0.5 else 'Low'} overall alignment (similarity: {test_sim:.4f})
   • Loss: {test_loss:.2f} vs random baseline ~{np.log(batch_size):.2f}
   • Retrieval performance: {'Good' if i2t_recalls[1]/n_samples > 0.5 else 'Moderate'}
   • Class performance varies (range: {class_stats[-1]['mean']:.4f} to {class_stats[0]['mean']:.4f})
   {f'• OOD predictions use full vocabulary of {n_vocab_total} words' if n_vocab_total else ''}

Model saved to: '{config['save_path']}'
Confusion matrix & retrieval examples saved
{' OOD analysis saved' if has_ood else ''}
""")

class TestCIFAR100Filtered(unittest.TestCase):

    def test_custom_transform_applied(self):
        """Custom transforms should be applied correctly."""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        ds = CIFAR100Filtered(split="train", transform=transform)
        img, _ = ds[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape[1:], (128, 128))

    def test_getitem_returns_tuple(self):
        """__getitem__ should return (image, label)."""
        ds = CIFAR100Filtered(split="train")
        item = ds[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)

    def test_invalid_split_raises_error(self):
        """Invalid split must raise AssertionError."""
        with self.assertRaises(AssertionError):
            CIFAR100Filtered(split="invalid")

    def test_dataset_length(self):
        """Dataset should have non-zero length."""
        ds = CIFAR100Filtered(split="train")
        self.assertTrue(len(ds) > 0)

    def test_train_split_creation(self):
        """Train split loads correctly."""
        ds = CIFAR100Filtered(split="train")
        self.assertTrue(len(ds) > 0)

    def test_val_split_uses_test_set(self):
        """Val split uses train=False (torchvision test set)."""
        ds = CIFAR100Filtered(split="val")
        self.assertTrue(len(ds) > 0)

    def test_getitem_type(self):
        """Default dataset returns tensor image + int label."""
        ds = CIFAR100Filtered(split="train")
        img, label = ds[0]
        self.assertIsInstance(img, torch.Tensor)   # CHANGED from PIL.Image
        self.assertIsInstance(label, int)

    def test_transform_output_tensor(self):
        """ToTensor transform converts image to torch.Tensor."""
        transform = transforms.Compose([transforms.ToTensor()])
        ds = CIFAR100Filtered(split="train", transform=transform)
        img, _ = ds[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape[0], 3)  # RGB channels

    def test_train_val_split_sizes(self):
        """
        Train/val filtered datasets should match the underlying
        torchvision train/test sizes (no 'test' split in wrapper).
        """
        full_train = datasets.CIFAR100(root="./data", train=True, download=True)
        full_test  = datasets.CIFAR100(root="./data", train=False, download=True)

        ds_train = CIFAR100Filtered(split="train")
        ds_val   = CIFAR100Filtered(split="val")

        self.assertEqual(len(ds_train), len(full_train))
        self.assertEqual(len(ds_val), len(full_test))

    def test_label_range(self):
        """Labels must be in [0, 99]."""
        ds = CIFAR100Filtered(split="train")
        _, label = ds[0]
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 100)

    def test_multiple_transforms(self):
        """Resize + ToTensor + Normalize should work."""
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = CIFAR100Filtered(split="train", transform=transform)
        img, _ = ds[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape[1:], (64, 64))


# -------------------------------
# ImageEncoder Tests 
# -------------------------------
class TestImageEncoder(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.model = ImageEncoder(proj_dim=64, device=self.device)

    def test_model_initialization(self):
        """Model should initialize without errors."""
        self.assertIsInstance(self.model, nn.Module)

    def test_forward_output_shapes(self):
        """Forward pass returns correct shapes."""
        x = torch.randn(2, 3, 224, 224)
        features, proj = self.model(x)
        self.assertEqual(features.shape, (2, 576))
        self.assertEqual(proj.shape[0], 2)
        self.assertEqual(proj.shape[1], self.model.projection[-1].out_features)

    def test_backbone_frozen(self):
        """Backbone parameters must be frozen."""
        for p in self.model.backbone.parameters():
            self.assertFalse(p.requires_grad)

    def test_projection_trainable(self):
        """Projection head parameters must be trainable."""
        for p in self.model.projection.parameters():
            self.assertTrue(p.requires_grad)

    def test_no_nan_outputs(self):
        """Forward pass should not produce NaNs."""
        x = torch.randn(2, 3, 224, 224)
        features, proj = self.model(x)
        self.assertFalse(torch.isnan(features).any())
        self.assertFalse(torch.isnan(proj).any())

    def test_different_projection_dims(self):
        """Different proj_dim values should work."""
        for dim in [32, 128, 256]:
            model = ImageEncoder(proj_dim=dim, device=self.device)
            x = torch.randn(2, 3, 224, 224)
            _, proj = model(x)
            self.assertEqual(proj.shape[1], dim)

    def test_forward_gradient_control(self):
        """Backbone should not receive gradients."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        _, proj = self.model(x)
        loss = proj.sum()
        loss.backward()
        for p in self.model.backbone.parameters():
            self.assertIsNone(p.grad)

    def test_projection_head_structure(self):
        """
        Projection head should start and end with Linear, and contain
        at least one BatchNorm + ReLU (compatible with multi-layer head).
        """
        layers = list(self.model.projection.children())
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertIsInstance(layers[-1], nn.Linear)
        self.assertTrue(any(isinstance(l, nn.BatchNorm1d) for l in layers))
        self.assertTrue(any(isinstance(l, nn.ReLU) for l in layers))

    def test_forward_multiple_batch_sizes(self):
        """Forward pass should work for different batch sizes."""
        self.model.eval()   # avoid BatchNorm train-mode issue for batch_size=1
        with torch.no_grad():
            for batch_size in [1, 5, 10]:
                x = torch.randn(batch_size, 3, 224, 224)
                features, proj = self.model(x)
                self.assertEqual(features.shape[0], batch_size)
                self.assertEqual(proj.shape[0], batch_size)

    def test_projection_head_output_dim(self):
        """Projection output dim must match proj_dim."""
        x = torch.randn(2, 3, 224, 224)
        _, proj = self.model(x)
        self.assertEqual(proj.shape[1], self.model.projection[-1].out_features)

    def test_backbone_requires_grad_false(self):
        """Backbone stays frozen over multiple passes."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        for _ in range(3):
            _, proj = self.model(x)
            loss = proj.sum()
            loss.backward()
        for p in self.model.backbone.parameters():
            self.assertIsNone(p.grad)


# -------------------------------
# ComputeContrastiveLoss Tests
# -------------------------------
class TestComputeContrastiveLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.proj_dim = 16
        # unnormalized; function will normalize internally
        self.visual = torch.randn(self.batch_size, self.proj_dim)
        self.text   = torch.randn(self.batch_size, self.proj_dim)
        self.temp   = 0.07

    def test_loss_computation(self):
        """Basic loss computation returns positive scalar."""
        loss = compute_contrastive_loss(self.visual, self.text, self.temp)
        self.assertTrue(loss.item() > 0)

    def test_loss_symmetric(self):
        """Loss should be symmetric between modalities."""
        loss1 = compute_contrastive_loss(self.visual, self.text, self.temp)
        loss2 = compute_contrastive_loss(self.text, self.visual, self.temp)
        self.assertAlmostEqual(loss1.item(), loss2.item(), delta=1e-5)

    def test_no_nan_or_inf(self):
        """Loss should never be NaN or Inf."""
        loss = compute_contrastive_loss(self.visual, self.text, self.temp)
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss))

    def test_gradient_flow(self):
        """Gradients should flow through both inputs."""
        visual = self.visual.clone().requires_grad_(True)
        text   = self.text.clone().requires_grad_(True)
        loss = compute_contrastive_loss(visual, text, self.temp)
        loss.backward()
        self.assertIsNotNone(visual.grad)
        self.assertIsNotNone(text.grad)

    def test_batch_size_minimum(self):
        """Works with minimum batch size 2."""
        visual = torch.randn(2, self.proj_dim)
        text   = torch.randn(2, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, self.temp)
        self.assertTrue(loss.item() > 0)

    def test_normalization_robustness(self):
        """Unnormalized inputs are handled internally."""
        visual = torch.randn(self.batch_size, self.proj_dim)
        text   = torch.randn(self.batch_size, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, self.temp)
        self.assertTrue(loss.item() > 0)

    def test_identical_inputs_loss(self):
        """Identical embeddings still give finite positive loss."""
        visual = torch.randn(4, self.proj_dim)
        text   = visual.clone()
        loss = compute_contrastive_loss(visual, text, self.temp)
        self.assertTrue(loss.item() > 0)

    def test_zero_vector_input(self):
        """Zero vectors must not produce NaN / Inf."""
        visual = torch.zeros(4, self.proj_dim)
        text   = torch.zeros(4, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, self.temp)
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss))

    def test_high_temperature(self):
        """Very high temperature still yields valid loss."""
        visual = torch.randn(4, self.proj_dim)
        text   = torch.randn(4, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, temperature=100.0)
        self.assertTrue(loss.item() > 0)

    def test_low_temperature(self):
        """Very low temperature still yields valid loss."""
        visual = torch.randn(4, self.proj_dim)
        text   = torch.randn(4, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, temperature=1e-6)
        self.assertTrue(loss.item() > 0)

    def test_batch_size_edge_case(self):
        """Another batch-size=2 check."""
        visual = torch.randn(2, self.proj_dim)
        text   = torch.randn(2, self.proj_dim)
        loss = compute_contrastive_loss(visual, text, self.temp)
        self.assertTrue(loss.item() > 0)


def run_tests():
    print("="*70)
    print("RUNNING UPDATED LAB 8 UNIT TESTS")
    print("="*70)
    suite = unittest.TestLoader().loadTestsFromModule(__import__("cifar100_contrastive_alignment"))
    unittest.TextTestRunner(verbosity=2).run(suite)