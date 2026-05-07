import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import List

from deap_pipeline import DEAPPipeline
from transformer_model import MultimodalTransformerEncoder

NUM_CLASSES = 5
CLASS_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]

class DEAPDataset(Dataset):
    def __init__(self, data_dir: str, subjects: List[str], pipeline: DEAPPipeline):
        super().__init__()
        self.X_eeg = []
        self.X_ecg = []
        self.y = []
        
        print(f"Loading {len(subjects)} subjects...")
        for subj in subjects:
            file_path = os.path.join(data_dir, f"{subj}.dat")
            if not os.path.exists(file_path):
                print(f"  [Warning] Skipping {subj}, file not found at {file_path}")
                continue
                
            data, labels = pipeline.load_participant_data(file_path)
            # Use 5-class labels
            stress_labels = pipeline.extract_stress_labels_5class(labels)
            
            X_windows, y_windows, hrv_windows, feature_windows = pipeline.create_segments(data, stress_labels)
            
            if len(X_windows) > 0:
                self.X_eeg.append(feature_windows[:, :97])
                self.X_ecg.append(feature_windows[:, 97:])
                self.y.append(y_windows)

        if len(self.X_eeg) > 0:
            self.X_eeg = np.concatenate(self.X_eeg, axis=0)
            self.X_ecg = np.concatenate(self.X_ecg, axis=0)
            self.y    = np.concatenate(self.y, axis=0)
            # Print class distribution for transparency
            unique, counts = np.unique(self.y, return_counts=True)
            print(f"Loaded {len(self.y)} windows successfully.")
            print("Class distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt} samples ({100*cnt/len(self.y):.1f}%)")
        else:
            self.X_eeg = np.array([])
            self.X_ecg = np.array([])
            self.y     = np.array([])
            print("No valid data loaded.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_eeg[idx], dtype=torch.float32),
            torch.tensor(self.X_ecg[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)   # CrossEntropyLoss needs long
        )


def train_model():
    data_dir = r"c:\Users\sambh\Downloads\archive (9)\deap-dataset\data_preprocessed_python"
    
    # Train: s01-s24, Test: s25-s32
    train_subjects = [f"s{i:02d}" for i in range(1, 25)]
    test_subjects  = [f"s{i:02d}" for i in range(25, 33)]
    
    pipeline = DEAPPipeline(
        data_dir=data_dir,
        window_size_sec=30.0,
        overlap_sec=15.0,
        remove_baseline=True,
        extract_eeg_features=True,
        extract_hrv=True
    )
    
    print("Preparing Training Dataset...")
    train_dataset = DEAPDataset(data_dir, train_subjects, pipeline)
    print("\nPreparing Testing Dataset...")
    test_dataset  = DEAPDataset(data_dir, test_subjects, pipeline)
    
    # Mock data fallback (for dev/testing without DEAP files)
    if len(train_dataset) == 0:
        print("\n[MOCK MODE] No DEAP data found. Using dummy data...")
        train_dataset.X_eeg = np.random.randn(200, 97)
        train_dataset.X_ecg = np.random.randn(200, 4)
        train_dataset.y     = np.random.randint(0, NUM_CLASSES, 200)
        
        test_dataset.X_eeg = np.random.randn(40, 97)
        test_dataset.X_ecg = np.random.randn(40, 4)
        test_dataset.y     = np.random.randint(0, NUM_CLASSES, 40)

    batch_size   = 32
    # Use class-balanced sampling for the extremely imbalanced 5-class DEAP labels
    class_counts = np.bincount(train_dataset.y, minlength=NUM_CLASSES)
    class_counts[class_counts == 0] = 1
    class_weights_np = len(train_dataset) / (NUM_CLASSES * class_counts.astype(np.float32))
    class_weights = torch.from_numpy(class_weights_np).float()
    sample_weights = class_weights_np[train_dataset.y]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Training class counts: {dict(enumerate(class_counts.tolist()))}")
    print(f"Training class weights: {[float(w) for w in class_weights.tolist()]}")

    # 5-class model — use_logits=True, CrossEntropyLoss handles softmax internally
    model = MultimodalTransformerEncoder(
        eeg_dim=97, ecg_dim=4, d_model=128, n_heads=4, n_layers=3,
        ff_dim=256, dropout=0.3, num_classes=NUM_CLASSES, use_logits=True
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    num_epochs   = 50
    best_val_acc = 0.0
    
    print("\nStarting Training (5-Class)...\n" + "="*60)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for eeg_b, ecg_b, y_b in train_loader:
            eeg_b, ecg_b, y_b = eeg_b.to(device), ecg_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(eeg_b, ecg_b)          # (batch, 5)
            loss   = criterion(logits, y_b)        # CrossEntropyLoss wants (batch,5) vs (batch,)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * eeg_b.size(0)
        
        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for eeg_b, ecg_b, y_b in test_loader:
                eeg_b, ecg_b, y_b = eeg_b.to(device), ecg_b.to(device), y_b.to(device)
                logits  = model(eeg_b, ecg_b)
                loss    = criterion(logits, y_b)
                val_loss += loss.item() * eeg_b.size(0)
                
                probs  = torch.softmax(logits, dim=1)   # (batch, 5)
                preds  = torch.argmax(probs, dim=1)     # (batch,)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        val_loss /= len(test_dataset)
        all_labels = np.array(all_labels)
        all_preds  = np.array(all_preds)
        all_probs  = np.array(all_probs)

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # ROC-AUC (one-vs-rest, macro) only if all classes are present
        try:
            val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            val_auc = float('nan')

        print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1(macro): {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Backup old model if it exists
            if os.path.exists('best_model.pth') and not os.path.exists('best_model_binary.pth'):
                shutil.copy('best_model.pth', 'best_model_binary.pth')
                print("  --> Backed up binary model to best_model_binary.pth")
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  --> Saved better model (Val Acc: {best_val_acc:.4f})")

    print("="*60 + f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")
    print("Model saved to best_model.pth")

if __name__ == "__main__":
    train_model()
