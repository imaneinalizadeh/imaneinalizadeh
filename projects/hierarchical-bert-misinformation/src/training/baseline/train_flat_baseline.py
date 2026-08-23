"""
Step 2: Train the Flat BERT Baseline (Experiment 1 comparison)
===============================================================
Trains an identical BERT encoder with a single linear head mapping
directly to 6 classes — no hierarchy.

This is used as the controlled comparison against the hierarchical model
in Experiment 1.  All hyperparameters are kept identical:
  LR=2e-5, batch=64, max_len=64, 2 epochs, AdamW + linear warmup.

Outputs:
  flat_best_model.pt  — saved to DATA_DIR (Google Drive)

Results (from paper):
  Flat macro F1:        0.75  (±0.006 across 3 seeds)
  Hierarchical macro F1:0.77  (±0.007)  → statistically significant (p < 0.05)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "/content/drive/MyDrive/fakeddit"
EPOCHS      = 2
BATCH_SIZE  = 64
MAX_LEN     = 64
LR          = 2e-5
NUM_CLASSES = 6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class FlatFakedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts     = df['clean_title'].fillna('').tolist()
        self.labels    = df['6_way_label'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class FlatBERTClassifier(nn.Module):
    """Standard BERT + single linear head → 6 classes."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        hidden          = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion  = nn.CrossEntropyLoss()
    for i, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if i % 500 == 0:
            print(f"  Batch {i}/{len(loader)} — Loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            preds          = model(input_ids, attention_mask).argmax(dim=1).tolist()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].tolist())

    print(classification_report(all_labels, all_preds,
          target_names=['True Content', 'Satire/Parody', 'Misleading',
                        'Imposter', 'False Connection', 'Manipulated']))
    return all_preds


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_df = pd.read_csv(f"{DATA_DIR}/train.tsv",       sep='\t')
    val_df   = pd.read_csv(f"{DATA_DIR}/validate.tsv",    sep='\t')
    test_df  = pd.read_csv(f"{DATA_DIR}/test_public.tsv", sep='\t')

    train_ds = FlatFakedditDataset(train_df, tokenizer)
    val_ds   = FlatFakedditDataset(val_df,   tokenizer)
    test_ds  = FlatFakedditDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, num_workers=2, pin_memory=True)

    model        = FlatBERTClassifier().to(DEVICE)
    optimizer    = AdamW(model.parameters(), lr=LR)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    criterion     = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        avg_loss = train(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Avg training loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['label'].to(DEVICE)
                logits         = model(input_ids, attention_mask)
                val_loss      += criterion(logits, labels).item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{DATA_DIR}/flat_best_model.pt")
            print("✓ Model saved")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(f"{DATA_DIR}/flat_best_model.pt", map_location=DEVICE))
    print("\nTest set evaluation:")
    evaluate(model, test_loader, DEVICE)
