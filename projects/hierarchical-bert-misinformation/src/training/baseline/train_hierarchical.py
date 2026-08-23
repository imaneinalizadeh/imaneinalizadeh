"""
Step 1: Train the Hierarchical BERT Fake News Classifier
=========================================================
Trains a two-stage hierarchical model on the Fakeddit dataset.

Architecture:
  - Shared BERT encoder (bert-base-uncased)
  - Coarse head: 3-way group classifier (Authentic / Structural Deception / Fabricated)
  - Fine heads: per-group classifiers
      Group 0 (Authentic):             1 output  → True Content
      Group 1 (Structural Deception):  2 outputs → Misleading Content, False Connection
      Group 2 (Fabricated/Manip.):     3 outputs → Satire/Parody, Imposter, Manipulated

Label mapping (6-way fine labels):
  0 = True Content       → Group 0
  1 = Satire/Parody      → Group 2
  2 = Misleading Content → Group 1
  3 = Imposter Content   → Group 2
  4 = False Connection   → Group 1
  5 = Manipulated Content→ Group 2

Outputs:
  best_model.pt  — saved to DATA_DIR (Google Drive)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

# ── Label maps ────────────────────────────────────────────────────────────────
FINE_TO_COARSE   = {0: 0, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2}
COARSE_LABELS    = {0: 'Authentic', 1: 'Structural Deception', 2: 'Fabricated/Manipulated'}
LOCAL_TO_FINE    = {(0, 0): 0, (1, 0): 2, (1, 1): 4, (2, 0): 1, (2, 1): 3, (2, 2): 5}
FINE_LOCAL_INDEX = {0: {0: 0}, 1: {2: 0, 4: 1}, 2: {1: 0, 3: 1, 5: 2}}

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/fakeddit"
EPOCHS     = 2          # paper trains for 2 epochs (loss converges)
BATCH_SIZE = 64
MAX_LEN    = 64         # Fakeddit titles are short; 64 tokens wastes no compute
LR         = 2e-5       # standard BERT fine-tuning rate


# ── Dataset ───────────────────────────────────────────────────────────────────
class FakedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts        = df['clean_title'].fillna('').tolist()
        self.fine_labels  = df['6_way_label'].tolist()
        self.coarse_labels = [FINE_TO_COARSE[l] for l in self.fine_labels]
        self.tokenizer    = tokenizer
        self.max_len      = max_len

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
            'fine_label':     torch.tensor(self.fine_labels[idx],   dtype=torch.long),
            'coarse_label':   torch.tensor(self.coarse_labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class HierarchicalFakeNewsClassifier(nn.Module):
    """
    Two-stage hierarchical classifier:
      Stage 1 (coarse_head) → group (0 / 1 / 2)
      Stage 2 (fine_heads)  → category within that group
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden = self.bert.config.hidden_size          # 768

        self.coarse_head = nn.Linear(hidden, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(hidden, 1),   # Group 0: True Content only
            nn.Linear(hidden, 2),   # Group 1: Misleading, False Connection
            nn.Linear(hidden, 3),   # Group 2: Satire, Imposter, Manipulated
        ])

    def forward(self, input_ids, attention_mask):
        out          = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls          = out.pooler_output                       # (B, 768)
        coarse_logits = self.coarse_head(cls)                  # (B, 3)
        fine_logits   = [head(cls) for head in self.fine_heads]  # list of (B, k)
        return coarse_logits, fine_logits


# ── Loss ──────────────────────────────────────────────────────────────────────
def hierarchical_loss(coarse_logits, fine_logits, coarse_labels, fine_labels, alpha=0.5):
    """
    Joint loss: 0.5 * coarse_CE  +  0.5 * mean(per-group fine_CE)
    Only samples belonging to group g contribute to fine_heads[g].
    """
    ce         = nn.CrossEntropyLoss()
    coarse_loss = ce(coarse_logits, coarse_labels)

    fine_loss  = torch.tensor(0.0, device=coarse_logits.device)
    count      = 0
    for g in range(3):
        mask = (coarse_labels == g)
        if mask.sum() == 0:
            continue
        local_labels = torch.tensor(
            [FINE_LOCAL_INDEX[g][fl.item()] for fl in fine_labels[mask]],
            device=coarse_logits.device
        )
        fine_loss += ce(fine_logits[g][mask], local_labels)
        count     += 1

    if count > 0:
        fine_loss /= count

    return alpha * coarse_loss + (1 - alpha) * fine_loss


# ── Inference helper ──────────────────────────────────────────────────────────
def predict_6way(coarse_logits, fine_logits):
    coarse_preds = coarse_logits.argmax(dim=1)
    preds = []
    for i, g in enumerate(coarse_preds):
        g     = g.item()
        local = fine_logits[g][i].argmax().item()
        preds.append(LOCAL_TO_FINE[(g, local)])
    return preds


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        fine_labels    = batch['fine_label'].to(device)
        coarse_labels  = batch['coarse_label'].to(device)

        optimizer.zero_grad()
        coarse_logits, fine_logits = model(input_ids, attention_mask)
        loss = hierarchical_loss(coarse_logits, fine_logits, coarse_labels, fine_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')

    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coarse_logits, fine_logits = model(input_ids, attention_mask)
            preds = predict_6way(coarse_logits, fine_logits)
            all_preds.extend(preds)
            all_labels.extend(batch['fine_label'].tolist())

    print(classification_report(all_labels, all_preds,
          target_names=['True Content', 'Satire/Parody', 'Misleading',
                        'Imposter', 'False Connection', 'Manipulated']))
    return all_preds


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.tsv",        sep='\t')
    val_df   = pd.read_csv(f"{DATA_DIR}/validate.tsv",     sep='\t')
    test_df  = pd.read_csv(f"{DATA_DIR}/test_public.tsv",  sep='\t')
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_ds  = FakedditDataset(train_df, tokenizer)
    val_ds    = FakedditDataset(val_df,   tokenizer)
    test_ds   = FakedditDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=2)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    model = HierarchicalFakeNewsClassifier().to(DEVICE)

    optimizer    = AdamW(model.parameters(), lr=LR)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{DATA_DIR}/best_model.pt')
            print("  Model saved to Drive!")

    print("\nTest set evaluation:")
    evaluate(model, test_loader, DEVICE)
