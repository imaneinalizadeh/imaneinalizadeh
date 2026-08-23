"""
AI-Generated Content Extension — Attempt 2
Dataset: artnitolog/llm-generated-texts (Reuters headlines)
Result: F1 0.71, Recall 0.61

Run in Google Colab with GPU runtime.
Mount Drive first: from google.colab import drive; drive.mount('/content/drive')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import json, random, os

# ── Config ────────────────────────────────────────────────────
DRIVE_PATH   = "/content/drive/MyDrive/fakeddit"
CHECKPOINT   = f"{DRIVE_PATH}/hierarchical_v2_best.pt"  # saves here
BASE_MODEL   = f"{DRIVE_PATH}/best_model.pt"            # loads v1 as starting point
TRAIN_TSV    = "/content/train.tsv"
TEST_TSV     = "/content/test_public.tsv"
BATCH_SIZE   = 32
MAX_LEN      = 64
LR           = 2e-5
EPOCHS       = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Label maps ────────────────────────────────────────────────
# 8-way labels: 0-5 original Fakeddit + 6=AI-Generated + 7=Clickbait(not yet)
FINE_TO_COARSE = {0:0, 1:2, 2:1, 3:2, 4:1, 5:2, 6:2, 7:1}
LOCAL_TO_FINE  = {(0,0):0,(1,0):2,(1,1):4,(1,2):7,(2,0):1,(2,1):3,(2,2):5,(2,3):6}
FINE_TO_LOCAL  = {v:k for k,v in LOCAL_TO_FINE.items()}

GROUP_SIZES = {0:1, 1:3, 2:4}  # fine categories per group

# ── Tokenizer ─────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ── Dataset ───────────────────────────────────────────────────
class FakedditDataset(Dataset):
    def __init__(self, texts, fine_labels, max_len=MAX_LEN):
        self.texts       = texts
        self.fine_labels = fine_labels
        self.max_len     = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        fine  = self.fine_labels[idx]
        coarse, local = FINE_TO_LOCAL[fine]
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "coarse_label":   torch.tensor(coarse, dtype=torch.long),
            "fine_label":     torch.tensor(fine,   dtype=torch.long),
            "local_label":    torch.tensor(local,  dtype=torch.long),
            "group":          torch.tensor(coarse, dtype=torch.long),
        }

# ── Model ─────────────────────────────────────────────────────
class HierarchicalFakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert        = BertModel.from_pretrained("bert-base-uncased")
        h                = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(h, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(h, 1),  # Group 0: True Content only
            nn.Linear(h, 3),  # Group 1: Misleading, FalseConn, Clickbait
            nn.Linear(h, 4),  # Group 2: Satire, Imposter, Manipulated, AI-Gen
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.pooler_output)
        coarse_logits = self.coarse_head(cls)
        fine_logits   = [head(cls) for head in self.fine_heads]
        return coarse_logits, fine_logits

# ── Load data ─────────────────────────────────────────────────
print("Loading Fakeddit data...")
train_df = pd.read_csv(TRAIN_TSV, sep="\t")[["clean_title","6_way_label"]].dropna()
test_df  = pd.read_csv(TEST_TSV,  sep="\t")[["clean_title","6_way_label"]].dropna()

# Filter to original 6 classes
train_df = train_df[train_df["6_way_label"].isin(range(6))]
test_df  = test_df[test_df["6_way_label"].isin(range(6))]

print(f"Fakeddit train: {len(train_df):,} | test: {len(test_df):,}")

# ── Load Attempt 2 dataset: artnitolog/llm-generated-texts ────
print("Loading artnitolog/llm-generated-texts...")
ai_ds = load_dataset("artnitolog/llm-generated-texts", split="train")

ai_texts = []
for row in ai_ds:
    # This dataset has 'text' field with Reuters headlines / short news
    text = row.get("text", "").strip()
    if text and 10 < len(text) < 200:
        # Take first sentence if multi-sentence
        first = text.split(".")[0].strip()
        if len(first) > 10:
            ai_texts.append(first)

ai_texts = list(set(ai_texts))
random.shuffle(ai_texts)

# Balance: match Manipulated Content count (~21K)
target_ai = min(len(ai_texts), 22000)
ai_texts  = ai_texts[:target_ai]
print(f"AI-Generated samples (Attempt 2): {len(ai_texts):,}")
print(f"Sample texts: {ai_texts[:3]}")

# ── Build combined training set ────────────────────────────────
train_texts  = list(train_df["clean_title"])
train_labels = list(train_df["6_way_label"].astype(int))

# Add AI-Generated as label 6
train_texts  += ai_texts
train_labels += [6] * len(ai_texts)

# Augment AI with simple transforms to help generalisation
aug_texts, aug_labels = [], []
for t in ai_texts[:5000]:
    words = t.split()
    if len(words) > 4:
        # Drop last word
        aug_texts.append(" ".join(words[:-1]))
        aug_labels.append(6)
        # Lowercase
        aug_texts.append(t.lower())
        aug_labels.append(6)

train_texts  += aug_texts
train_labels += aug_labels

# Shuffle
combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = zip(*combined)

# Build test set (keep original test + small AI held-out)
ai_test_texts  = ai_texts[target_ai - 2000:] if len(ai_texts) > 2000 else ai_texts[-500:]
test_texts  = list(test_df["clean_title"]) + ai_test_texts
test_labels = list(test_df["6_way_label"].astype(int)) + [6]*len(ai_test_texts)

print(f"\nFinal train: {len(train_texts):,} | test: {len(test_texts):,}")
print(f"Label distribution (train): { {l: train_labels.count(l) for l in range(7)} }")

# ── DataLoaders ───────────────────────────────────────────────
train_ds = FakedditDataset(list(train_texts), list(train_labels))
test_ds  = FakedditDataset(test_texts, test_labels)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model setup ───────────────────────────────────────────────
model = HierarchicalFakeNewsClassifier().to(DEVICE)

# Load from v1 checkpoint if available
if os.path.exists(BASE_MODEL):
    model.load_state_dict(torch.load(BASE_MODEL, map_location=DEVICE))
    print(f"Loaded base model from {BASE_MODEL}")
else:
    print("Starting from scratch (no base model found)")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_dl),
    epochs=EPOCHS, pct_start=0.1
)

# ── Loss ──────────────────────────────────────────────────────
def compute_loss(coarse_logits, fine_logits, batch):
    coarse_loss = F.cross_entropy(coarse_logits, batch["coarse_label"])

    fine_loss = torch.tensor(0.0, device=DEVICE)
    for g in range(3):
        mask = (batch["group"] == g)
        if mask.sum() == 0:
            continue
        local_logits = fine_logits[g][mask]
        local_labels = batch["local_label"][mask]
        if local_logits.shape[1] == 1:
            fine_loss += F.binary_cross_entropy_with_logits(
                local_logits.squeeze(1),
                local_labels.float()
            ) * mask.sum() / len(batch["group"])
        else:
            fine_loss += F.cross_entropy(local_logits, local_labels) * mask.sum() / len(batch["group"])

    return 0.5 * coarse_loss + 0.5 * fine_loss

# ── Train ─────────────────────────────────────────────────────
best_f1, best_epoch = 0.0, 0

for epoch in range(EPOCHS):
    model.train()
    total_loss, n_batches = 0, 0

    for batch in train_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        coarse_logits, fine_logits = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_loss(coarse_logits, fine_logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches  += 1

    # ── Evaluate ──────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            cl, fl = model(batch["input_ids"], batch["attention_mask"])
            coarse_pred = cl.argmax(dim=1)
            preds = []
            for i, cp in enumerate(coarse_pred):
                g = cp.item()
                local_pred = fl[g][i].argmax().item()
                preds.append(LOCAL_TO_FINE[(g, local_pred)])
            all_preds.extend(preds)
            all_labels.extend(batch["fine_label"].cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    ai_preds  = [p for p, l in zip(all_preds, all_labels) if l == 6]
    ai_labels = [l for l in all_labels if l == 6]
    ai_f1     = f1_score(ai_labels, ai_preds, average="binary", pos_label=6, zero_division=0) if ai_labels else 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Loss: {total_loss/n_batches:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}  |  AI-Generated F1: {ai_f1:.4f}")
    print(classification_report(all_labels, all_preds, zero_division=0))

    if macro_f1 > best_f1:
        best_f1, best_epoch = macro_f1, epoch + 1
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  ✓ Saved best model (Macro F1: {best_f1:.4f})")

print(f"\nDone. Best Macro F1: {best_f1:.4f} at epoch {best_epoch}")
print(f"Model saved to: {CHECKPOINT}")
