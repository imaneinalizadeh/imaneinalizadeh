"""
Extensibility Experiment — Round 1: Conspiracy Theory Training
==============================================================
Adds Conspiracy Theory (label 8) to Group 2.
Group 2 fine head expanded from 4 → 5 outputs.

Label map after this round:
  Group 2: Satire(1), Imposter(3), Manipulated(5), AI-Gen(6), Conspiracy(8)

Viability criteria (strict):
  ✓ New category F1 > 0.65
  ✓ Max existing category F1 drop < 0.03
  ✓ Routing error rate change < +5%
  ✓ ECE after adding < 0.20

Baseline (hierarchical_v5_best.pt):
  Accuracy: 82.43% | Macro F1: 0.7755
  Routing errors: 71.2% of all errors
  ECE: ~0.13 (estimated from v4, v5 slightly better due to label smoothing)

Author: Iman Ein Alizadeh (s2901349)
University of Edinburgh EPCC — MSc Dissertation 2025-26
"""

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
import json, random, os

# ── Config ────────────────────────────────────────────────────
DRIVE_ROOT  = "/content/drive/MyDrive"
DRIVE_DATA  = "/content/drive/MyDrive/fakeddit"
BASE_MODEL  = f"{DRIVE_ROOT}/hierarchical_v5_best.pt"
SAVE_PATH   = f"{DRIVE_ROOT}/hierarchical_v6_conspiracy.pt"
TRAIN_TSV   = f"{DRIVE_DATA}/train.tsv"
TEST_TSV    = f"{DRIVE_DATA}/test_public.tsv"
CONSPIRACY_DATA = f"{DRIVE_DATA}/conspiracy_dataset.json"

EPOCHS      = 3
BATCH_SIZE  = 32
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Label maps (9-way: adding Conspiracy Theory as label 8) ───
# Group 2 now has 5 fine classes: Satire, Imposter, Manipulated, AI-Gen, Conspiracy
FINE_TO_COARSE = {0:0, 1:2, 2:1, 3:2, 4:1, 5:2, 6:2, 7:1, 8:2}
LOCAL_TO_FINE  = {
    (0,0):0,
    (1,0):2,(1,1):4,(1,2):7,
    (2,0):1,(2,1):3,(2,2):5,(2,3):6,(2,4):8   # ← 8=Conspiracy added
}
FINE_TO_LOCAL  = {v:k for k,v in LOCAL_TO_FINE.items()}

CATEGORY_NAMES = {
    0:"True Content", 1:"Satire/Parody", 2:"Misleading Content",
    3:"Imposter Content", 4:"False Connection", 5:"Manipulated Content",
    6:"AI-Generated Content", 7:"Clickbait", 8:"Conspiracy Theory"
}
GROUP_NAMES = {0:"Authentic", 1:"Structural Deception", 2:"Fabricated/Manipulated"}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ── Model (9-category: Group 2 now has 5 outputs) ─────────────
class HierarchicalFakeNewsClassifier(nn.Module):
    def __init__(self, g2_classes=5):
        super().__init__()
        self.bert        = BertModel.from_pretrained("bert-base-uncased")
        h                = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(h, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(h, 1),          # Group 0: True Content
            nn.Linear(h, 3),          # Group 1: Misleading, FalseConn, Clickbait
            nn.Linear(h, g2_classes), # Group 2: Satire, Imposter, Manip, AI-Gen, Conspiracy
        ])

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.coarse_head(cls), [h(cls) for h in self.fine_heads]

# ── Dataset ───────────────────────────────────────────────────
class FakedditDataset(Dataset):
    def __init__(self, texts, labels, max_len=64):
        self.texts, self.labels = texts, labels
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), max_length=self.max_len,
                        padding="max_length", truncation=True, return_tensors="pt")
        fine   = self.labels[idx]
        coarse, local = FINE_TO_LOCAL[fine]
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "fine_label":     torch.tensor(fine,   dtype=torch.long),
            "coarse_label":   torch.tensor(coarse, dtype=torch.long),
            "local_label":    torch.tensor(local,  dtype=torch.long),
            "group":          torch.tensor(coarse, dtype=torch.long),
        }

# ── Load base model with partial weight transfer ───────────────
print("Loading base model (hierarchical_v5_best.pt)...")
new_model = HierarchicalFakeNewsClassifier(g2_classes=5).to(DEVICE)

# Load v5 weights (Group 2 had 4 outputs, now needs 5)
old_state = torch.load(BASE_MODEL, map_location=DEVICE)
new_state = new_model.state_dict()

transferred, skipped = 0, 0
for k, v in old_state.items():
    if k in new_state:
        if new_state[k].shape == v.shape:
            new_state[k] = v
            transferred += 1
        else:
            # Shape mismatch — Group 2 fine head (4→5 outputs)
            # Copy existing 4 rows, leave 5th row randomly initialised
            if 'fine_heads.2' in k and 'weight' in k:
                new_state[k][:4, :] = v  # copy existing category weights
                # 5th row stays randomly initialised for Conspiracy Theory
                print(f"  Partial transfer: {k} ({v.shape} → {new_state[k].shape})")
            elif 'fine_heads.2' in k and 'bias' in k:
                new_state[k][:4] = v
                print(f"  Partial transfer: {k} ({v.shape} → {new_state[k].shape})")
            skipped += 1
    else:
        skipped += 1

new_model.load_state_dict(new_state)
print(f"Transferred: {transferred} layers | Skipped/partial: {skipped} layers")
print("Base model loaded with partial weight transfer for new category")

# ── Load data ─────────────────────────────────────────────────
print("\nLoading Fakeddit data...")
train_df = pd.read_csv(TRAIN_TSV, sep="\t")[["clean_title","6_way_label"]].dropna()
test_df  = pd.read_csv(TEST_TSV,  sep="\t")[["clean_title","6_way_label"]].dropna()
train_df = train_df[train_df["6_way_label"].isin(range(6))]
test_df  = test_df[test_df["6_way_label"].isin(range(6))]
print(f"Fakeddit: {len(train_df):,} train | {len(test_df):,} test")

print("Loading conspiracy dataset...")
with open(CONSPIRACY_DATA) as f:
    conspiracy = json.load(f)
conspiracy_train = conspiracy["train"]
conspiracy_test  = conspiracy["test"]
print(f"Conspiracy: {len(conspiracy_train):,} train | {len(conspiracy_test):,} test")

# ── Build combined training set ───────────────────────────────
train_texts  = list(train_df["clean_title"]) + conspiracy_train
train_labels = list(train_df["6_way_label"].astype(int)) + [8]*len(conspiracy_train)

# Augment Conspiracy Theory (it's a new class, needs extra signal)
aug_t, aug_l = [], []
for text in random.sample(conspiracy_train, min(3000, len(conspiracy_train))):
    words = text.split()
    if len(words) > 3:
        aug_t.append(text.lower());            aug_l.append(8)
        aug_t.append(" ".join(words[:-1]));    aug_l.append(8)
train_texts  += aug_t
train_labels += aug_l

combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = zip(*combined)

test_texts  = list(test_df["clean_title"]) + conspiracy_test
test_labels = list(test_df["6_way_label"].astype(int)) + [8]*len(conspiracy_test)

print(f"\nFinal train: {len(train_texts):,} | test: {len(test_texts):,}")
label_counts = {i: list(train_labels).count(i) for i in range(9) if list(train_labels).count(i) > 0}
print(f"Label distribution: {label_counts}")

# ── DataLoaders ───────────────────────────────────────────────
train_ds = FakedditDataset(list(train_texts), list(train_labels))
test_ds  = FakedditDataset(test_texts, test_labels)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=64,         shuffle=False, num_workers=2, pin_memory=True)

# ── Optimizer: same differential LRs as v5 ────────────────────
# Freeze Group 0 and Group 1 fine heads — only Group 2 and coarse head train
for param in new_model.fine_heads[0].parameters(): param.requires_grad = False
for param in new_model.fine_heads[1].parameters(): param.requires_grad = False

optimizer = AdamW([
    {"params": new_model.bert.parameters(),           "lr": 1e-5,  "weight_decay": 0.01},
    {"params": new_model.coarse_head.parameters(),    "lr": 3e-5,  "weight_decay": 0.01},
    {"params": new_model.fine_heads[2].parameters(),  "lr": 2e-5,  "weight_decay": 0.01},
])

total_steps = len(train_dl) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(optimizer,
                  num_warmup_steps=int(0.1*total_steps),
                  num_training_steps=total_steps)

# ── Class weights ─────────────────────────────────────────────
# Conspiracy Theory gets 3x — new class being introduced
# Keep existing weights from v5 for other hard categories
CLASS_WEIGHTS = torch.tensor(
    [1.0, 3.0, 1.0, 5.0, 1.0, 3.0, 2.0, 1.0, 3.0],  # index 8 = Conspiracy
    device=DEVICE
)

def compute_loss(cl, fl, batch):
    coarse_loss = F.cross_entropy(cl, batch["coarse_label"], label_smoothing=0.1)
    fine_loss   = torch.tensor(0.0, device=DEVICE)
    for g in range(3):
        mask = (batch["group"] == g)
        if mask.sum() == 0: continue
        gl = fl[g][mask]
        ll = batch["local_label"][mask]
        w  = CLASS_WEIGHTS[batch["fine_label"][mask]]
        if gl.shape[1] == 1:
            base = F.binary_cross_entropy_with_logits(gl.squeeze(1), ll.float(), reduction="none")
        else:
            base = F.cross_entropy(gl, ll, label_smoothing=0.1, reduction="none")
        fine_loss += (base * w).mean() * mask.sum() / len(batch["group"])
    return 0.5 * coarse_loss + 0.5 * fine_loss

# ── Evaluation ────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    all_coarse_preds, all_coarse_true = [], []
    all_confidences = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            cl, fl = model(ids, mask)
            cp = cl.argmax(dim=1)
            for i, g in enumerate(cp.tolist()):
                lp   = fl[g][i].argmax().item()
                pred = LOCAL_TO_FINE[(g, lp)]
                conf = torch.softmax(fl[g][i], dim=0)[lp].item()
                all_preds.append(pred)
                all_coarse_preds.append(g)
                all_confidences.append(conf)
            all_labels.extend(batch["fine_label"].tolist())
            all_coarse_true.extend([FINE_TO_COARSE[l.item()] for l in batch["fine_label"]])

    # Overall metrics
    orig_labels = [l for l in all_labels if l < 8]
    orig_preds  = [p for p,l in zip(all_preds, all_labels) if l < 8]
    acc         = sum(p==l for p,l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1    = f1_score(all_labels, all_preds, labels=list(range(9)), average="macro", zero_division=0)

    # Routing error rate
    errors       = [(p,l) for p,l in zip(all_preds, all_labels) if p!=l]
    routing_errs = [(p,l) for p,l in errors if FINE_TO_COARSE.get(p,-1) != FINE_TO_COARSE.get(l,-2)]
    routing_rate = len(routing_errs)/len(errors)*100 if errors else 0

    # ECE
    ece = 0
    for lo, hi in [(i/10,(i+1)/10) for i in range(10)]:
        mask_ = [lo<=c<hi for c in all_confidences]
        n     = sum(mask_)
        if n == 0: continue
        acc_  = sum(p==l for p,l,m in zip(all_preds,all_labels,mask_) if m)/n
        avgc  = np.mean([c for c,m in zip(all_confidences,mask_) if m])
        ece  += (n/len(all_labels)) * abs(acc_-avgc)

    # Conspiracy-specific metrics
    con_labels = [l for l in all_labels if l==8]
    con_preds  = [p for p,l in zip(all_preds,all_labels) if l==8]
    con_f1     = f1_score(con_labels, con_preds, average="binary", pos_label=8, zero_division=0) if con_labels else 0

    return acc, macro_f1, routing_rate, ece, con_f1, all_preds, all_labels

# ── Baseline before training ──────────────────────────────────
print("\n" + "="*60)
print("BASELINE (hierarchical_v5_best.pt before Conspiracy added):")
acc, mf1, rr, ece, con_f1, _, _ = evaluate(new_model, test_dl)
print(f"  Accuracy:      {acc*100:.2f}%")
print(f"  Macro F1:      {mf1:.4f}")
print(f"  Routing errors:{rr:.1f}%")
print(f"  ECE:           {ece:.4f}")
print(f"  Conspiracy F1: {con_f1:.4f} (random init — expected ~0)")
print("="*60)

baseline_per_cat = {}
# Get per-category F1 for regression check
from sklearn.metrics import classification_report as cr
_, _, _, _, _, bp, bl = evaluate(new_model, test_dl)
report = cr(bl, bp, labels=list(range(8)), zero_division=0, output_dict=True)
for i in range(8):
    name = CATEGORY_NAMES[i]
    baseline_per_cat[i] = report.get(str(i), {}).get('f1-score', 0)
    print(f"  Baseline {name}: F1 {baseline_per_cat[i]:.4f}")

best_acc = acc
best_mf1 = mf1

# ── Training ──────────────────────────────────────────────────
print("\nTraining Round 1 — Conspiracy Theory extension...\n")

for epoch in range(EPOCHS):
    new_model.train()
    total_loss, nb = 0, 0

    for batch_idx, batch in enumerate(train_dl):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        cl, fl = new_model(batch["input_ids"], batch["attention_mask"])
        loss   = compute_loss(cl, fl, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item(); nb += 1

        if batch_idx % 1000 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_dl)} | Loss: {loss.item():.4f}")

    acc, mf1, rr, ece, con_f1, preds, labels = evaluate(new_model, test_dl)
    avg_loss = total_loss / nb

    print(f"\nEpoch {epoch+1}/{EPOCHS} — Avg Loss: {avg_loss:.4f}")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  Macro F1:           {mf1:.4f}")
    print(f"  Conspiracy F1:      {con_f1:.4f}")
    print(f"  Routing error rate: {rr:.1f}%")
    print(f"  ECE:                {ece:.4f}")

    # Per-category regression check
    report = cr(labels, preds, labels=list(range(9)), zero_division=0, output_dict=True)
    max_drop = 0
    print(f"\n  Per-category F1 vs baseline:")
    for i in range(8):
        name    = CATEGORY_NAMES[i]
        new_f1  = report.get(str(i), {}).get('f1-score', 0)
        delta   = new_f1 - baseline_per_cat[i]
        max_drop = min(max_drop, delta)
        flag    = "✗ REGRESSION" if delta < -0.03 else ("✓" if delta >= 0 else "")
        print(f"    {name:<25}: {new_f1:.4f} ({delta:+.4f}) {flag}")
    con_report = report.get('8', {})
    print(f"    {'Conspiracy Theory':<25}: {con_report.get('f1-score',0):.4f} (NEW)")

    # ── Viability check ───────────────────────────────────────
    print(f"\n  VIABILITY CHECK (strict — fail ANY = stop):")
    c1 = con_f1 > 0.65;             print(f"    New F1 > 0.65:              {con_f1:.4f} {'✓' if c1 else '✗ FAIL'}")
    c2 = abs(max_drop) < 0.03;      print(f"    Max existing drop < 0.03:  {max_drop:.4f} {'✓' if c2 else '✗ FAIL'}")
    c3 = rr < 76.2;                 print(f"    Routing rate < baseline+5%: {rr:.1f}% {'✓' if c3 else '✗ FAIL'}")
    c4 = ece < 0.20;                print(f"    ECE < 0.20:                 {ece:.4f} {'✓' if c4 else '✗ FAIL'}")

    if all([c1, c2, c3, c4]):
        print(f"\n  ✓ ALL CRITERIA MET — Conspiracy Theory extension VIABLE")
    else:
        failed = [n for n,c in [("New F1",c1),("Regression",c2),("Routing",c3),("ECE",c4)] if not c]
        print(f"\n  ✗ FAILED: {', '.join(failed)} — extensibility limit may be reached")

    if acc > best_acc or (acc == best_acc and mf1 > best_mf1):
        best_acc, best_mf1 = acc, mf1
        torch.save(new_model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model to {SAVE_PATH}")

print(f"\n{'='*60}")
print(f"ROUND 1 COMPLETE")
print(f"  Best Accuracy:    {best_acc*100:.2f}%")
print(f"  Best Macro F1:    {best_mf1:.4f}")
print(f"  Saved to:         {SAVE_PATH}")
print(f"{'='*60}")
print("\nPaste the full output — I will record the viability verdict and prepare Round 2.")
