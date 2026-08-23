"""
Extensibility Experiment — Round 2: Propaganda Training
========================================================
Adds Propaganda (label 8) to Group 2.
Group 2 fine head expanded from 4 → 5 outputs.

Dataset: Fakeddit r/propagandaposters — 13,456 train | 1,455 test
Base model: hierarchical_v5_best.pt (shared from ieinalizadeh Drive)

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

# ── Paths (all on ieinalizadeh — confirmed working) ────────────
BASE_MODEL = "/content/drive/MyDrive/hierarchical_v5_best.pt"
SAVE_PATH  = "/content/drive/MyDrive/hierarchical_v6_propaganda.pt"
TRAIN_TSV  = "/content/drive/MyDrive/fakeddit/train.tsv"
TEST_TSV   = "/content/drive/MyDrive/fakeddit/test_public.tsv"
PROP_DATA  = "/content/drive/MyDrive/fakeddit/propaganda_dataset.json"
print(f"Using model: {BASE_MODEL}")

EPOCHS     = 3
BATCH_SIZE = 32
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Label maps (9-way: Propaganda = label 8, Group 2) ─────────
FINE_TO_COARSE = {0:0, 1:2, 2:1, 3:2, 4:1, 5:2, 6:2, 7:1, 8:2}
LOCAL_TO_FINE  = {
    (0,0):0,
    (1,0):2,(1,1):4,(1,2):7,
    (2,0):1,(2,1):3,(2,2):5,(2,3):6,(2,4):8
}
FINE_TO_LOCAL  = {v:k for k,v in LOCAL_TO_FINE.items()}
CATEGORY_NAMES = {
    0:"True Content", 1:"Satire/Parody", 2:"Misleading Content",
    3:"Imposter Content", 4:"False Connection", 5:"Manipulated Content",
    6:"AI-Generated Content", 7:"Clickbait", 8:"Propaganda"
}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ── Model (Group 2 now has 5 outputs) ─────────────────────────
class HierarchicalFakeNewsClassifier(nn.Module):
    def __init__(self, g2_classes=5):
        super().__init__()
        self.bert        = BertModel.from_pretrained("bert-base-uncased")
        h                = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(h, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(h, 1),
            nn.Linear(h, 3),
            nn.Linear(h, g2_classes),
        ])
    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.coarse_head(cls), [h(cls) for h in self.fine_heads]

# ── Load base model with partial weight transfer ───────────────
print(f"\nLoading {BASE_MODEL}...")
new_model = HierarchicalFakeNewsClassifier(g2_classes=5).to(DEVICE)
old_state  = torch.load(BASE_MODEL, map_location=DEVICE)
new_state  = new_model.state_dict()

transferred = 0
for k, v in old_state.items():
    if k in new_state:
        if new_state[k].shape == v.shape:
            new_state[k] = v
            transferred += 1
        elif 'fine_heads.2' in k:
            if 'weight' in k:
                new_state[k][:4, :] = v
                print(f"  Partial: {k} {v.shape}→{new_state[k].shape}")
            elif 'bias' in k:
                new_state[k][:4] = v
                print(f"  Partial: {k} {v.shape}→{new_state[k].shape}")

new_model.load_state_dict(new_state)
print(f"Model loaded — {transferred} layers transferred")

# ── Dataset ───────────────────────────────────────────────────
class FakedditDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), max_length=64,
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

# ── Load data ─────────────────────────────────────────────────
print("\nLoading data...")
train_df = pd.read_csv(TRAIN_TSV, sep="\t")[["clean_title","6_way_label","subreddit"]].dropna()
test_df  = pd.read_csv(TEST_TSV,  sep="\t")[["clean_title","6_way_label","subreddit"]].dropna()

# Remove propagandaposters from Manipulated Content
# (we are re-labelling them as Propaganda label 8)
train_df = train_df[~((train_df['subreddit']=='propagandaposters') & (train_df['6_way_label']==5))]
test_df  = test_df[~((test_df['subreddit']=='propagandaposters')   & (test_df['6_way_label']==5))]
train_df = train_df[train_df["6_way_label"].isin(range(6))]
test_df  = test_df[test_df["6_way_label"].isin(range(6))]
print(f"Fakeddit (excl. propaganda): {len(train_df):,} train | {len(test_df):,} test")

with open(PROP_DATA) as f:
    prop = json.load(f)
print(f"Propaganda: {len(prop['train']):,} train | {len(prop['test']):,} test")

train_texts  = list(train_df["clean_title"]) + prop["train"]
train_labels = list(train_df["6_way_label"].astype(int)) + [8]*len(prop["train"])
test_texts   = list(test_df["clean_title"])  + prop["test"]
test_labels  = list(test_df["6_way_label"].astype(int))  + [8]*len(prop["test"])

combined = list(zip(train_texts, train_labels))
random.shuffle(combined)
train_texts, train_labels = zip(*combined)

print(f"Final train: {len(train_texts):,} | test: {len(test_texts):,}")
lc = {i: list(train_labels).count(i) for i in range(9) if list(train_labels).count(i)>0}
print(f"Label distribution: {lc}")

train_ds = FakedditDataset(list(train_texts), list(train_labels))
test_ds  = FakedditDataset(test_texts, test_labels)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=64,         shuffle=False, num_workers=2, pin_memory=True)

# ── Optimizer: freeze Group 0 and Group 1 fine heads ──────────
for param in new_model.fine_heads[0].parameters(): param.requires_grad = False
for param in new_model.fine_heads[1].parameters(): param.requires_grad = False

optimizer = AdamW([
    {"params": new_model.bert.parameters(),          "lr": 1e-5, "weight_decay": 0.01},
    {"params": new_model.coarse_head.parameters(),   "lr": 3e-5, "weight_decay": 0.01},
    {"params": new_model.fine_heads[2].parameters(), "lr": 2e-5, "weight_decay": 0.01},
])
scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=int(0.1*len(train_dl)*EPOCHS),
    num_training_steps=len(train_dl)*EPOCHS)

CLASS_WEIGHTS = torch.tensor([1.0,3.0,1.0,5.0,1.0,3.0,2.0,1.0,3.0], device=DEVICE)

def compute_loss(cl, fl, batch):
    coarse_loss = F.cross_entropy(cl, batch["coarse_label"], label_smoothing=0.1)
    fine_loss   = torch.tensor(0.0, device=DEVICE)
    for g in range(3):
        mask = (batch["group"]==g)
        if mask.sum()==0: continue
        gl = fl[g][mask]; ll = batch["local_label"][mask]
        w  = CLASS_WEIGHTS[batch["fine_label"][mask]]
        if gl.shape[1]==1:
            base = F.binary_cross_entropy_with_logits(gl.squeeze(1), ll.float(), reduction="none")
        else:
            base = F.cross_entropy(gl, ll, label_smoothing=0.1, reduction="none")
        fine_loss += (base*w).mean() * mask.sum()/len(batch["group"])
    return 0.5*coarse_loss + 0.5*fine_loss

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            cl, fl = model(ids, mask)
            cp = cl.argmax(dim=1)
            for i, g in enumerate(cp.tolist()):
                lp   = fl[g][i].argmax().item()
                pred = LOCAL_TO_FINE[(g,lp)]
                conf = torch.softmax(fl[g][i],dim=0)[lp].item()
                all_preds.append(pred)
                all_confs.append(conf)
            all_labels.extend(batch["fine_label"].tolist())

    acc      = sum(p==l for p,l in zip(all_preds,all_labels))/len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, labels=list(range(9)), average="macro", zero_division=0)
    errors   = [(p,l) for p,l in zip(all_preds,all_labels) if p!=l]
    routing  = [(p,l) for p,l in errors if FINE_TO_COARSE.get(p,-1)!=FINE_TO_COARSE.get(l,-2)]
    rr       = len(routing)/len(errors)*100 if errors else 0
    ece = 0
    for lo,hi in [(i/10,(i+1)/10) for i in range(10)]:
        m  = [lo<=c<hi for c in all_confs]
        n  = sum(m)
        if n==0: continue
        a  = sum(p==l for p,l,m_ in zip(all_preds,all_labels,m) if m_)/n
        ac = np.mean([c for c,m_ in zip(all_confs,m) if m_])
        ece += (n/len(all_labels))*abs(a-ac)
    prop_mask   = [l==8 for l in all_labels]
    prop_labels = [l for l,m in zip(all_labels,prop_mask) if m]
    prop_preds  = [p for p,m in zip(all_preds,prop_mask) if m]
    prop_f1     = f1_score(prop_labels, prop_preds, average="macro", zero_division=0) if prop_labels else 0
    return acc, macro_f1, rr, ece, prop_f1, all_preds, all_labels

# ── Baseline ──────────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE before Propaganda added:")
acc, mf1, rr, ece, pf1, bp, bl = evaluate(new_model, test_dl)
print(f"  Accuracy: {acc*100:.2f}% | Macro F1: {mf1:.4f}")
print(f"  Routing:  {rr:.1f}%     | ECE: {ece:.4f}")
print(f"  Propaganda F1: {pf1:.4f} (random init)")
report     = classification_report(bl, bp, labels=list(range(8)), zero_division=0, output_dict=True)
baseline   = {i: report.get(str(i),{}).get('f1-score',0) for i in range(8)}
baseline_rr = rr
print("\n  Existing category baselines:")
for i in range(8):
    print(f"    {CATEGORY_NAMES[i]:<25}: {baseline[i]:.4f}")
print("="*60)

best_acc, best_mf1 = acc, mf1

# ── Train ──────────────────────────────────────────────────────
print("\nTraining — Round 2: Propaganda extension\n")
for epoch in range(EPOCHS):
    new_model.train()
    total_loss, nb = 0, 0
    for bidx, batch in enumerate(train_dl):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        optimizer.zero_grad()
        cl, fl = new_model(batch["input_ids"], batch["attention_mask"])
        loss   = compute_loss(cl, fl, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item(); nb += 1
        if bidx % 1000 == 0:
            print(f"  Ep{epoch+1} | Batch {bidx}/{len(train_dl)} | Loss {loss.item():.4f}")

    acc, mf1, rr, ece, pf1, preds, labels = evaluate(new_model, test_dl)
    print(f"\nEpoch {epoch+1}/{EPOCHS} — Loss: {total_loss/nb:.4f}")
    print(f"  Accuracy:      {acc*100:.2f}%  | Macro F1: {mf1:.4f}")
    print(f"  Propaganda F1: {pf1:.4f}       | Routing:  {rr:.1f}%  | ECE: {ece:.4f}")

    report   = classification_report(labels, preds, labels=list(range(9)), zero_division=0, output_dict=True)
    max_drop = 0
    print(f"\n  Per-category F1 vs baseline:")
    for i in range(8):
        nf1   = report.get(str(i),{}).get('f1-score',0)
        delta = nf1 - baseline[i]
        max_drop = min(max_drop, delta)
        flag  = " ✗ REGRESSION" if delta < -0.03 else ""
        print(f"    {CATEGORY_NAMES[i]:<25}: {nf1:.4f} ({delta:+.4f}){flag}")
    print(f"    {'Propaganda':<25}: {pf1:.4f} (NEW)")

    print(f"\n  VIABILITY CHECK (strict):")
    c1 = pf1 > 0.65
    c2 = abs(max_drop) < 0.03
    c3 = rr < baseline_rr + 5
    c4 = ece < 0.20
    print(f"    New category F1 > 0.65:       {pf1:.4f}  {'✓' if c1 else '✗ FAIL'}")
    print(f"    Max existing drop < 0.03:     {max_drop:.4f}  {'✓' if c2 else '✗ FAIL'}")
    print(f"    Routing < {baseline_rr:.1f}+5%:        {rr:.1f}%   {'✓' if c3 else '✗ FAIL'}")
    print(f"    ECE < 0.20:                   {ece:.4f}  {'✓' if c4 else '✗ FAIL'}")

    if all([c1,c2,c3,c4]):
        print(f"\n  ✓ ALL CRITERIA MET — Propaganda VIABLE")
    else:
        failed = [n for n,c in [("New F1",c1),("Regression",c2),("Routing",c3),("ECE",c4)] if not c]
        print(f"\n  ✗ FAILED: {', '.join(failed)}")

    if acc > best_acc:
        best_acc, best_mf1 = acc, mf1
        torch.save(new_model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved to {SAVE_PATH}")

print(f"\n{'='*60}")
print(f"ROUND 2 COMPLETE")
print(f"  Best Accuracy: {best_acc*100:.2f}%  |  Best Macro F1: {best_mf1:.4f}")
print(f"  Saved to: {SAVE_PATH}")
print(f"{'='*60}")
print("\nPaste full output back.")
