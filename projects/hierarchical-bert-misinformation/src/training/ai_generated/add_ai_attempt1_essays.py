"""
Step 3a: Taxonomy Extension — AI-Generated Content (Attempt 1: artem9k dataset)
================================================================================
Extends the trained hierarchical model with a 7th category: AI-Generated Content.
AI-Generated is placed in Group 2 (Fabricated/Manipulated) alongside Satire,
Imposter and Manipulated Content.

Dataset used: artem9k/ai-text-detection-pile
  - 1.39 M samples of human and AI text (GPT-2, GPT-3, ChatGPT, GPT-J)
  - We sample 30 000 AI-generated samples

Strategy:
  1. Load the pre-trained hierarchical model (best_model.pt)
  2. Reinitialise fine_heads[2] with 4 outputs (was 3)
  3. Freeze BERT encoder and fine_heads[0], fine_heads[1]
  4. Train only fine_heads[2] for 2 epochs

Expected result (from paper):
  AI-Generated F1 = 0.11  ← VERY LOW due to domain mismatch
  Precision = 0.94, Recall = 0.18
  Diagnosis: long essays ≠ short Reddit post titles (8–12 words)
  → leads to Step 3b with better-matched data

Outputs: hierarchical_v2_best.pt (saved to Drive)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from datasets import load_dataset

# ── Label maps (7-way) ────────────────────────────────────────────────────────
# Label 6 = AI-Generated, placed in Group 2
FINE_TO_COARSE   = {0: 0, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2}
LOCAL_TO_FINE    = {(0, 0): 0, (1, 0): 2, (1, 1): 4,
                    (2, 0): 1, (2, 1): 3, (2, 2): 5, (2, 3): 6}
FINE_LOCAL_INDEX = {0: {0: 0}, 1: {2: 0, 4: 1}, 2: {1: 0, 3: 1, 5: 2, 6: 3}}

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/fakeddit"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {DEVICE}")


# ── Datasets ──────────────────────────────────────────────────────────────────
class FakedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts         = df['clean_title'].fillna('').tolist()
        self.fine_labels   = df['6_way_label'].tolist()
        self.coarse_labels = [FINE_TO_COARSE[l] for l in self.fine_labels]
        self.tokenizer     = tokenizer
        self.max_len       = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'fine_label':     torch.tensor(self.fine_labels[idx],   dtype=torch.long),
            'coarse_label':   torch.tensor(self.coarse_labels[idx], dtype=torch.long),
        }


class AIDataset(Dataset):
    """Wraps a list of AI-generated texts; all labelled as label 6 / group 2."""
    def __init__(self, texts, tokenizer, max_len=64):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'fine_label':     torch.tensor(6, dtype=torch.long),
            'coarse_label':   torch.tensor(2, dtype=torch.long),
        }


# ── Model (V2 — Group 2 head expanded to 4 outputs) ──────────────────────────
class HierarchicalFakeNewsClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden    = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(hidden, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(hidden, 1),   # Group 0
            nn.Linear(hidden, 2),   # Group 1
            nn.Linear(hidden, 4),   # Group 2: Satire / Imposter / Manipulated / AI-Generated
        ])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.pooler_output
        return self.coarse_head(cls), [head(cls) for head in self.fine_heads]


def predict_7way(coarse_logits, fine_logits):
    coarse_preds = coarse_logits.argmax(dim=1)
    preds = []
    for i, g in enumerate(coarse_preds):
        g     = g.item()
        local = fine_logits[g][i].argmax().item()
        preds.append(LOCAL_TO_FINE[(g, local)])
    return preds


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coarse_logits, fine_logits = model(input_ids, attention_mask)
            preds = predict_7way(coarse_logits, fine_logits)
            all_preds.extend(preds)
            all_labels.extend(batch['fine_label'].tolist())
    target_names = ['True', 'Satire', 'Misleading', 'Imposter',
                    'FalseConnection', 'Manipulated', 'AI-Generated']
    print(classification_report(all_labels, all_preds, target_names=target_names))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading Fakeddit data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df  = pd.read_csv(f"{DATA_DIR}/train.tsv",       sep='\t')
    test_df   = pd.read_csv(f"{DATA_DIR}/test_public.tsv", sep='\t')

    # ── Load AI-generated data (Attempt 1: long essays — expect poor recall) ──
    print("Loading AI-generated dataset (artem9k/ai-text-detection-pile) ...")
    ai_dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")
    ai_texts   = [x['text'] for x in ai_dataset if x['source'] != 'human'][:30_000]
    print(f"Loaded {len(ai_texts)} AI-generated samples")
    # NOTE: These are long formal essays — very different from short Reddit titles.
    # This intentional mismatch is Attempt 1 in the iterative dataset refinement experiment.

    split          = int(0.8 * len(ai_texts))
    ai_train_texts = ai_texts[:split]
    ai_test_texts  = ai_texts[split:]

    fakeddit_train = FakedditDataset(train_df, tokenizer)
    ai_train_ds    = AIDataset(ai_train_texts, tokenizer)
    ai_test_ds     = AIDataset(ai_test_texts,  tokenizer)
    fakeddit_test  = FakedditDataset(test_df,  tokenizer)

    combined_train = ConcatDataset([fakeddit_train, ai_train_ds])
    combined_test  = ConcatDataset([fakeddit_test,  ai_test_ds])

    train_loader = DataLoader(combined_train, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(combined_test,  batch_size=128,
                              num_workers=2, pin_memory=True)

    # ── Build V2 model and transfer weights ───────────────────────────────────
    print("Loading pretrained hierarchical model...")
    model     = HierarchicalFakeNewsClassifierV2().to(DEVICE)
    old_state = torch.load(f"{DATA_DIR}/best_model.pt", map_location=DEVICE)
    new_state = model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
    model.load_state_dict(new_state)
    print("Pretrained weights loaded (Group 2 head reinitialised)")

    # ── Freeze everything except fine_heads[2] ────────────────────────────────
    for name, param in model.named_parameters():
        if 'fine_heads.2' not in name:
            param.requires_grad = False
    print("All layers frozen except Group 2 fine head")

    # ── Train ─────────────────────────────────────────────────────────────────
    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    total_steps = len(train_loader) * 2
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            fine_labels    = batch['fine_label'].to(DEVICE)
            coarse_labels  = batch['coarse_label'].to(DEVICE)

            optimizer.zero_grad()
            coarse_logits, fine_logits = model(input_ids, attention_mask)

            mask = (coarse_labels == 2)
            if mask.sum() == 0:
                continue

            g2_logits = fine_logits[2][mask]
            g2_labels = fine_labels[mask]
            g2_local  = torch.tensor(
                [FINE_LOCAL_INDEX[2][l.item()] for l in g2_labels],
                dtype=torch.long).to(DEVICE)

            loss = criterion(g2_logits, g2_local)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i % 500 == 0:
                print(f"  Epoch {epoch+1} Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"{DATA_DIR}/hierarchical_v2_best.pt")
        print("✓ Model saved")

    print("\nEvaluating extended model...")
    evaluate(model, test_loader, DEVICE)
    # Expected: precision=0.94, recall=0.18, F1=0.11 for AI-Generated
    # → proceed to step3b with better-matched data
