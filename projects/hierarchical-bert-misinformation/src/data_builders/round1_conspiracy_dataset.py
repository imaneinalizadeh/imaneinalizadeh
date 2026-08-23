"""
Extensibility Experiment — Round 1: Conspiracy Theory (Label 8)
================================================================
Research question: At what point does adding a new category stop being beneficial?

This script builds the Conspiracy Theory training dataset from:
1. Fakeddit subreddit-based posts (r/conspiracy, r/conspiracytheories, r/conspiracyfact)
2. HuggingFace: mediabiasgroup/MBIC-dataset (conspiracy-labelled samples)
3. HuggingFace: ERCDiDip/conspiracy-detection

Viability criteria (strict — fail ANY one = stop):
  - New category F1 > 0.65
  - Max existing category F1 drop < 0.03
  - Routing error rate change < +5%
  - ECE after adding < 0.20

Author: Iman Ein Alizadeh (s2901349)
University of Edinburgh EPCC — MSc Dissertation 2025-26
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import json, re, random, os
from datasets import load_dataset

DRIVE_DATA = "/content/drive/MyDrive/fakeddit"
SAVE_PATH  = f"{DRIVE_DATA}/conspiracy_dataset.json"
SEED       = 42
random.seed(SEED)

# ── Conspiracy linguistic patterns ────────────────────────────
# These are the surface signals that make conspiracy headlines
# linguistically distinct from other categories
CONSPIRACY_PATTERNS = [
    r'\bthey (don\'t want|never told|are hiding|won\'t tell)\b',
    r'\b(deep state|new world order|illuminati|cabal|elites?)\b',
    r'\b(exposed|uncovered|revealed|the truth about|wake up)\b',
    r'\b(false flag|crisis actor|hoax|staged|fake shooting)\b',
    r'\b(chemtrail|microchip|5g|fluoride|depopulation)\b',
    r'\b(mainstream media|msm|fake news|propaganda)\b',
    r'\b(agenda|cover.?up|suppressed|banned|censored)\b',
    r'\bwhat (they|the government|the media) (don\'t|won\'t|never)\b',
    r'\b(pills?|red.?pill|blue.?pill|wake up sheeple)\b',
    r'\b(follow the money|cui bono|who benefits)\b',
    r'\b(secret|hidden|classified|declassified) (documents?|files?|truth|agenda)\b',
    r'\bproof that\b',
]

CONSPIRACY_SUBREDDITS = {
    'conspiracy', 'conspiracytheories', 'conspiracyfact',
    'conspiracyNOPOL', 'C_S_T', 'conspiracies'
}

def score_conspiracy(text):
    """Score how conspiracy-like a headline is based on linguistic patterns."""
    text_lower = text.lower()
    score = 0
    for pattern in CONSPIRACY_PATTERNS:
        if re.search(pattern, text_lower):
            score += 1
    return score

# ── Source 1: Fakeddit train.tsv subreddit filter ─────────────
print("Loading Fakeddit train.tsv...")
train_df = pd.read_csv(f"{DRIVE_DATA}/train.tsv", sep="\t")
print(f"Total Fakeddit samples: {len(train_df):,}")

# Filter by subreddit
conspiracy_from_fakeddit = train_df[
    train_df['subreddit'].str.lower().isin(CONSPIRACY_SUBREDDITS)
][['clean_title', 'subreddit']].dropna()

print(f"Fakeddit conspiracy subreddit posts: {len(conspiracy_from_fakeddit):,}")
print(f"Subreddit breakdown:\n{conspiracy_from_fakeddit['subreddit'].value_counts()}")

# Score each title
conspiracy_from_fakeddit['score'] = conspiracy_from_fakeddit['clean_title'].apply(score_conspiracy)
print(f"\nScore distribution:")
print(conspiracy_from_fakeddit['score'].value_counts().sort_index())

# Take all subreddit-matched + at least 1 pattern match for quality
high_quality = conspiracy_from_fakeddit[conspiracy_from_fakeddit['score'] >= 1]
any_quality  = conspiracy_from_fakeddit[conspiracy_from_fakeddit['score'] == 0]
print(f"\nHigh quality (subreddit + pattern): {len(high_quality):,}")
print(f"Subreddit only (score=0):            {len(any_quality):,}")

fakeddit_texts = list(high_quality['clean_title']) + list(any_quality['clean_title'])
print(f"Total from Fakeddit: {len(fakeddit_texts):,}")

# ── Source 2: Pattern-matched from all Fakeddit Group 2 posts ─
print("\nPattern-matching conspiracy headlines from full Fakeddit...")
all_texts = train_df['clean_title'].dropna().tolist()
pattern_matched = [t for t in all_texts if score_conspiracy(t) >= 2]
print(f"Pattern-matched (score >= 2): {len(pattern_matched):,}")

# Sample to avoid overwhelming
random.shuffle(pattern_matched)
pattern_matched = pattern_matched[:5000]

# ── Source 3: HuggingFace conspiracy datasets ──────────────────
hf_texts = []

print("\nLoading HuggingFace conspiracy datasets...")

# Dataset 1: conspiracy-detection
try:
    ds1 = load_dataset("ERCDiDip/conspiracy-detection", split="train")
    print(f"ERCDiDip/conspiracy-detection: {len(ds1)} samples")
    print(f"Columns: {ds1.column_names}")
    for row in ds1:
        # Get the text field
        text = row.get('text', row.get('sentence', row.get('title', ''))).strip()
        label = row.get('label', row.get('conspiracy', 1))
        # Only take positive conspiracy examples
        if label == 1 and text and 10 < len(text) < 150:
            first = text.split('.')[0].strip()
            if len(first) > 10:
                hf_texts.append(first)
    print(f"  Extracted {len(hf_texts)} conspiracy texts")
except Exception as e:
    print(f"  Could not load ERCDiDip: {e}")

# Dataset 2: liar dataset (has conspiracy-related labels)
try:
    ds2 = load_dataset("liar", split="train")
    print(f"\nliar dataset: {len(ds2)} samples")
    conspiracy_liar = []
    for row in ds2:
        text = row.get('statement', '').strip()
        if text and score_conspiracy(text) >= 1 and 10 < len(text) < 150:
            conspiracy_liar.append(text)
    hf_texts.extend(conspiracy_liar)
    print(f"  Extracted {len(conspiracy_liar)} conspiracy-like texts from liar")
except Exception as e:
    print(f"  Could not load liar: {e}")

# Dataset 3: fake_news_english
try:
    ds3 = load_dataset("GonzaloA/fake_news", split="train")
    print(f"\nfake_news dataset: {len(ds3)} samples")
    conspiracy_fn = []
    for row in ds3:
        text = row.get('title', '').strip()
        if text and score_conspiracy(text) >= 2 and 10 < len(text) < 150:
            conspiracy_fn.append(text)
    hf_texts.extend(conspiracy_fn)
    print(f"  Extracted {len(conspiracy_fn)} conspiracy-like texts from fake_news")
except Exception as e:
    print(f"  Could not load fake_news: {e}")

print(f"\nTotal from HuggingFace: {len(hf_texts):,}")

# ── Combine and deduplicate ────────────────────────────────────
all_conspiracy = list(set(fakeddit_texts + pattern_matched + hf_texts))
random.shuffle(all_conspiracy)

# Filter: remove very short, very long, or duplicate-adjacent
all_conspiracy = [t for t in all_conspiracy if 8 < len(t) < 150]
print(f"\nAfter deduplication and filtering: {len(all_conspiracy):,}")

# Length analysis
lengths = [len(t) for t in all_conspiracy]
print(f"Avg length: {np.mean(lengths):.0f} chars")
print(f"Min: {min(lengths)} | Max: {max(lengths)}")

# Sample comparison with Fakeddit avg
print(f"\nFakeddit avg title length: ~55 chars")
print(f"Conspiracy dataset avg: {np.mean(lengths):.0f} chars")

# ── Score distribution of final dataset ───────────────────────
scores = [score_conspiracy(t) for t in all_conspiracy]
print(f"\nFinal dataset pattern score distribution:")
for s in range(max(scores)+1):
    n = scores.count(s)
    print(f"  Score {s}: {n:,} ({n/len(scores)*100:.1f}%)")

# ── Train/test split ──────────────────────────────────────────
target = min(len(all_conspiracy), 20000)
all_conspiracy = all_conspiracy[:target]

split     = int(len(all_conspiracy) * 0.9)
train_set = all_conspiracy[:split]
test_set  = all_conspiracy[split:]

print(f"\nFinal dataset:")
print(f"  Train: {len(train_set):,}")
print(f"  Test:  {len(test_set):,}")
print(f"  Total: {len(all_conspiracy):,}")

# ── Sample headlines ──────────────────────────────────────────
print(f"\nSample conspiracy headlines:")
for t in random.sample(train_set, min(15, len(train_set))):
    print(f"  ({len(t):3d}c) {t}")

# ── Save ──────────────────────────────────────────────────────
dataset = {
    "category":     "Conspiracy Theory",
    "label":        8,
    "group":        2,
    "description":  "Content promoting conspiracy theories — hidden agendas, cover-ups, secret organisations, false flag events",
    "total":        len(all_conspiracy),
    "train":        train_set,
    "test":         test_set,
    "sources": {
        "fakeddit_subreddits": len(fakeddit_texts),
        "pattern_matched":     len(pattern_matched),
        "huggingface":         len(hf_texts),
    },
    "patterns_used": CONSPIRACY_PATTERNS,
    "subreddits_used": list(CONSPIRACY_SUBREDDITS),
}

with open(SAVE_PATH, 'w') as f:
    json.dump(dataset, f)

print(f"\n✓ Saved to {SAVE_PATH}")
print(f"  File size: {os.path.getsize(SAVE_PATH)/1024:.1f} KB")

from google.colab import files
local = "/content/conspiracy_dataset.json"
with open(local, 'w') as f:
    json.dump(dataset, f)
files.download(local)
print("✓ Downloaded locally")
