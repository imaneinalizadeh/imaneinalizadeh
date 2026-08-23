"""
clickbait_dataset_builder.py
============================
Builds the Clickbait extension dataset from existing Fakeddit processed splits.

What this does:
- Takes your existing train.csv / val.csv / test.csv (Fakeddit processed)
- Identifies Clickbait-style posts using subreddit + linguistic heuristics
- Assigns them Category 7 (Clickbait) in the taxonomy
- Re-labels Group 1 fine head targets: Satire=0, False Connection=1, Clickbait=2
- Saves new CSVs ready for retraining Group 1 fine head only

Usage:
    python src/clickbait_dataset_builder.py \
        --input_dir data/processed/ \
        --output_dir data/clickbait/ \
        --verbose

Author: Iman Ein Alizadeh (s2901349)
University of Edinburgh EPCC — MSc HPC Dissertation 2026
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from collections import Counter

# ── Clickbait subreddits in Fakeddit ─────────────────────────────────────────
# These subreddits are known for sensational engagement-bait headlines
CLICKBAIT_SUBREDDITS = {
    "Uplifting_News",
    "UpliftingNews",
    "nottheonion",
    "worldnews",       # often sensational headlines
    "news",
    "interestingasfuck",
    "mildlyinteresting",
    "todayilearned",
    "YouShouldKnow",
    "LifeProTips",
}

# ── Clickbait linguistic patterns ─────────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    # Number-led ("10 reasons why...")
    r"^\d+\s+(reasons?|ways?|things?|facts?|signs?|tips?|tricks?|secrets?|examples?|times?)",
    # "You won't believe..."
    r"\byou(r|'ll|'re)?\s+(won'?t|will never|can'?t|cannot)\s+believe",
    # "This is why..."
    r"^this\s+is\s+(why|what|how)",
    # "What happens when..."
    r"^what\s+happens?\s+when",
    # "Here's why..."
    r"^here'?s?\s+(why|what|how)",
    # "The reason why..."
    r"^the\s+reason\s+(why|that)",
    # Ellipsis (trailing ...)
    r"\.\.\.\s*$",
    # All caps words (SHOCKING, BREAKING etc)
    r"\b(SHOCKING|BREAKING|VIRAL|MUST.?SEE|WATCH|INCREDIBLE|UNBELIEVABLE|AMAZING|WOW)\b",
    # Question headlines
    r"\?+\s*$",
    # "What you need to know"
    r"what\s+you\s+(need|should|must)\s+to?\s+know",
    # "Find out..."
    r"^find\s+out",
    # superlatives
    r"\b(best|worst|most|greatest|biggest|largest|highest|lowest|funniest|craziest|weirdest)\b",
]

CLICKBAIT_REGEX = re.compile(
    "|".join(CLICKBAIT_PATTERNS),
    re.IGNORECASE
)


def clickbait_score(title: str, subreddit: str = "") -> float:
    """
    Returns a score 0.0-1.0 indicating how likely a headline is clickbait.
    Uses subreddit membership + linguistic pattern matching.
    """
    if not isinstance(title, str) or len(title.strip()) < 5:
        return 0.0

    score = 0.0

    # Subreddit signal (strong)
    if subreddit in CLICKBAIT_SUBREDDITS:
        score += 0.45

    # Pattern matches
    matches = len(CLICKBAIT_REGEX.findall(title))
    score += min(matches * 0.25, 0.55)

    # Capitalisation ratio (ALL CAPS words)
    words = title.split()
    if words:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
        score += caps_ratio * 0.2

    # Exclamation marks
    score += min(title.count("!") * 0.1, 0.2)

    # Length signal (clickbait tends to be medium length, 8-20 words)
    word_count = len(words)
    if 8 <= word_count <= 20:
        score += 0.05

    return min(score, 1.0)


def is_clickbait(title: str, subreddit: str = "", threshold: float = 0.45) -> bool:
    return clickbait_score(title, subreddit) >= threshold


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add linguistic feature columns used during training."""
    df = df.copy()

    df["has_number_lead"]   = df["title"].str.match(r"^\d+\s+\w+", na=False).astype(int)
    df["has_question"]      = df["title"].str.contains(r"\?", na=False).astype(int)
    df["has_ellipsis"]      = df["title"].str.contains(r"\.\.\.", na=False).astype(int)
    df["has_exclamation"]   = df["title"].str.contains(r"!", na=False).astype(int)
    df["word_count"]        = df["title"].str.split().str.len().fillna(0).astype(int)
    df["caps_ratio"]        = df["title"].apply(
        lambda t: sum(1 for w in str(t).split() if w.isupper() and len(w) > 2) / max(len(str(t).split()), 1)
    )
    df["clickbait_score"]   = df.apply(
        lambda r: clickbait_score(str(r.get("title", "")), str(r.get("subreddit", ""))),
        axis=1
    )

    return df


def remap_group1_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap fine labels for Group 1 (Structural Deception) to include Clickbait.

    Original Group 1 fine labels:
        Satire/Parody     (category 1) -> fine_label 0
        False Connection  (category 4) -> fine_label 1

    New Group 1 fine labels (3-class):
        Satire/Parody     (category 1) -> fine_label 0
        False Connection  (category 4) -> fine_label 1
        Clickbait         (category 7) -> fine_label 2
    """
    df = df.copy()
    mapping = {1: 0, 4: 1, 7: 2}
    df["group1_fine_label"] = df["category"].map(mapping)
    return df


def build_clickbait_dataset(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.45,
    verbose: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        input_path = os.path.join(input_dir, f"{split}.csv")
        if not os.path.exists(input_path):
            print(f"  SKIP: {input_path} not found")
            continue

        if verbose:
            print(f"\nProcessing {split}.csv ...")

        df = pd.read_csv(input_path)

        # ── Step 1: Identify clickbait candidates from Group 1 posts ──────────
        # Only consider posts already in Group 1 (categories 1, 4)
        # We re-label strong clickbait posts as Category 7
        group1_mask = df["category"].isin([1, 4])
        group1_df   = df[group1_mask].copy()

        subreddit_col = "subreddit" if "subreddit" in df.columns else None

        group1_df["_cb_score"] = group1_df.apply(
            lambda r: clickbait_score(
                str(r.get("title", "")),
                str(r.get(subreddit_col, "")) if subreddit_col else ""
            ),
            axis=1
        )

        # Re-label high-scoring Group 1 posts as Clickbait (category 7)
        clickbait_mask = group1_df["_cb_score"] >= threshold
        group1_df.loc[clickbait_mask, "category"] = 7

        if verbose:
            cb_count  = clickbait_mask.sum()
            total_g1  = len(group1_df)
            print(f"  Group 1 posts:     {total_g1:,}")
            print(f"  Relabelled as CB:  {cb_count:,} ({cb_count/total_g1*100:.1f}%)")
            print(f"  Remaining dist:    {dict(Counter(group1_df['category'].tolist()))}")

        # ── Step 2: Add linguistic features ───────────────────────────────────
        group1_df = extract_features(group1_df)

        # ── Step 3: Add Group 1 fine label (0=Satire, 1=FalseConn, 2=Clickbait)
        group1_df = remap_group1_labels(group1_df)

        # Drop internal column
        group1_df = group1_df.drop(columns=["_cb_score"], errors="ignore")

        # ── Step 4: Save ──────────────────────────────────────────────────────
        out_path = os.path.join(output_dir, f"{split}_clickbait.csv")
        group1_df.to_csv(out_path, index=False)

        if verbose:
            print(f"  Saved -> {out_path}  ({len(group1_df):,} rows)")

        # ── Step 5: Also save full dataset with Category 7 relabelling ────────
        # Update the main dataframe and save
        df.loc[df.index.isin(group1_df.index[clickbait_mask]), "category"] = 7
        full_out = os.path.join(output_dir, f"{split}_full_with_clickbait.csv")
        df.to_csv(full_out, index=False)

        if verbose:
            print(f"  Saved -> {full_out}  ({len(df):,} rows, full dataset)")

    if verbose:
        print("\nDone. Files written to:", output_dir)
        print("\nGroup 1 fine label mapping:")
        print("  0 -> Satire/Parody")
        print("  1 -> False Connection")
        print("  2 -> Clickbait  (NEW)")


def demo_scoring(titles: list) -> None:
    """Print clickbait scores for a list of sample titles."""
    print("\n── Clickbait Score Demo ──────────────────────────────")
    for title in titles:
        score = clickbait_score(title)
        label = "CLICKBAIT" if score >= 0.45 else "not clickbait"
        print(f"  [{score:.2f}] {label:12s}  {title[:70]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build clickbait extension dataset from Fakeddit splits"
    )
    parser.add_argument("--input_dir",  default="data/processed/",  help="Directory with train/val/test.csv")
    parser.add_argument("--output_dir", default="data/clickbait/",   help="Output directory")
    parser.add_argument("--threshold",  type=float, default=0.45,    help="Clickbait score threshold (default 0.45)")
    parser.add_argument("--verbose",    action="store_true",          help="Print progress")
    parser.add_argument("--demo",       action="store_true",          help="Run scoring demo only")
    args = parser.parse_args()

    if args.demo:
        DEMO_TITLES = [
            "10 Reasons Why You Should Never Trust This Government Report",
            "Scientists discover water on Mars",
            "You Won't Believe What Happened to This Town After the Storm",
            "New study shows coffee linked to reduced cancer risk",
            "SHOCKING: Local man wins lottery twice in one year!!!",
            "The Real Reason Why Banks Don't Want You to Know This",
            "Federal Reserve raises interest rates by 25 basis points",
            "What Doctors Don't Want You to Know About This Common Drug",
            "UK Parliament passes new climate legislation",
            "Find Out Which Foods Are SECRETLY Destroying Your Health",
        ]
        demo_scoring(DEMO_TITLES)
    else:
        build_clickbait_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            verbose=args.verbose,
        )
