# ======================================================================
# SHONA-SENT FRAMEWORK IMPLEMENTATION
# Baseline XLM-R vs SHONA-SENT / LEXSA-XLMR
# Google Colab-ready Python script — Version 4
# ======================================================================
#
# REQUIRED INPUTS (upload all five to /content before running the .py)


# V3: Automatically strips accidental whitespace from spreadsheet column headings.
# V4: Adds runtime compatibility handling for changing Hugging Face Trainer APIs.
# 1. Annotated corpus (.xlsx/.csv)
# 2. Shona-English AFINN extension (.txt/.csv/.xlsx)
# 3. Verified Shona stop-word list (.txt/.csv/.xlsx)
# 4. Sentiment protected-word list (.txt/.csv/.xlsx)
# 5. Domain protected-word list (.txt/.csv/.xlsx)
#
# EXPECTED CORPUS COLUMNS
# Tweet_id | Created_at | Tweet | Domain | DetectedLanguage |
# Baseline Lexicon Polarity | Human Annotor 1 | Human Annotor 2 | FinalLabel
#
# CONTROLLED COMPARISON
# - Common gold standard: FinalLabel
# - One stratified 70/15/15 split, seed=42
# - Same Tweet_ids in both experiments
# - Baseline: original Tweet -> XLM-R
# - Proposed: SHONA-SENT processed Tweet -> XLM-R
# - Same XLM-R architecture/hyperparameters
# - Best checkpoint selected by validation weighted F1
# - Same isolated test set for final evaluation
#
# IMPORTANT
# V5: AFINN is used for lexicon-guided corpus/input enhancement. Raw numerical
# AFINN scores are retained for audit but are NOT supplied as numerical model features.
# ======================================================================

# ======================================================================
# 0. INSTALL DEPENDENCIES (COLAB)
# ======================================================================
import sys, subprocess, importlib.util

def ensure_packages():
    packages = {
        "transformers": "transformers",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "sklearn": "scikit-learn",
        "sentencepiece": "sentencepiece",
        "openpyxl": "openpyxl",
    }
    missing = [pip_name for module, pip_name in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])

ensure_packages()

# ======================================================================
# 1. IMPORTS
# ======================================================================
import os
import re
import json
import random
import hashlib
import shutil
import platform
import gc
import inspect
import unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sklearn
import transformers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding, set_seed
)

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ======================================================================
# 2. CONFIGURATION
# ======================================================================
MODEL_NAME = "FacebookAI/xlm-roberta-base"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

MAX_LENGTH = 128
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 2

EXPECTED_CORPUS_SIZE = 39666
OUTPUT_ROOT = Path("/content/SHONA_SENT_Experiment" if IN_COLAB else "SHONA_SENT_Experiment")
DATA_DIR = OUTPUT_ROOT / "data_splits"
RESOURCE_DIR = OUTPUT_ROOT / "resources"
BASELINE_DIR = OUTPUT_ROOT / "baseline_xlmr"
LEXSA_DIR = OUTPUT_ROOT / "lexsa_xlmr"
GUIDED_LEXSA_DIR = OUTPUT_ROOT / "guided_lexsa_xlmr"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"

for d in [OUTPUT_ROOT, DATA_DIR, RESOURCE_DIR, BASELINE_DIR, LEXSA_DIR, GUIDED_LEXSA_DIR, COMPARISON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "Tweet_id", "Created_at", "Tweet", "Domain", "DetectedLanguage",
    "Baseline Lexicon Polarity", "Human Annotor 1", "Human Annotor 2", "FinalLabel"
]

CONCEPTUAL_TO_MODEL = {-1: 1, 0: 0, 1: 2}
MODEL_TO_CONCEPTUAL = {0: 0, 1: -1, 2: 1}
ID2LABEL = {0: "Neutral", 1: "Negative", 2: "Positive"}
LABEL2ID = {"Neutral": 0, "Negative": 1, "Positive": 2}

# ======================================================================
# 3. REPRODUCIBILITY
# ======================================================================
def reset_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    set_seed(RANDOM_SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reset_seeds()

# ======================================================================
# 4. INPUT FILE DISCOVERY
# ======================================================================
# IMPORTANT FOR GOOGLE COLAB:
# A standalone .py process cannot open google.colab.files.upload().
#
# BEFORE running this script, run this in a NORMAL Colab notebook cell:
#
#     from google.colab import files
#     uploaded = files.upload()
#
# Select ALL FIVE research files at once. They will be placed in /content.
# This script then identifies and validates them automatically.
# ======================================================================

def _peek_table_columns(path):
    """Read only enough of a CSV/XLSX file to inspect its column names."""
    try:
        if path.suffix.lower() == ".xlsx":
            return list(pd.read_excel(path, nrows=3).columns)
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=3).columns)
    except Exception:
        return []
    return []

def _norm_col(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def _looks_like_afinn(path):
    """Conservative structural/name check for the AFINN extension."""
    name = path.name.lower()
    if "afinn" in name:
        return True

    if path.suffix.lower() in {".csv", ".xlsx"}:
        cols = {_norm_col(c) for c in _peek_table_columns(path)}
        word_cols = {"word", "term", "token", "shona", "phrase"}
        score_cols = {"score", "value", "polarity", "sentiment_score", "afinn_score"}
        return bool(cols & word_cols) and bool(cols & score_cols)

    if path.suffix.lower() == ".txt":
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                checked = 0
                numeric_pairs = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    checked += 1
                    parts = re.split(r"\t|,|;", line)
                    if len(parts) < 2:
                        parts = line.rsplit(maxsplit=1)
                    if len(parts) >= 2:
                        try:
                            float(parts[-1].strip())
                            numeric_pairs += 1
                        except Exception:
                            pass
                    if checked >= 10:
                        break
                return checked > 0 and numeric_pairs >= max(1, checked // 2)
        except Exception:
            return False
    return False

def _resource_name_score(path, kind):
    """Filename hints used only after the corpus and AFINN are structurally found."""
    n = path.name.lower().replace("-", "_").replace(" ", "_")

    if kind == "stopwords":
        positive = ["stopword", "stop_word", "stopwords", "stop_words"]
        negative = ["protected", "sentiment", "domain", "afinn"]
    elif kind == "sentiment":
        positive = ["sentiment_protected", "sentimentprotected",
                    "protected_sentiment", "sentiment"]
        negative = ["domain", "stopword", "stop_word", "afinn"]
    elif kind == "domain":
        positive = ["domain_protected", "domainprotected",
                    "protected_domain", "domain"]
        negative = ["sentiment", "stopword", "stop_word", "afinn"]
    else:
        return 0

    score = sum(3 for x in positive if x in n)
    score -= sum(2 for x in negative if x in n)
    if "protected" in n and kind in {"sentiment", "domain"}:
        score += 2
    return score

def _choose_named_resource(candidates, kind):
    scored = sorted(
        [(_resource_name_score(p, kind), p) for p in candidates],
        key=lambda x: (-x[0], x[1].name.lower())
    )
    if not scored or scored[0][0] <= 0:
        return None
    # Avoid silently choosing between equally plausible files.
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None
    return scored[0][1]

def discover_inputs():
    """
    Locate the five uploaded research inputs.

    Safety rules:
    - Corpus is identified by its exact required schema, not filename.
    - AFINN is identified structurally and/or by an explicit AFINN filename.
    - Stop/protected resources use conservative filename hints.
    - If identification is ambiguous, STOP rather than guess.
    """
    search_dir = Path("/content") if Path("/content").exists() else Path.cwd()

    allowed = {".xlsx", ".csv", ".txt"}
    candidates = [
        p for p in search_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in allowed
        and not p.name.startswith("SHONA_SENT_")
    ]

    print("\n" + "=" * 78)
    print("SHONA-SENT INPUT FILE DETECTION")
    print("=" * 78)
    print(f"Searching: {search_dir}")

    if not candidates:
        raise FileNotFoundError(
            "\nNo research input files were found.\n\n"
            "In a NORMAL Google Colab cell, run:\n\n"
            "    from google.colab import files\n"
            "    uploaded = files.upload()\n\n"
            "Select all five research files, then rerun this .py script."
        )

    print("\nCandidate files found:")
    for i, p in enumerate(sorted(candidates), 1):
        print(f"  {i}. {p.name}")

    # 1) Corpus: exact schema
    required_norm = {_norm_col(c) for c in REQUIRED_COLUMNS}
    corpus_matches = []
    for p in candidates:
        if p.suffix.lower() in {".xlsx", ".csv"}:
            cols = {_norm_col(c) for c in _peek_table_columns(p)}
            if required_norm.issubset(cols):
                corpus_matches.append(p)

    if len(corpus_matches) != 1:
        raise RuntimeError(
            "\nCould not uniquely identify the annotated corpus.\n"
            f"Schema matches found: {[p.name for p in corpus_matches]}\n"
            "Keep only the intended corpus among the uploaded candidate files, "
            "or ensure its columns exactly match the required corpus schema."
        )
    corpus = corpus_matches[0]

    remaining = [p for p in candidates if p != corpus]

    # 2) AFINN
    afinn_matches = [p for p in remaining if _looks_like_afinn(p)]
    # Prefer explicit AFINN filename when multiple structurally plausible resources exist.
    explicit_afinn = [p for p in afinn_matches if "afinn" in p.name.lower()]
    if len(explicit_afinn) == 1:
        afinn = explicit_afinn[0]
    elif len(afinn_matches) == 1:
        afinn = afinn_matches[0]
    else:
        raise RuntimeError(
            "\nCould not uniquely identify the Shona-English AFINN extension.\n"
            f"Possible matches: {[p.name for p in afinn_matches]}\n"
            "Rename the intended file so its filename contains 'AFINN', "
            "then rerun the script."
        )

    remaining = [p for p in remaining if p != afinn]

    # 3-5) Word resources
    stopwords = _choose_named_resource(remaining, "stopwords")
    sentiment = _choose_named_resource(remaining, "sentiment")
    domain = _choose_named_resource(remaining, "domain")

    selected = [x for x in [stopwords, sentiment, domain] if x is not None]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise RuntimeError(
            "\nThe three word-list resources could not be identified safely.\n\n"
            "Please use descriptive filenames containing these terms:\n"
            "  - shona_stopwords\n"
            "  - sentiment_protected\n"
            "  - domain_protected\n\n"
            f"Remaining candidates: {[p.name for p in remaining]}\n"
            "The script intentionally stops rather than guessing."
        )

    print("\nDetected research inputs:")
    print(f"  ✓ Annotated corpus          : {corpus.name}")
    print(f"  ✓ AFINN extension           : {afinn.name}")
    print(f"  ✓ Shona stop-word list      : {stopwords.name}")
    print(f"  ✓ Sentiment protected list  : {sentiment.name}")
    print(f"  ✓ Domain protected list     : {domain.name}")
    print("\n✓ All five inputs identified. Proceeding to full validation.")

    return corpus, afinn, stopwords, sentiment, domain

# ======================================================================
# 5. GENERIC RESOURCE LOADERS
# ======================================================================
def read_table(path):
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"{path.name} is not a table file.")

def clean_token(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x)).strip().lower()
    return x

def load_word_resource(path, resource_name):
    """
    Strict but flexible loader for a one-word/phrase resource.
    XLSX/CSV:
      - If one column exists, that column is used.
      - If multiple columns exist, a likely word column must be identifiable.
    TXT:
      - one entry per line is preferred;
      - if a line contains tab/comma separated metadata, the first field is used.
    """
    ext = path.suffix.lower()

    if ext == ".txt":
        entries = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                first = re.split(r"[\t,;]", line, maxsplit=1)[0].strip()
                if first:
                    entries.append(clean_token(first))
    else:
        table = read_table(path)
        if table.empty:
            raise ValueError(f"{resource_name} is empty.")

        likely = ["word", "term", "token", "shona", "stopword", "stop_word",
                  "protected_word", "phrase"]
        if len(table.columns) == 1:
            col = table.columns[0]
        else:
            normalized = {str(c).strip().lower().replace(" ", "_"): c for c in table.columns}
            matches = [normalized[k] for k in likely if k in normalized]
            if not matches:
                raise ValueError(
                    f"\nCannot safely identify the word column in {resource_name}.\n"
                    f"Columns found: {list(table.columns)}\n"
                    "Rename the relevant column to Word, Term, Token, Shona, "
                    "Stopword, Protected_Word, or Phrase."
                )
            col = matches[0]
        entries = [clean_token(v) for v in table[col].tolist()]

    entries = sorted({x for x in entries if x})
    if not entries:
        raise ValueError(f"No usable entries found in {resource_name}.")
    return set(entries)

def load_afinn(path):
    """
    Loads and validates the AFINN extension.
    The resource is used for audit/enhancement documentation and lexical
    validation. Its numeric scores are NOT passed to XLM-R.

    Accepted table column patterns include:
      word/term/token/shona + score/value/polarity/sentiment_score
    TXT supports common formats such as:
      word<TAB>score
      word,score
      word score
    """
    ext = path.suffix.lower()
    rows = []

    if ext == ".txt":
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\t|,|;", line)
                if len(parts) < 2:
                    parts = line.rsplit(maxsplit=1)
                if len(parts) < 2:
                    raise ValueError(
                        f"AFINN line {line_no} has no detectable score: {line!r}"
                    )
                word = clean_token(parts[0])
                try:
                    score = float(str(parts[-1]).strip())
                except ValueError:
                    # permit a single header line
                    if line_no == 1:
                        continue
                    raise ValueError(
                        f"AFINN score is not numeric on line {line_no}: {line!r}"
                    )
                rows.append((word, score))
    else:
        table = read_table(path)
        if table.empty:
            raise ValueError("AFINN extension is empty.")

        normalized = {str(c).strip().lower().replace(" ", "_"): c for c in table.columns}
        word_candidates = ["word", "term", "token", "shona", "phrase"]
        score_candidates = ["score", "value", "polarity", "sentiment_score", "afinn_score"]

        word_cols = [normalized[k] for k in word_candidates if k in normalized]
        score_cols = [normalized[k] for k in score_candidates if k in normalized]

        if not word_cols or not score_cols:
            raise ValueError(
                "\nCannot safely identify AFINN word and score columns.\n"
                f"Columns found: {list(table.columns)}\n"
                "Use a word column such as Word/Shona/Term and a numeric "
                "column such as Score/Polarity/AFINN_Score."
            )

        wcol, scol = word_cols[0], score_cols[0]
        for _, row in table[[wcol, scol]].dropna().iterrows():
            word = clean_token(row[wcol])
            try:
                score = float(row[scol])
            except Exception:
                raise ValueError(f"Non-numeric AFINN score for {word!r}: {row[scol]!r}")
            rows.append((word, score))

    if not rows:
        raise ValueError("No valid AFINN entries were loaded.")

    afinn_df = pd.DataFrame(rows, columns=["word", "score"])
    afinn_df = afinn_df[afinn_df["word"] != ""].drop_duplicates("word", keep="last")
    afinn = dict(zip(afinn_df["word"], afinn_df["score"]))
    return afinn, afinn_df

# ======================================================================
# 6. LABEL NORMALISATION
# ======================================================================
def normalize_label(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) in (-1, 0, 1) else np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        iv = int(value)
        return iv if iv in (-1, 0, 1) else np.nan

    text = str(value).strip().lower()
    mapping = {
        "-1": -1, "negative": -1, "neg": -1,
        "0": 0, "neutral": 0, "neu": 0,
        "1": 1, "+1": 1, "positive": 1, "pos": 1
    }
    return mapping.get(text, np.nan)

# ======================================================================
# 7. SHONA-SENT TEXT PREPROCESSING
# ======================================================================
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)
RT_RE = re.compile(r"^\s*RT\s+", re.I)
WHITESPACE_RE = re.compile(r"\s+")
# Preserve letters/numbers/underscore/apostrophe/hyphen/whitespace.
PUNCT_RE = re.compile(r"[^\w\s'\-]", re.UNICODE)

def basic_normalize_text(text):
    """Shared minimal normalisation: no linguistic stop-word enhancement."""
    text = unicodedata.normalize("NFKC", str(text))
    text = RT_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    # Preserve hashtag text, remove only '#'
    text = text.replace("#", "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def shona_sent_preprocess(text, effective_stopwords):
    """
    Proposed SHONA-SENT preparation.
    Protected terms have already been removed from effective_stopwords.
    """
    text = basic_normalize_text(text).lower()
    text = PUNCT_RE.sub(" ", text)
    tokens = [t for t in WHITESPACE_RE.sub(" ", text).strip().split(" ") if t]
    kept = [t for t in tokens if t not in effective_stopwords]
    return " ".join(kept)

def lexicon_enhance_text(text, afinn):
    """
    V5 LEXSA enhancement:
    Preserve the minimally normalised original tweet and append lexicon-derived
    sentiment evidence from the Shona-English AFINN extension.

    The enhancement is categorical/textual rather than a raw numerical feature:
      [LEX_POS]   positive lexicon evidence only
      [LEX_NEG]   negative lexicon evidence only
      [LEX_MIXED] both positive and negative evidence
      [LEX_NONE]  no AFINN match

    Matched sentiment-bearing lexicon terms are appended after their polarity
    markers. This makes AFINN materially affect the proposed XLM-R input while
    preserving the original contextual sentence.
    """
    base = basic_normalize_text(text)
    normalized = PUNCT_RE.sub(" ", base.lower())
    tokens = [t for t in WHITESPACE_RE.sub(" ", normalized).strip().split(" ") if t]

    positive_terms = []
    negative_terms = []
    for token in tokens:
        if token in afinn:
            score = afinn[token]
            if score > 0 and token not in positive_terms:
                positive_terms.append(token)
            elif score < 0 and token not in negative_terms:
                negative_terms.append(token)

    if positive_terms and negative_terms:
        evidence = ["[LEX_MIXED]", "[LEX_POS]"] + positive_terms + ["[LEX_NEG]"] + negative_terms
    elif positive_terms:
        evidence = ["[LEX_POS]"] + positive_terms
    elif negative_terms:
        evidence = ["[LEX_NEG]"] + negative_terms
    else:
        evidence = ["[LEX_NONE]"]

    return (base + " " + " ".join(evidence)).strip()


def guided_lexicon_enhance_text(
    text,
    afinn,
    verified_stopwords,
    sentiment_protected,
    domain_protected
):
    """
    V6 FULL SHONA-SENT / LEXSA-XLMR enhancement.

    Preserve the complete minimally-normalised tweet. AFINN-derived evidence is
    appended only after linguistic guidance:

      1. Detect AFINN-matched sentiment terms.
      2. If a matched term is a verified Shona stop word, suppress it from the
         appended evidence UNLESS it is protected.
      3. Sentiment-protected and domain-protected terms take precedence.
      4. Never delete these words from the original tweet itself.

    This tests whether the verified Shona stop-word and protected-word resources
    improve the QUALITY of AFINN evidence supplied textually to XLM-R.
    """
    base = basic_normalize_text(text)
    normalized = PUNCT_RE.sub(" ", base.lower())
    tokens = [t for t in WHITESPACE_RE.sub(" ", normalized).strip().split(" ") if t]

    protected = set(sentiment_protected) | set(domain_protected)
    positive_terms = []
    negative_terms = []

    for token in tokens:
        if token not in afinn:
            continue

        # Stop words are filtered only from appended lexical evidence.
        # Protection always overrides stop-word filtering.
        if token in verified_stopwords and token not in protected:
            continue

        score = afinn[token]
        if score > 0 and token not in positive_terms:
            positive_terms.append(token)
        elif score < 0 and token not in negative_terms:
            negative_terms.append(token)

    if positive_terms and negative_terms:
        evidence = ["[LEX_MIXED]", "[LEX_POS]"] + positive_terms + ["[LEX_NEG]"] + negative_terms
    elif positive_terms:
        evidence = ["[LEX_POS]"] + positive_terms
    elif negative_terms:
        evidence = ["[LEX_NEG]"] + negative_terms
    else:
        evidence = ["[LEX_NONE]"]

    return (base + " " + " ".join(evidence)).strip()

def lexical_audit(text, afinn):
    """
    Audit of AFINN coverage. Raw numerical values are saved for analysis and
    reproducibility but are not passed as separate numerical XLM-R features.
    """
    normalized = basic_normalize_text(text).lower()
    normalized = PUNCT_RE.sub(" ", normalized)
    tokens = [t for t in normalized.split() if t]
    hits = [(t, afinn[t]) for t in tokens if t in afinn]
    score = float(sum(s for _, s in hits))
    positive = sum(1 for _, s in hits if s > 0)
    negative = sum(1 for _, s in hits if s < 0)
    coverage = len(hits) / len(tokens) if tokens else 0.0
    return score, positive, negative, len(hits), coverage

# ======================================================================
# 8. HASHING / SAVE HELPERS
# ======================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def save_df_both(df, stem):
    csv_path = Path(str(stem) + ".csv")
    xlsx_path = Path(str(stem) + ".xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path

# ======================================================================
# 9. METRICS / FIGURES
# ======================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    pw, rw, fw, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_weighted": pw,
        "recall_weighted": rw,
        "f1_weighted": fw,
        "precision_macro": pm,
        "recall_macro": rm,
        "f1_macro": fm
    }

def save_confusion_matrix(y_true, y_pred, outdir, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    df_cm = pd.DataFrame(
        cm,
        index=["Actual Neutral", "Actual Negative", "Actual Positive"],
        columns=["Predicted Neutral", "Predicted Negative", "Predicted Positive"]
    )
    save_df_both(df_cm.reset_index(names="Actual"), outdir / "confusion_matrix")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1, 2], ["Neutral", "Negative", "Positive"])
    ax.set_yticks([0, 1, 2], ["Neutral", "Negative", "Positive"])
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Actual Sentiment")
    ax.set_title(f"{title} Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def save_training_history(trainer, outdir, title):
    hist = pd.DataFrame(trainer.state.log_history)
    save_df_both(hist, outdir / "training_history")

    if "eval_f1_weighted" in hist.columns:
        d = hist.dropna(subset=["eval_f1_weighted"])
        if not d.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(d["epoch"], d["eval_f1_weighted"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted F1-Score")
            plt.title(f"{title} Validation Weighted F1")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "validation_f1_history.png",
                        dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

# ======================================================================
# 10. MAIN
# ======================================================================
def main():
    print("=" * 78)
    print("SHONA-SENT FRAMEWORK — V6 FULL GUIDED LEXICON-ENHANCED EXPERIMENT")
    print("=" * 78)
    print("Gold standard: FinalLabel")
    print("Baseline: Original Tweet -> XLM-R")
    print("V5 arm: Original Tweet + AFINN Lexicon Enhancement -> XLM-R\nV6 arm: Original Tweet + AFINN + Stop/Protected Guidance -> XLM-R")
    print("AFINN lexicon evidence IS supplied textually to LEXSA-XLMR; raw scores are audit-only.")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    corpus_file, afinn_file, stopword_file, sentiment_file, domain_file = discover_inputs()

    # Copy original research resources into results package
    for src in [corpus_file, afinn_file, stopword_file, sentiment_file, domain_file]:
        shutil.copy2(src, RESOURCE_DIR / src.name)

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------
    corpus = read_table(corpus_file)

    # Normalize spreadsheet column headings before exact schema validation.
    # This removes accidental leading/trailing whitespace (e.g. "Tweet ")
    # without changing any research data or intended column names.
    original_columns = list(corpus.columns)
    corpus.columns = [str(c).strip() for c in corpus.columns]

    renamed_columns = [
        (old_col, new_col)
        for old_col, new_col in zip(original_columns, corpus.columns)
        if str(old_col) != str(new_col)
    ]

    print(f"\nCorpus loaded: {len(corpus):,} rows")
    print("Columns:", list(corpus.columns))

    if renamed_columns:
        print("\nColumn-heading whitespace normalized:")
        for old_col, new_col in renamed_columns:
            print(f"  ✓ {old_col!r} -> {new_col!r}")

    # Guard against two different source headings becoming identical after stripping.
    if len(set(corpus.columns)) != len(corpus.columns):
        duplicates = pd.Series(corpus.columns)[pd.Series(corpus.columns).duplicated()].tolist()
        raise ValueError(
            "Duplicate column names exist after whitespace normalization: "
            f"{duplicates}. Please correct the spreadsheet headings."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in corpus.columns]
    if missing:
        raise ValueError(
            f"\nCorpus schema validation failed.\nMissing: {missing}\n"
            f"Actual columns: {list(corpus.columns)}"
        )

    if len(corpus) != EXPECTED_CORPUS_SIZE:
        print(f"WARNING: Expected {EXPECTED_CORPUS_SIZE:,} rows; found {len(corpus):,}.")
    else:
        print(f"✓ Expected corpus size confirmed: {len(corpus):,}")

    if corpus["Tweet_id"].isna().any():
        raise ValueError("Tweet_id contains missing values.")
    if corpus["Tweet_id"].duplicated().any():
        n = int(corpus["Tweet_id"].duplicated().sum())
        raise ValueError(f"Duplicate Tweet_id values detected: {n}")

    corpus = corpus.dropna(subset=["Tweet", "FinalLabel"]).copy()
    corpus["Tweet"] = corpus["Tweet"].astype(str).str.strip()
    corpus = corpus[corpus["Tweet"] != ""].copy()

    corpus["ConceptualLabel"] = corpus["FinalLabel"].apply(normalize_label)
    bad = int(corpus["ConceptualLabel"].isna().sum())
    if bad:
        bad_values = corpus.loc[corpus["ConceptualLabel"].isna(), "FinalLabel"].value_counts().head(20)
        raise ValueError(
            f"{bad} FinalLabel values could not be interpreted.\nExamples:\n{bad_values}"
        )

    corpus["ConceptualLabel"] = corpus["ConceptualLabel"].astype(int)
    corpus["ModelLabel"] = corpus["ConceptualLabel"].map(CONCEPTUAL_TO_MODEL).astype(int)

    # Annotation distributions retained for thesis/audit
    annotation_summary = []
    for col in ["Baseline Lexicon Polarity", "Human Annotor 1", "Human Annotor 2", "FinalLabel"]:
        vc = corpus[col].value_counts(dropna=False)
        for label, count in vc.items():
            annotation_summary.append({
                "Annotation_Source": col, "Label": str(label), "Count": int(count),
                "Percentage": float(count / len(corpus) * 100)
            })
    save_df_both(pd.DataFrame(annotation_summary), COMPARISON_DIR / "annotation_distributions")

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------
    print("\nLoading research resources...")
    afinn, afinn_df = load_afinn(afinn_file)
    stopwords = load_word_resource(stopword_file, "Verified Shona stop-word list")
    sentiment_protected = load_word_resource(sentiment_file, "Sentiment protected-word list")
    domain_protected = load_word_resource(domain_file, "Domain protected-word list")

    protected_words = sentiment_protected | domain_protected
    protected_overlap = stopwords & protected_words
    effective_stopwords = stopwords - protected_words  # PROTECTION WINS

    print(f"AFINN entries             : {len(afinn):,}")
    print(f"Verified stop words       : {len(stopwords):,}")
    print(f"Sentiment protected words : {len(sentiment_protected):,}")
    print(f"Domain protected words    : {len(domain_protected):,}")
    print(f"Protected union           : {len(protected_words):,}")
    print(f"Stop/protected overlap    : {len(protected_overlap):,}")
    print(f"Effective stop words      : {len(effective_stopwords):,}")
    print("✓ Protection precedence applied.")

    # Save normalized resource inventories
    save_df_both(afinn_df.sort_values("word"), RESOURCE_DIR / "normalized_afinn_extension")
    save_df_both(pd.DataFrame({"word": sorted(stopwords)}), RESOURCE_DIR / "verified_stopwords")
    save_df_both(pd.DataFrame({"word": sorted(sentiment_protected)}), RESOURCE_DIR / "sentiment_protected_words")
    save_df_both(pd.DataFrame({"word": sorted(domain_protected)}), RESOURCE_DIR / "domain_protected_words")
    save_df_both(pd.DataFrame({"word": sorted(protected_words)}), RESOURCE_DIR / "all_protected_words")
    save_df_both(pd.DataFrame({"word": sorted(protected_overlap)}), RESOURCE_DIR / "stopword_protected_overlap")
    save_df_both(pd.DataFrame({"word": sorted(effective_stopwords)}), RESOURCE_DIR / "effective_stopwords")

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------
    print("\nPreparing baseline and SHONA-SENT text...")
    corpus["OriginalTweet"] = corpus["Tweet"].astype(str)
    corpus["BaselineText"] = corpus["OriginalTweet"].apply(basic_normalize_text)
    # Keep the earlier stop-word processed version as an ABLATION artefact only.
    corpus["ProcessedTweet"] = corpus["OriginalTweet"].apply(
        lambda x: shona_sent_preprocess(x, effective_stopwords)
    )

    # V5 MAIN PROPOSED INPUT: original context + AFINN-derived lexical evidence.
    corpus["LexiconEnhancedText"] = corpus["OriginalTweet"].apply(
        lambda x: lexicon_enhance_text(x, afinn)
    )

    # V6 FULL GUIDED INPUT: AFINN evidence filtered by the verified Shona
    # stop-word list, with sentiment/domain protected lists taking precedence.
    # The original tweet remains intact.
    corpus["GuidedLexiconEnhancedText"] = corpus["OriginalTweet"].apply(
        lambda x: guided_lexicon_enhance_text(
            x,
            afinn,
            stopwords,
            sentiment_protected,
            domain_protected
        )
    )

    # Never silently train on empty text.
    empty_processed = corpus["ProcessedTweet"].str.strip().eq("")
    if empty_processed.any():
        n = int(empty_processed.sum())
        print(f"WARNING: {n} stop-word processed tweets became empty; reverting those rows to BaselineText.")
        corpus.loc[empty_processed, "ProcessedTweet"] = corpus.loc[empty_processed, "BaselineText"]

    empty_enhanced = corpus["LexiconEnhancedText"].str.strip().eq("")
    if empty_enhanced.any():
        n = int(empty_enhanced.sum())
        print(f"WARNING: {n} lexicon-enhanced tweets became empty; reverting those rows to BaselineText.")
        corpus.loc[empty_enhanced, "LexiconEnhancedText"] = corpus.loc[empty_enhanced, "BaselineText"]

    empty_guided = corpus["GuidedLexiconEnhancedText"].str.strip().eq("")
    if empty_guided.any():
        n = int(empty_guided.sum())
        print(f"WARNING: {n} guided lexicon-enhanced tweets became empty; reverting those rows to BaselineText.")
        corpus.loc[empty_guided, "GuidedLexiconEnhancedText"] = corpus.loc[empty_guided, "BaselineText"]

    # AFINN audit accompanying the actual lexicon enhancement
    audits = corpus["OriginalTweet"].apply(lambda x: lexical_audit(x, afinn))
    corpus["AFINN_AuditScore"] = audits.apply(lambda x: x[0])
    corpus["AFINN_PositiveHits"] = audits.apply(lambda x: x[1])
    corpus["AFINN_NegativeHits"] = audits.apply(lambda x: x[2])
    corpus["AFINN_LexiconHits"] = audits.apply(lambda x: x[3])
    corpus["AFINN_LexicalCoverage"] = audits.apply(lambda x: x[4])

    corpus["AFINN_EvidenceClass"] = corpus.apply(
        lambda r: (
            "Mixed" if r["AFINN_PositiveHits"] > 0 and r["AFINN_NegativeHits"] > 0
            else "Positive" if r["AFINN_PositiveHits"] > 0
            else "Negative" if r["AFINN_NegativeHits"] > 0
            else "NoHit"
        ),
        axis=1
    )
    print("\nAFINN enhancement coverage:")
    print(corpus["AFINN_EvidenceClass"].value_counts(dropna=False).to_string())
    print(f"Tweets with >=1 AFINN hit: {(corpus['AFINN_LexiconHits'] > 0).sum():,} / {len(corpus):,}")

    def _guided_evidence_class(s):
        s = str(s)
        if "[LEX_MIXED]" in s:
            return "Mixed"
        if "[LEX_POS]" in s:
            return "Positive"
        if "[LEX_NEG]" in s:
            return "Negative"
        return "NoHit"

    corpus["Guided_AFINN_EvidenceClass"] = corpus["GuidedLexiconEnhancedText"].apply(
        _guided_evidence_class
    )

    print("\nV6 guided AFINN enhancement coverage:")
    print(corpus["Guided_AFINN_EvidenceClass"].value_counts(dropna=False).to_string())
    print(
        "Tweets with guided lexicon evidence: "
        f"{(corpus['Guided_AFINN_EvidenceClass'] != 'NoHit').sum():,} / {len(corpus):,}"
    )

    # Preprocessing audit
    prep_summary = pd.DataFrame([{
        "Corpus_Rows": len(corpus),
        "Original_Empty": int(corpus["OriginalTweet"].str.strip().eq("").sum()),
        "Processed_Initially_Empty": int(empty_processed.sum()),
        "Verified_Stopwords": len(stopwords),
        "Sentiment_Protected": len(sentiment_protected),
        "Domain_Protected": len(domain_protected),
        "Protected_Union": len(protected_words),
        "Stopword_Protected_Overlap": len(protected_overlap),
        "Effective_Stopwords": len(effective_stopwords),
        "AFINN_Entries": len(afinn),
        "AFINN_Direct_Model_Input": True,
        "AFINN_Input_Mode": "categorical textual lexicon evidence; raw numeric scores not supplied as numerical features",
        "V6_Guided_Enhancement": "AFINN evidence filtered by verified stop words; sentiment/domain protected terms override filtering; original tweet preserved"
    }])
    save_df_both(prep_summary, COMPARISON_DIR / "preprocessing_summary")

    # ------------------------------------------------------------------
    # ONE stratified split
    # ------------------------------------------------------------------
    print("\nCreating ONE stratified 70/15/15 split...")
    train_df, temp_df = train_test_split(
        corpus,
        test_size=(VALIDATION_RATIO + TEST_RATIO),
        stratify=corpus["ConceptualLabel"],
        random_state=RANDOM_SEED
    )
    relative_test = TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["ConceptualLabel"],
        random_state=RANDOM_SEED
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Guard against leakage
    train_ids = set(train_df["Tweet_id"].astype(str))
    val_ids = set(validation_df["Tweet_id"].astype(str))
    test_ids = set(test_df["Tweet_id"].astype(str))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == len(corpus)

    print(f"Training   : {len(train_df):,} ({len(train_df)/len(corpus)*100:.2f}%)")
    print(f"Validation : {len(validation_df):,} ({len(validation_df)/len(corpus)*100:.2f}%)")
    print(f"Testing    : {len(test_df):,} ({len(test_df)/len(corpus)*100:.2f}%)")

    # Save exact partitions
    train_csv, _ = save_df_both(train_df, DATA_DIR / "train_split")
    val_csv, _ = save_df_both(validation_df, DATA_DIR / "validation_split")
    test_csv, _ = save_df_both(test_df, DATA_DIR / "test_split")

    manifest = pd.concat([
        train_df[["Tweet_id"]].assign(Split="train"),
        validation_df[["Tweet_id"]].assign(Split="validation"),
        test_df[["Tweet_id"]].assign(Split="test")
    ], ignore_index=True)
    manifest_csv, _ = save_df_both(manifest, DATA_DIR / "split_manifest")

    split_summary = []
    for name, part in [("Training", train_df), ("Validation", validation_df), ("Testing", test_df)]:
        vc = part["ConceptualLabel"].value_counts()
        split_summary.append({
            "Split": name,
            "Total": len(part),
            "Negative": int(vc.get(-1, 0)),
            "Neutral": int(vc.get(0, 0)),
            "Positive": int(vc.get(1, 0)),
            "Percentage_of_Corpus": len(part) / len(corpus) * 100
        })
    split_summary_df = pd.DataFrame(split_summary)
    save_df_both(split_summary_df, DATA_DIR / "split_summary")
    print("\nSplit summary:")
    print(split_summary_df.to_string(index=False))

    hashes = {
        "train_split_sha256": sha256_file(train_csv),
        "validation_split_sha256": sha256_file(val_csv),
        "test_split_sha256": sha256_file(test_csv),
        "split_manifest_sha256": sha256_file(manifest_csv),
    }
    with open(DATA_DIR / "split_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    # ------------------------------------------------------------------
    # Environment/config
    # ------------------------------------------------------------------
    environment = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    with open(COMPARISON_DIR / "software_environment.json", "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)

    config = {
        "framework": "SHONA-SENT",
        "proposed_model": "LEXSA-XLMR",
        "baseline_model": "XLM-R",
        "pretrained_model": MODEL_NAME,
        "gold_standard": "FinalLabel",
        "corpus_rows_used": len(corpus),
        "split": {"train": TRAIN_RATIO, "validation": VALIDATION_RATIO, "test": TEST_RATIO},
        "stratified_on": "FinalLabel / ConceptualLabel",
        "random_seed": RANDOM_SEED,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio_requested": WARMUP_RATIO,
        "transformers_version": transformers.__version__,
        "trainer_api_compatibility_detection": True,
        "optimizer": "AdamW",
        "best_checkpoint_metric": "weighted_f1",
        "same_tweet_id_partitions": True,
        "baseline_text": "BaselineText (minimal normalization of OriginalTweet)",
        "proposed_text": "LexiconEnhancedText (original context + Shona-English AFINN evidence)",
        "protected_words_override_stopwords": True,
        "afinn_scores_direct_model_input": False,
        "conceptual_labels": {"Negative": -1, "Neutral": 0, "Positive": 1},
        "model_labels": {"Neutral": 0, "Negative": 1, "Positive": 2}
    }
    with open(COMPARISON_DIR / "experiment_configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # XLM-R tokenizer
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\nWARNING: GPU not detected. XLM-R training will be very slow.")
        print("In Colab: Runtime > Change runtime type > GPU.")

    print("\nLoading XLM-R tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def make_dataset(frame, text_col):
        tmp = frame[["Tweet_id", text_col, "ModelLabel"]].copy()
        tmp = tmp.rename(columns={text_col: "text", "ModelLabel": "labels"})
        tmp["labels"] = tmp["labels"].astype(int)
        ds = Dataset.from_pandas(tmp, preserve_index=False)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH
            )
        return ds.map(tokenize, batched=True)

    # ------------------------------------------------------------------
    # Training/evaluation
    # ------------------------------------------------------------------
    def train_experiment(name, text_col, outdir):
        reset_seeds()
        outdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 78)
        print(name.upper())
        print("=" * 78)
        print("Text column:", text_col)
        print("Gold standard: FinalLabel")

        train_ds = make_dataset(train_df, text_col)
        val_ds = make_dataset(validation_df, text_col)
        test_ds = make_dataset(test_df, text_col)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        # Tokenizer contains V5 lexicon markers. Resize both models identically;
        # baseline never sees the markers, while LEXSA-XLMR does.
        model.resize_token_embeddings(len(tokenizer))

        # --------------------------------------------------------------
        # Hugging Face API compatibility layer
        # --------------------------------------------------------------
        # TrainingArguments has changed across Transformers releases.
        # We inspect the installed signature and use only supported names.
        ta_params = inspect.signature(TrainingArguments.__init__).parameters

        training_kwargs = {
            "output_dir": str(outdir / "checkpoints"),
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": EVAL_BATCH_SIZE,
            "num_train_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_weighted",
            "greater_is_better": True,
            "save_total_limit": 2,
            "seed": RANDOM_SEED,
            "data_seed": RANDOM_SEED,
            "fp16": torch.cuda.is_available(),
            "report_to": "none",
        }

        # Evaluation strategy was historically named evaluation_strategy.
        if "eval_strategy" in ta_params:
            training_kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in ta_params:
            training_kwargs["evaluation_strategy"] = "epoch"
        else:
            raise RuntimeError(
                "Installed Transformers TrainingArguments supports neither "
                "'eval_strategy' nor 'evaluation_strategy'."
            )

        # Warm-up API compatibility:
        # Current releases accept a fractional warmup_steps value in [0,1)
        # as a proportion of total steps. Older releases expose warmup_ratio.
        if "warmup_ratio" in ta_params:
            training_kwargs["warmup_ratio"] = WARMUP_RATIO
            warmup_api_used = f"warmup_ratio={WARMUP_RATIO}"
        elif "warmup_steps" in ta_params:
            training_kwargs["warmup_steps"] = WARMUP_RATIO
            warmup_api_used = f"warmup_steps={WARMUP_RATIO} (fractional ratio)"
        else:
            raise RuntimeError(
                "Installed Transformers TrainingArguments exposes no supported "
                "warm-up parameter."
            )

        # Keep only parameters genuinely accepted by the installed version.
        unsupported = [k for k in training_kwargs if k not in ta_params]
        for key in unsupported:
            print(f"Compatibility note: omitting unsupported TrainingArguments parameter: {key}")
            training_kwargs.pop(key)

        print(f"Transformers version : {transformers.__version__}")
        print(f"Evaluation API       : "
              f"{'eval_strategy' if 'eval_strategy' in training_kwargs else 'evaluation_strategy'}")
        print(f"Warm-up API          : {warmup_api_used}")

        args = TrainingArguments(**training_kwargs)

        # Trainer also changed tokenizer -> processing_class in newer releases.
        trainer_params = inspect.signature(Trainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": val_ds,
            "data_collator": collator,
            "compute_metrics": compute_metrics,
            "callbacks": [EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )],
        }

        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        trainer.train()
        trainer.save_model(outdir / "model")
        tokenizer.save_pretrained(outdir / "model")
        save_training_history(trainer, outdir, name)

        result = trainer.predict(test_ds)
        logits = result.predictions
        y_true = result.label_ids
        y_pred = np.argmax(logits, axis=-1)

        metrics = compute_metrics((logits, y_true))
        metrics["model"] = name
        metrics["best_validation_f1"] = (
            float(trainer.state.best_metric) if trainer.state.best_metric is not None else None
        )
        metrics["best_checkpoint"] = trainer.state.best_model_checkpoint

        with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=["Neutral", "Negative", "Positive"],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose().reset_index(names="Class")
        save_df_both(report_df, outdir / "classification_report")
        save_confusion_matrix(y_true, y_pred, outdir, name)

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_df = test_df[
            ["Tweet_id", "Created_at", "Tweet", "Domain", "DetectedLanguage",
             "FinalLabel", "ConceptualLabel", "BaselineText", "ProcessedTweet", "LexiconEnhancedText", "GuidedLexiconEnhancedText"]
        ].copy()
        pred_df["ActualModelLabel"] = y_true
        pred_df["PredictedModelLabel"] = y_pred
        pred_df["ActualSentiment"] = [ID2LABEL[int(x)] for x in y_true]
        pred_df["PredictedSentiment"] = [ID2LABEL[int(x)] for x in y_pred]
        pred_df["ActualConceptualLabel"] = [MODEL_TO_CONCEPTUAL[int(x)] for x in y_true]
        pred_df["PredictedConceptualLabel"] = [MODEL_TO_CONCEPTUAL[int(x)] for x in y_pred]
        pred_df["Correct"] = y_true == y_pred
        pred_df["ProbabilityNeutral"] = probs[:, 0]
        pred_df["ProbabilityNegative"] = probs[:, 1]
        pred_df["ProbabilityPositive"] = probs[:, 2]
        save_df_both(pred_df, outdir / "predictions")

        print("\nIndependent test results")
        print(f"Accuracy           : {metrics['accuracy']:.4f}")
        print(f"Weighted Precision : {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall    : {metrics['recall_weighted']:.4f}")
        print(f"Weighted F1        : {metrics['f1_weighted']:.4f}")
        print(f"Macro F1           : {metrics['f1_macro']:.4f}")
        print("Best checkpoint    :", trainer.state.best_model_checkpoint)

        # Return plain data; release model later
        return trainer, metrics, pred_df

    # EXPERIMENT 1: baseline
    baseline_trainer, baseline_metrics, baseline_predictions = train_experiment(
        "Baseline XLM-R", "BaselineText", BASELINE_DIR
    )
    del baseline_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # EXPERIMENT 2: proposed
    lexsa_trainer, lexsa_metrics, lexsa_predictions = train_experiment(
        "SHONA-SENT / LEXSA-XLMR", "LexiconEnhancedText", LEXSA_DIR
    )
    del lexsa_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # EXPERIMENT 3: full guided LEXSA-XLMR
    guided_lexsa_trainer, guided_lexsa_metrics, guided_lexsa_predictions = train_experiment(
        "FULL SHONA-SENT / GUIDED LEXSA-XLMR",
        "GuidedLexiconEnhancedText",
        GUIDED_LEXSA_DIR
    )
    del guided_lexsa_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Three-arm controlled comparison
    # ------------------------------------------------------------------
    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy", "Weighted Precision", "Weighted Recall",
            "Weighted F1", "Macro Precision", "Macro Recall", "Macro F1"
        ],
        "Baseline XLM-R": [
            baseline_metrics["accuracy"],
            baseline_metrics["precision_weighted"],
            baseline_metrics["recall_weighted"],
            baseline_metrics["f1_weighted"],
            baseline_metrics["precision_macro"],
            baseline_metrics["recall_macro"],
            baseline_metrics["f1_macro"],
        ],
        "AFINN-only LEXSA-XLMR": [
            lexsa_metrics["accuracy"],
            lexsa_metrics["precision_weighted"],
            lexsa_metrics["recall_weighted"],
            lexsa_metrics["f1_weighted"],
            lexsa_metrics["precision_macro"],
            lexsa_metrics["recall_macro"],
            lexsa_metrics["f1_macro"],
        ],
        "Full Guided LEXSA-XLMR": [
            guided_lexsa_metrics["accuracy"],
            guided_lexsa_metrics["precision_weighted"],
            guided_lexsa_metrics["recall_weighted"],
            guided_lexsa_metrics["f1_weighted"],
            guided_lexsa_metrics["precision_macro"],
            guided_lexsa_metrics["recall_macro"],
            guided_lexsa_metrics["f1_macro"],
        ]
    })
    comparison["AFINN_vs_Baseline"] = (
        comparison["AFINN-only LEXSA-XLMR"] - comparison["Baseline XLM-R"]
    )
    comparison["Guided_vs_Baseline"] = (
        comparison["Full Guided LEXSA-XLMR"] - comparison["Baseline XLM-R"]
    )
    comparison["Guided_vs_AFINN"] = (
        comparison["Full Guided LEXSA-XLMR"] - comparison["AFINN-only LEXSA-XLMR"]
    )
    comparison["AFINN_vs_Baseline_pp"] = comparison["AFINN_vs_Baseline"] * 100
    comparison["Guided_vs_Baseline_pp"] = comparison["Guided_vs_Baseline"] * 100
    comparison["Guided_vs_AFINN_pp"] = comparison["Guided_vs_AFINN"] * 100
    save_df_both(comparison, COMPARISON_DIR / "V6_three_arm_model_comparison")

    print("\n" + "=" * 78)
    print("V6 THREE-ARM CONTROLLED MODEL COMPARISON")
    print("=" * 78)
    print(comparison.round(4).to_string(index=False))

    # Three-arm comparison plot
    plot_df = comparison[comparison["Metric"].isin(
        ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
    )]
    x = np.arange(len(plot_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, plot_df["Baseline XLM-R"], width, label="Baseline XLM-R")
    ax.bar(x, plot_df["AFINN-only LEXSA-XLMR"], width, label="AFINN-only LEXSA-XLMR")
    ax.bar(x + width, plot_df["Full Guided LEXSA-XLMR"], width, label="Full Guided LEXSA-XLMR")
    ax.set_ylabel("Performance Score")
    ax.set_xlabel("Evaluation Metric")
    ax.set_title("Three-Arm Controlled XLM-R Comparison")
    ax.set_xticks(x, plot_df["Metric"])
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "V6_three_arm_performance_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Paired predictions for the identical test Tweet_ids across all three arms
    bp = baseline_predictions[
        ["Tweet_id", "PredictedSentiment", "PredictedConceptualLabel", "Correct"]
    ].rename(columns={
        "PredictedSentiment": "BaselinePrediction",
        "PredictedConceptualLabel": "BaselinePredictionLabel",
        "Correct": "BaselineCorrect"
    })
    lp = lexsa_predictions[
        ["Tweet_id", "ActualSentiment", "ActualConceptualLabel",
         "PredictedSentiment", "PredictedConceptualLabel", "Correct"]
    ].rename(columns={
        "ActualSentiment": "GoldStandardSentiment",
        "ActualConceptualLabel": "GoldStandardLabel",
        "PredictedSentiment": "AFINNPrediction",
        "PredictedConceptualLabel": "AFINNPredictionLabel",
        "Correct": "AFINNCorrect"
    })
    gp = guided_lexsa_predictions[
        ["Tweet_id", "PredictedSentiment", "PredictedConceptualLabel", "Correct"]
    ].rename(columns={
        "PredictedSentiment": "GuidedPrediction",
        "PredictedConceptualLabel": "GuidedPredictionLabel",
        "Correct": "GuidedCorrect"
    })
    paired = bp.merge(lp, on="Tweet_id", how="inner", validate="one_to_one")
    paired = paired.merge(gp, on="Tweet_id", how="inner", validate="one_to_one")
    save_df_both(paired, COMPARISON_DIR / "V6_test_prediction_comparison")

    final_summary = {
        "framework": "SHONA-SENT",
        "baseline": "XLM-R using BaselineText",
        "afinn_only": "LEXSA-XLMR using original tweet context plus Shona-English AFINN textual evidence",
        "full_guided": "LEXSA-XLMR using original tweet context plus AFINN evidence guided by verified Shona stop words and protected lists",
        "gold_standard": "FinalLabel",
        "corpus_rows": len(corpus),
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "random_seed": RANDOM_SEED,
        "same_test_tweet_ids": True,
        "original_context_preserved_in_all_arms": True,
        "stopwords_removed_from_original_guided_tweet": False,
        "afinn_scores_direct_model_input": False,
        "baseline_accuracy": baseline_metrics["accuracy"],
        "afinn_accuracy": lexsa_metrics["accuracy"],
        "guided_accuracy": guided_lexsa_metrics["accuracy"],
        "baseline_weighted_f1": baseline_metrics["f1_weighted"],
        "afinn_weighted_f1": lexsa_metrics["f1_weighted"],
        "guided_weighted_f1": guided_lexsa_metrics["f1_weighted"],
        "afinn_vs_baseline_weighted_f1": lexsa_metrics["f1_weighted"] - baseline_metrics["f1_weighted"],
        "guided_vs_baseline_weighted_f1": guided_lexsa_metrics["f1_weighted"] - baseline_metrics["f1_weighted"],
        "guided_vs_afinn_weighted_f1": guided_lexsa_metrics["f1_weighted"] - lexsa_metrics["f1_weighted"],
    }
    with open(COMPARISON_DIR / "V6_final_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    # Hash normalized research resources as well
    resource_hashes = {}
    for p in RESOURCE_DIR.iterdir():
        if p.is_file():
            resource_hashes[p.name] = sha256_file(p)
    with open(RESOURCE_DIR / "resource_hashes.json", "w", encoding="utf-8") as f:
        json.dump(resource_hashes, f, indent=2)

    
    # ------------------------------------------------------------------
    # ZIP everything after all V6 outputs have been written
    # ------------------------------------------------------------------
    archive = shutil.make_archive(
        str(OUTPUT_ROOT.parent / "SHONA_SENT_Experiment_Results"),
        "zip",
        OUTPUT_ROOT
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETED")
    print("=" * 78)
    print("Results folder:", OUTPUT_ROOT)
    print("ZIP archive   :", archive)
    print("\nSaved reproducibility artefacts include:")
    print("- Original uploaded research resources")
    print("- Normalized resource inventories")
    print("- Protected/stop-word overlap and effective stop-word list")
    print("- Original, baseline and SHONA-SENT processed text")
    print("- Exact train/validation/test CSV and XLSX files")
    print("- Tweet_id split manifest and SHA-256 hashes")
    print("- Experiment configuration and software environment")
    print("- All three trained models and best checkpoints")
    print("- Training histories and validation-F1 figures")
    print("- Test predictions and probabilities")
    print("- Classification reports and confusion matrices")
    print("- Three-arm Baseline vs AFINN-only vs Full Guided performance comparison")
    print("- Final ZIP reproducibility package")

    if IN_COLAB:
        print("\nThe ZIP is ready in the Colab Files pane.")
        print("To download automatically, run:")
        print("from google.colab import files")
        print(f"files.download({archive!r})")

# ======================================================================
# 11. RUN
# ======================================================================
if __name__ == "__main__":
    main()
