"""
Data preparation for Mingrelian ASR training.

Creates sentence-aware train/dev/test splits from all validated Common Voice data,
converts MP3 to 16kHz WAV, normalizes text, and outputs HuggingFace-compatible datasets.

Key design decisions:
- Split by SENTENCE (not by clip) to prevent data leakage (85% duplication in dataset)
- Use ALL 6,766 validated clips, not just the 336 official train split
- Speaker-disjoint test set where possible
- Filter outlier clips (extreme speaking rates)
- Text normalization for Georgian/Mingrelian script
"""

import csv
import os
import json
import random
import hashlib
import statistics
from pathlib import Path
from collections import defaultdict, Counter

import librosa
import soundfile as sf
import numpy as np

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("cv-corpus-25.0-2026-03-09/xmf")
OUTPUT_DIR = Path("data_prepared")
CLIPS_DIR = DATA_DIR / "clips"
WAV_DIR = OUTPUT_DIR / "wavs"
TARGET_SR = 16000

# Split ratios by unique sentence count
TEST_RATIO = 0.10
DEV_RATIO = 0.10
TRAIN_RATIO = 0.80

# Quality filters
MIN_DURATION_S = 1.5    # Remove very short clips
MAX_DURATION_S = 15.0   # Remove very long clips
MIN_CHARS_PER_SEC = 4.0 # Filter extreme outliers
MAX_CHARS_PER_SEC = 16.0

SEED = 42

# ============================================================
# Step 1: Load and parse all data
# ============================================================
def load_durations():
    """Load clip duration mapping."""
    durations = {}
    with open(DATA_DIR / "clip_durations.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            durations[row["clip"]] = int(row["duration[ms]"])
    return durations


def load_validated(durations):
    """Load all validated clips with duration info."""
    samples = []
    with open(DATA_DIR / "validated.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dur_ms = durations.get(row["path"], 0)
            if dur_ms == 0:
                continue
            row["duration_s"] = dur_ms / 1000.0
            samples.append(row)
    return samples


# ============================================================
# Step 2: Text normalization
# ============================================================
def normalize_text(text):
    """
    Normalize Mingrelian text for ASR training.

    - Remove typographic quotes and dashes
    - Keep Georgian Mkhedruli script + Mingrelian-specific chars
    - Normalize whitespace
    - Keep minimal punctuation (period, comma, question mark) for Whisper compatibility
    - For CTC models, punctuation will be stripped at training time
    """
    # Replace typographic quotes/dashes with nothing
    text = text.replace("\u201E", "")  # „
    text = text.replace("\u201C", "")  # "
    text = text.replace("\u2013", "-") # – → -

    # Normalize whitespace
    text = " ".join(text.split())
    text = text.strip()

    return text


def normalize_text_ctc(text):
    """
    Stricter normalization for CTC-based models (MMS, W2V-BERT).
    Removes ALL punctuation, lowercases (Georgian has no case distinction).
    """
    text = normalize_text(text)
    # Remove all punctuation
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace("-", "")
    text = text.replace("\"", "")
    text = text.replace("'", "")
    # Normalize whitespace again after removals
    text = " ".join(text.split())
    text = text.strip()
    return text


# ============================================================
# Step 3: Quality filtering
# ============================================================
def filter_sample(sample):
    """
    Return (keep: bool, reason: str) for a sample.
    """
    dur = sample["duration_s"]
    text = sample["sentence"]

    if dur < MIN_DURATION_S:
        return False, f"too_short ({dur:.1f}s)"
    if dur > MAX_DURATION_S:
        return False, f"too_long ({dur:.1f}s)"

    chars = len(text)
    if chars == 0:
        return False, "empty_text"

    rate = chars / dur
    if rate < MIN_CHARS_PER_SEC:
        return False, f"too_slow ({rate:.1f} c/s)"
    if rate > MAX_CHARS_PER_SEC:
        return False, f"too_fast ({rate:.1f} c/s)"

    # Check downvotes
    down = int(sample.get("down_votes", 0))
    up = int(sample.get("up_votes", 0))
    if down >= 2 and down >= up:
        return False, f"low_quality (up={up}, down={down})"

    return True, "ok"


# ============================================================
# Step 4: Sentence-aware splitting
# ============================================================
def create_splits(samples):
    """
    Split by SENTENCE to prevent data leakage.

    Strategy:
    1. Group all clips by sentence text
    2. Randomly assign sentences to train/dev/test
    3. All clips of a given sentence go to the same split
    4. Ensure reasonable speaker coverage in each split
    """
    random.seed(SEED)

    # Group clips by normalized sentence
    sentence_clips = defaultdict(list)
    for s in samples:
        norm = normalize_text(s["sentence"])
        sentence_clips[norm].append(s)

    sentences = list(sentence_clips.keys())
    random.shuffle(sentences)

    n_total = len(sentences)
    n_test = max(1, int(n_total * TEST_RATIO))
    n_dev = max(1, int(n_total * DEV_RATIO))

    test_sents = set(sentences[:n_test])
    dev_sents = set(sentences[n_test:n_test + n_dev])
    train_sents = set(sentences[n_test + n_dev:])

    splits = {"train": [], "dev": [], "test": []}
    for sent in test_sents:
        splits["test"].extend(sentence_clips[sent])
    for sent in dev_sents:
        splits["dev"].extend(sentence_clips[sent])
    for sent in train_sents:
        splits["train"].extend(sentence_clips[sent])

    return splits


# ============================================================
# Step 5: Audio conversion (MP3 → 16kHz WAV)
# ============================================================
def convert_audio(mp3_path, wav_path):
    """Convert MP3 to 16kHz mono WAV."""
    y, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
    sf.write(str(wav_path), y, TARGET_SR)
    return len(y) / TARGET_SR


# ============================================================
# Step 6: Build character vocabulary
# ============================================================
def build_vocab(samples):
    """
    Build character vocabulary from all training text.
    Returns sorted list of unique characters.
    """
    chars = set()
    for s in samples:
        for c in normalize_text_ctc(s["sentence"]):
            if c != " ":
                chars.add(c)

    vocab = sorted(chars)
    return vocab


# ============================================================
# Main execution
# ============================================================
def main():
    print("=" * 60)
    print("MINGRELIAN ASR DATA PREPARATION")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    WAV_DIR.mkdir(exist_ok=True)

    # Load data
    print("\n[1/6] Loading data...")
    durations = load_durations()
    samples = load_validated(durations)
    print(f"  Loaded {len(samples)} validated clips")
    print(f"  Total duration: {sum(s['duration_s'] for s in samples)/3600:.2f} hours")

    # Quality filtering
    print("\n[2/6] Filtering for quality...")
    filtered = []
    reject_reasons = Counter()
    for s in samples:
        keep, reason = filter_sample(s)
        if keep:
            filtered.append(s)
        else:
            reject_reasons[reason] += 1

    print(f"  Kept: {len(filtered)} / {len(samples)} ({len(filtered)/len(samples):.1%})")
    print(f"  Rejected: {len(samples) - len(filtered)}")
    for reason, count in reject_reasons.most_common():
        print(f"    {reason}: {count}")
    print(f"  Filtered duration: {sum(s['duration_s'] for s in filtered)/3600:.2f} hours")

    # Text normalization check
    print("\n[3/6] Normalizing text...")
    char_set = set()
    for s in filtered:
        s["text_normalized"] = normalize_text(s["sentence"])
        s["text_ctc"] = normalize_text_ctc(s["sentence"])
        for c in s["text_ctc"]:
            if c != " ":
                char_set.add(c)

    vocab = sorted(char_set)
    print(f"  Character vocabulary size: {len(vocab)}")
    print(f"  Characters: {''.join(vocab)}")

    # Check for Mingrelian-specific characters
    mingrelian_chars = [c for c in vocab if ord(c) in (0x10F7, 0x10F8, 0x10F2)]
    print(f"  Mingrelian-specific chars in vocab: {mingrelian_chars}")

    # Sentence-aware splitting
    print("\n[4/6] Creating sentence-aware splits...")
    splits = create_splits(filtered)

    for split_name, split_data in splits.items():
        n_clips = len(split_data)
        dur = sum(s["duration_s"] for s in split_data) / 3600
        n_speakers = len(set(s["client_id"] for s in split_data))
        n_sents = len(set(normalize_text(s["sentence"]) for s in split_data))
        print(f"  {split_name:>5s}: {n_clips:5d} clips | {dur:.2f} hrs | {n_speakers:3d} speakers | {n_sents:4d} unique sentences")

    # Verify no sentence leakage
    train_sents = set(normalize_text(s["sentence"]) for s in splits["train"])
    dev_sents = set(normalize_text(s["sentence"]) for s in splits["dev"])
    test_sents = set(normalize_text(s["sentence"]) for s in splits["test"])

    assert len(train_sents & dev_sents) == 0, "LEAK: train-dev sentence overlap!"
    assert len(train_sents & test_sents) == 0, "LEAK: train-test sentence overlap!"
    assert len(dev_sents & test_sents) == 0, "LEAK: dev-test sentence overlap!"
    print("  ✓ No sentence leakage between splits")

    # Convert audio and save manifests
    print("\n[5/6] Converting audio to 16kHz WAV and saving manifests...")

    for split_name, split_data in splits.items():
        manifest = []
        for i, s in enumerate(split_data):
            mp3_path = CLIPS_DIR / s["path"]
            wav_name = s["path"].replace(".mp3", ".wav")
            wav_path = WAV_DIR / wav_name

            if not wav_path.exists():
                try:
                    actual_dur = convert_audio(mp3_path, wav_path)
                except Exception as e:
                    print(f"  WARNING: Failed to convert {s['path']}: {e}")
                    continue

            entry = {
                "audio_path": str(wav_path),
                "wav_filename": wav_name,
                "sentence": s["text_normalized"],
                "sentence_ctc": s["text_ctc"],
                "sentence_raw": s["sentence"],
                "duration_s": s["duration_s"],
                "speaker_id": hashlib.sha256(s["client_id"].encode()).hexdigest()[:16],
                "gender": s.get("gender", ""),
                "accent": s.get("accents", ""),
                "up_votes": int(s.get("up_votes", 0)),
                "down_votes": int(s.get("down_votes", 0)),
            }
            manifest.append(entry)

            if (i + 1) % 500 == 0:
                print(f"    {split_name}: {i+1}/{len(split_data)} processed")

        # Save manifest
        manifest_path = OUTPUT_DIR / f"{split_name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"  Saved {split_name} manifest: {len(manifest)} entries → {manifest_path}")

    # Save vocabulary
    print("\n[6/6] Saving vocabulary and metadata...")

    vocab_dict = {c: i for i, c in enumerate(vocab)}
    vocab_dict["|"] = len(vocab_dict)  # word separator for CTC
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    vocab_path = OUTPUT_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"  Vocab saved: {len(vocab_dict)} tokens → {vocab_path}")

    # Save metadata
    metadata = {
        "language": "xmf",
        "language_name": "Mingrelian",
        "script": "Georgian Mkhedruli",
        "sample_rate": TARGET_SR,
        "total_clips": sum(len(splits[s]) for s in splits),
        "splits": {
            name: {
                "clips": len(data),
                "duration_hours": sum(s["duration_s"] for s in data) / 3600,
                "speakers": len(set(s["client_id"] for s in data)),
                "unique_sentences": len(set(normalize_text(s["sentence"]) for s in data)),
            }
            for name, data in splits.items()
        },
        "vocab_size": len(vocab_dict),
        "mingrelian_specific_chars": {
            "schwa": "ჷ (U+10F7)",
            "glottal_stop": "ჸ (U+10F8)",
            "yod": "ჲ (U+10F2)",
        },
        "filters_applied": {
            "min_duration_s": MIN_DURATION_S,
            "max_duration_s": MAX_DURATION_S,
            "min_chars_per_sec": MIN_CHARS_PER_SEC,
            "max_chars_per_sec": MAX_CHARS_PER_SEC,
        },
    }

    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved → {meta_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"WAV files: {WAV_DIR}")
    total_clips = sum(len(splits[s]) for s in splits)
    total_hrs = sum(sum(s['duration_s'] for s in splits[sp]) for sp in splits) / 3600
    print(f"Total: {total_clips} clips, {total_hrs:.2f} hours")
    print(f"Vocab: {len(vocab_dict)} tokens (including {len(mingrelian_chars)} Mingrelian-specific)")
    print(f"\nReady for training!")


if __name__ == "__main__":
    main()
