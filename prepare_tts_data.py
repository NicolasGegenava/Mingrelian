"""
Prepare single-speaker TTS data from Common Voice Mingrelian.
Uses Speaker #1 (most clips, 829 clips, 80 min, female, 1.4% downvotes).
"""

import csv
import json
import os
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
import librosa

DATA_DIR = Path("cv-corpus-25.0-2026-03-09/xmf")
OUTPUT_DIR = Path("tts_data")
WAV_DIR = OUTPUT_DIR / "wavs"
TARGET_SR = 16000


def load_validated():
    durations = {}
    with open(DATA_DIR / "clip_durations.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            durations[row["clip"]] = int(row["duration[ms]"])

    samples = []
    with open(DATA_DIR / "validated.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dur_ms = durations.get(row["path"], 0)
            if dur_ms > 0:
                row["duration_s"] = dur_ms / 1000.0
                samples.append(row)
    return samples


def normalize_tts_text(text):
    """Normalize text for TTS — keep punctuation for prosody hints."""
    text = text.replace("\u201e", "").replace("\u201c", "")
    text = text.replace("\u2013", "-")
    text = " ".join(text.split()).strip()
    return text


def main():
    print("=" * 60)
    print("TTS DATA PREPARATION — SINGLE SPEAKER")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    WAV_DIR.mkdir(exist_ok=True)

    samples = load_validated()
    print(f"  Total validated: {len(samples)}")

    # Find top speakers
    speaker_clips = defaultdict(list)
    for s in samples:
        speaker_clips[s["client_id"]].append(s)

    # Sort by clip count
    top_speakers = sorted(speaker_clips.items(), key=lambda x: -len(x[1]))

    print(f"\n  Top 5 speakers:")
    for i, (spk, clips) in enumerate(top_speakers[:5]):
        dur = sum(c["duration_s"] for c in clips) / 60
        downvotes = sum(int(c.get("down_votes", 0)) for c in clips)
        gender = next((c.get("gender", "") for c in clips if c.get("gender", "")), "unknown")
        print(f"    #{i+1}: {len(clips)} clips, {dur:.1f} min, gender={gender}, downvotes={downvotes}")

    # Use Speaker #1
    best_spk_id, best_clips = top_speakers[0]
    print(f"\n  Using Speaker #1: {len(best_clips)} clips")

    # Filter for quality
    good_clips = []
    for c in best_clips:
        down = int(c.get("down_votes", 0))
        dur = c["duration_s"]
        if down == 0 and 2.0 <= dur <= 12.0:
            good_clips.append(c)

    print(f"  After quality filter (0 downvotes, 2-12s): {len(good_clips)} clips")
    total_dur = sum(c["duration_s"] for c in good_clips) / 60
    print(f"  Total duration: {total_dur:.1f} min")

    # Split: 95% train, 5% eval
    import random
    random.seed(42)
    random.shuffle(good_clips)
    n_eval = max(10, int(len(good_clips) * 0.05))
    eval_clips = good_clips[:n_eval]
    train_clips = good_clips[n_eval:]

    print(f"  Train: {len(train_clips)} clips, Eval: {len(eval_clips)} clips")

    # Convert audio and create metadata
    for split_name, split_clips in [("train", train_clips), ("eval", eval_clips)]:
        metadata = []
        for i, clip in enumerate(split_clips):
            mp3_path = DATA_DIR / "clips" / clip["path"]
            wav_name = clip["path"].replace(".mp3", ".wav")
            wav_path = WAV_DIR / wav_name

            if not wav_path.exists():
                y, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
                sf.write(str(wav_path), y, TARGET_SR)

            text = normalize_tts_text(clip["sentence"])
            metadata.append({
                "audio_path": str(wav_path),
                "wav_filename": wav_name,
                "text": text,
                "duration_s": clip["duration_s"],
            })

        # Save as JSON manifest
        manifest_path = OUTPUT_DIR / f"{split_name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Also save as LJSpeech-style metadata.csv (pipe-delimited)
        csv_path = OUTPUT_DIR / f"metadata_{split_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            for item in metadata:
                name = item["wav_filename"].replace(".wav", "")
                f.write(f"{name}|{item['text']}|{item['text']}\n")

        print(f"  Saved {split_name}: {len(metadata)} entries")

    # Save speaker info
    info = {
        "speaker_id": hashlib.sha256(best_spk_id.encode()).hexdigest()[:16],
        "total_clips": len(good_clips),
        "train_clips": len(train_clips),
        "eval_clips": len(eval_clips),
        "total_duration_min": total_dur,
        "sample_rate": TARGET_SR,
    }
    with open(OUTPUT_DIR / "speaker_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\n  TTS data ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
