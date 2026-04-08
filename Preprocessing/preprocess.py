import pandas as pd
import numpy as np
import os
import json
import librosa
import soundfile as sf
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define paths
base_dir = r"d:\MegrelianSST\Common Voice\cv-corpus-24.0-2025-12-05\xmf"
clips_dir = os.path.join(base_dir, "clips")
processed_dir = os.path.join(base_dir, "processed")
audio_out_dir = os.path.join(processed_dir, "audio")

# Create directories
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(audio_out_dir, exist_ok=True)

print("Loading TSV files...")
train = pd.read_csv(os.path.join(base_dir, "train.tsv"), sep="\t")
dev = pd.read_csv(os.path.join(base_dir, "dev.tsv"), sep="\t")
test = pd.read_csv(os.path.join(base_dir, "test.tsv"), sep="\t")
clip_durations = pd.read_csv(os.path.join(base_dir, "clip_durations.tsv"), sep="\t")

if "clip" in clip_durations.columns:
    clip_durations = clip_durations.rename(columns={"clip": "path"})

# 1. Fill NaN Gender with 'male'
for df in [train, dev, test]:
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("male")

# 2. Text Normalization
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Replace standard English/Georgian punctuation with space
    text = re.sub(r'[.,?!;:\'"()\[\]{}...-]', ' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Normalizing text...")
for df in [train, dev, test]:
    df["sentence_norm"] = df["sentence"].apply(normalize_text)

# 3. Generate Vocabulary
print("Generating vocabulary...")
all_text = " ".join(train["sentence_norm"].tolist() + dev["sentence_norm"].tolist() + test["sentence_norm"].tolist())
vocab = list(set(all_text))
vocab.sort()

# Convert to huggingface / Meta ASR format vocab (character to ID mapping)
vocab_dict = {v: k for k, v in enumerate(vocab)}
vocab_dict["|"] = vocab_dict.get(" ", len(vocab_dict)) # Often space is mapped to |
if " " in vocab_dict:
    del vocab_dict[" "]

# Add UNK and PAD
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open(os.path.join(processed_dir, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print("Vocabulary saved. Total size:", len(vocab_dict))

# 4. Merge Durations
print("Merging duration...")
train = pd.merge(train, clip_durations[["path", "duration[ms]"]], on="path", how="left")
dev = pd.merge(dev, clip_durations[["path", "duration[ms]"]], on="path", how="left")
test = pd.merge(test, clip_durations[["path", "duration[ms]"]], on="path", how="left")

# 5. Audio Conversion Task
def process_audio(filename):
    src_path = os.path.join(clips_dir, filename)
    dst_name = filename.replace(".mp3", ".wav")
    dst_path = os.path.join(audio_out_dir, dst_name)
    
    # If already exists and size > 0, skip
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1000:
        return filename, dst_name, True
        
    try:
        y, sr = librosa.load(src_path, sr=16000)
        sf.write(dst_path, y, 16000)
        return filename, dst_name, True
    except Exception as e:
        return filename, None, False

print("Converting audio (this might take a while)...")
all_files = pd.concat([train["path"], dev["path"], test["path"]]).unique()

path_mapping = {}
failed_files = []

# Using ThreadPoolExecutor or ProcessPoolExecutor depending on I/O vs CPU bound. Librosa is CPU bound mostly.
# We'll use ThreadPool here to be safe with Windows, but might be slow.
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
num_workers = max(1, multiprocessing.cpu_count() - 1)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_audio, f): f for f in all_files}
    for future in tqdm(as_completed(futures), total=len(all_files), desc="Converting to standard 16kHz WAV"):
        orig, new_path, success = future.result()
        if success:
            path_mapping[orig] = new_path
        else:
            failed_files.append(orig)

if failed_files:
    print(f"Warning: Failed to convert {len(failed_files)} files.")

# Update paths
for df in [train, dev, test]:
    df["path"] = df["path"].map(lambda x: path_mapping.get(x, x))

# Save processed TSVs
print("Saving processed TSVs...")
train.to_csv(os.path.join(processed_dir, "train_processed.tsv"), sep="\t", index=False)
dev.to_csv(os.path.join(processed_dir, "dev_processed.tsv"), sep="\t", index=False)
test.to_csv(os.path.join(processed_dir, "test_processed.tsv"), sep="\t", index=False)

print("Preprocessing complete!")
