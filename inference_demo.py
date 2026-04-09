"""
Inference demo: Run the trained MMS Mingrelian ASR on test samples.
Shows side-by-side reference vs prediction with quality indicators.
Picks diverse samples (different speakers, lengths, with/without special chars).
"""

import json
import os
import random
import torch
import soundfile as sf
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

MODEL_DIR = "./mms-xmf-adapter"
DATA_DIR = "data_prepared"
TARGET_LANG = "xmf"

def load_model():
    nested_vocab_path = os.path.join(DATA_DIR, "vocab_mms.json")
    tokenizer = Wav2Vec2CTCTokenizer(
        nested_vocab_path, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|", target_lang=TARGET_LANG,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device


def transcribe(model, processor, device, audio_path):
    audio, sr = sf.read(audio_path)
    assert sr == 16000
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def char_diff(ref, pred):
    """Simple character-level diff indicator."""
    errors = 0
    for i in range(max(len(ref), len(pred))):
        r = ref[i] if i < len(ref) else ""
        p = pred[i] if i < len(pred) else ""
        if r != p:
            errors += 1
    return errors


def main():
    print("Loading model...")
    model, processor, device = load_model()

    # Load test data
    with open(os.path.join(DATA_DIR, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Select diverse samples
    random.seed(42)

    # Group by sentence to pick unique ones
    seen_sents = set()
    unique_samples = []
    for item in test_data:
        if item["sentence_ctc"] not in seen_sents:
            seen_sents.add(item["sentence_ctc"])
            unique_samples.append(item)

    # Pick samples with different characteristics
    schwa_samples = [s for s in unique_samples if "\u10f7" in s["sentence_ctc"]]
    glottal_samples = [s for s in unique_samples if "\u10f8" in s["sentence_ctc"]]
    short_samples = [s for s in unique_samples if s["duration_s"] < 4.5]
    long_samples = [s for s in unique_samples if s["duration_s"] > 8]
    normal_samples = [s for s in unique_samples if 4.5 <= s["duration_s"] <= 8]

    selected = []
    # Pick 5 with schwa
    selected.extend(random.sample(schwa_samples, min(5, len(schwa_samples))))
    # Pick 3 with glottal stop
    selected.extend(random.sample(glottal_samples, min(3, len(glottal_samples))))
    # Pick 3 short
    selected.extend(random.sample(short_samples, min(3, len(short_samples))))
    # Pick 3 long
    selected.extend(random.sample(long_samples, min(3, len(long_samples))))
    # Pick 6 normal
    selected.extend(random.sample(normal_samples, min(6, len(normal_samples))))

    # Deduplicate
    seen = set()
    final = []
    for s in selected:
        if s["sentence_ctc"] not in seen:
            seen.add(s["sentence_ctc"])
            final.append(s)
    final = final[:20]

    # Run inference
    results = []
    print(f"\n{'='*70}")
    print(f" MINGRELIAN ASR INFERENCE DEMO")
    print(f" Model: MMS-1b-all adapter (Georgian warm-start)")
    print(f" Test samples: {len(final)} diverse clips")
    print(f"{'='*70}\n")

    perfect = 0
    for i, sample in enumerate(final):
        pred = transcribe(model, processor, device, sample["audio_path"])
        ref = sample["sentence_ctc"]
        is_match = pred.strip() == ref.strip()
        if is_match:
            perfect += 1

        errors = char_diff(ref, pred)
        cer = errors / max(len(ref), 1)

        tag = "PERFECT" if is_match else f"CER:{cer:.0%}"
        dur = sample["duration_s"]

        # Detect special chars
        special = []
        if "\u10f7" in ref: special.append("schwa")
        if "\u10f8" in ref: special.append("glottal")

        print(f"  Sample {i+1:2d} [{tag:>8s}] ({dur:.1f}s) {' '.join(f'[{s}]' for s in special)}")
        print(f"    REF: {ref}")
        print(f"    PRD: {pred}")
        if not is_match:
            # Show differences
            diff = ""
            for j in range(max(len(ref), len(pred))):
                r = ref[j] if j < len(ref) else " "
                p = pred[j] if j < len(pred) else " "
                diff += "^" if r != p else " "
            print(f"    DIF: {diff}")
        print()

        results.append({
            "reference": ref,
            "prediction": pred,
            "exact_match": is_match,
            "duration_s": dur,
            "audio": sample["wav_filename"],
            "has_schwa": "\u10f7" in ref,
            "has_glottal": "\u10f8" in ref,
        })

    print(f"{'='*70}")
    print(f"  SUMMARY: {perfect}/{len(final)} perfect matches ({perfect/len(final)*100:.0f}%)")
    print(f"{'='*70}")

    # Save results
    with open("inference_samples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to inference_samples.json")


if __name__ == "__main__":
    main()
