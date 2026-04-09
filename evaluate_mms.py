"""
Evaluate the trained MMS adapter on the test set.
Also show sample predictions for qualitative analysis.
"""

import json
import os
import numpy as np
import torch
import soundfile as sf
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
from evaluate import load as load_metric

MODEL_DIR = "./mms-xmf-adapter"
DATA_DIR = "data_prepared"
TARGET_LANG = "xmf"


def main():
    print("=" * 60)
    print("EVALUATING MMS ADAPTER ON TEST SET")
    print("=" * 60)

    # Load processor
    nested_vocab_path = os.path.join(DATA_DIR, "vocab_mms.json")
    tokenizer = Wav2Vec2CTCTokenizer(
        nested_vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=TARGET_LANG,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load model from best checkpoint
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Model loaded on {device}")

    # Load test data
    with open(os.path.join(DATA_DIR, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Test samples: {len(test_data)}")

    # Run inference
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    predictions = []
    references = []
    samples = []

    for i, item in enumerate(test_data):
        audio, sr = sf.read(item["audio_path"])
        assert sr == 16000

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0]
        ref_text = item["sentence_ctc"]

        predictions.append(pred_text)
        references.append(ref_text)

        if i < 20:  # Save first 20 for display
            samples.append({
                "reference": ref_text,
                "prediction": pred_text,
                "audio": item["wav_filename"],
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")

    # Compute metrics
    # Filter empty
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    pred_filtered, ref_filtered = zip(*pairs)

    wer = wer_metric.compute(predictions=pred_filtered, references=ref_filtered)
    cer = cer_metric.compute(predictions=pred_filtered, references=ref_filtered)

    print(f"\n{'=' * 60}")
    print(f"TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  WER: {wer:.4f} ({wer*100:.1f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.1f}%)")
    print(f"  Samples evaluated: {len(pred_filtered)}")

    # Show sample predictions
    print(f"\n{'=' * 60}")
    print(f"SAMPLE PREDICTIONS (first 20)")
    print(f"{'=' * 60}")
    for i, s in enumerate(samples):
        match = "OK" if s["reference"] == s["prediction"] else "XX"
        print(f"\n  [{match}] Sample {i+1}:")
        print(f"    REF: {s['reference']}")
        print(f"    PRD: {s['prediction']}")

    # Analyze error patterns
    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS")
    print(f"{'=' * 60}")

    # Check Mingrelian-specific char accuracy
    schwa_refs = [r for r in ref_filtered if "\u10f7" in r]
    schwa_preds = [p for p, r in zip(pred_filtered, ref_filtered) if "\u10f7" in r]
    if schwa_refs:
        schwa_cer = cer_metric.compute(predictions=schwa_preds, references=schwa_refs)
        print(f"  CER on sentences with schwa (U+10F7): {schwa_cer:.4f} ({schwa_cer*100:.1f}%)")
        print(f"    ({len(schwa_refs)} sentences)")

    glottal_refs = [r for r in ref_filtered if "\u10f8" in r]
    glottal_preds = [p for p, r in zip(pred_filtered, ref_filtered) if "\u10f8" in r]
    if glottal_refs:
        glottal_cer = cer_metric.compute(predictions=glottal_preds, references=glottal_refs)
        print(f"  CER on sentences with glottal stop (U+10F8): {glottal_cer:.4f} ({glottal_cer*100:.1f}%)")
        print(f"    ({len(glottal_refs)} sentences)")

    # Exact match rate
    exact_matches = sum(1 for p, r in zip(pred_filtered, ref_filtered) if p.strip() == r.strip())
    print(f"  Exact match rate: {exact_matches}/{len(pred_filtered)} ({exact_matches/len(pred_filtered)*100:.1f}%)")

    # Save results
    results = {
        "model": "MMS-1b-all adapter (Georgian warm-start)",
        "test_wer": wer,
        "test_cer": cer,
        "test_samples": len(pred_filtered),
        "exact_match_rate": exact_matches / len(pred_filtered),
        "samples": samples[:20],
    }
    with open("mms_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to mms_test_results.json")


if __name__ == "__main__":
    main()
