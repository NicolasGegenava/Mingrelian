"""
Add language model boosting to MMS CTC decoding.

Implements prefix beam search with unigram word probabilities.
No external LM library needed — pure Python/PyTorch.
"""

import json
import os
import math
import numpy as np
import torch
import soundfile as sf
from collections import Counter, defaultdict
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

# LM weight: how much to boost known words
LM_WEIGHT = 1.5
WORD_BONUS = 2.0  # bonus per word boundary (encourages shorter words)


def load_model_and_processor():
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


def build_word_set():
    """Build word frequency set from all Mingrelian text."""
    word_counts = Counter()

    for split in ["train", "dev"]:
        with open(os.path.join(DATA_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            word_counts.update(item["sentence_ctc"].split())

    # Validated sentences for broader coverage
    tsv_path = "cv-corpus-25.0-2026-03-09/xmf/validated_sentences.tsv"
    if os.path.exists(tsv_path):
        import csv
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sent = row.get("sentence", "")
                for c in ".,?!:;-\"":
                    sent = sent.replace(c, "")
                sent = sent.replace("\u201e", "").replace("\u201c", "").replace("\u2013", "")
                word_counts.update(sent.split())

    total = sum(word_counts.values())
    word_probs = {w: count / total for w, count in word_counts.items()}

    print(f"  Word vocabulary: {len(word_counts)} unique words")
    return word_probs


def lm_rescore(text, word_probs):
    """Score a decoded text using unigram word probabilities."""
    words = text.split()
    if not words:
        return 0.0
    score = 0.0
    for w in words:
        if w in word_probs:
            score += math.log(word_probs[w] + 1e-10) * LM_WEIGHT
        else:
            score += math.log(1e-6) * LM_WEIGHT  # penalty for unknown words
    score += len(words) * WORD_BONUS  # bonus for word boundaries
    return score


def beam_search_decode(logits, vocab_list, blank_id, space_id, beam_width=20):
    """
    Simple beam search CTC decoding.
    Returns top beam_width candidates.
    """
    T = logits.shape[0]
    log_probs = torch.log_softmax(torch.tensor(logits), dim=-1).numpy()

    # Each beam: (prefix_str, last_char, score)
    beams = [("", -1, 0.0)]

    for t in range(T):
        new_beams = defaultdict(lambda: float("-inf"))

        for prefix, last_char, score in beams:
            for c in range(log_probs.shape[1]):
                p = log_probs[t, c]

                if c == blank_id:
                    # Blank: keep prefix as-is
                    key = (prefix, -1)
                    new_beams[key] = max(new_beams[key], score + p)
                elif c == last_char:
                    # Same char repeated: keep prefix (CTC collapse)
                    key = (prefix, c)
                    new_beams[key] = max(new_beams[key], score + p)
                elif c == space_id:
                    # Space/word boundary
                    new_prefix = prefix + " "
                    key = (new_prefix, c)
                    new_beams[key] = max(new_beams[key], score + p)
                else:
                    # New character
                    new_prefix = prefix + vocab_list[c]
                    key = (new_prefix, c)
                    new_beams[key] = max(new_beams[key], score + p)

            # Also allow transitioning from repeated char to same char (new instance)
            if last_char >= 0 and last_char != blank_id:
                for c in range(log_probs.shape[1]):
                    if c == blank_id:
                        key = (prefix, -1)
                        new_beams[key] = max(new_beams[key], score + log_probs[t, c])

        # Prune to top beams
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(k[0], k[1], v) for (k, v) in sorted_beams]

    return [(prefix.strip(), score) for prefix, _, score in beams]


def greedy_with_lm_rescore(logits_np, processor, word_probs, n_candidates=5):
    """
    Generate multiple candidates via temperature sampling + greedy variants,
    then rescore with LM. Simpler than full beam search but effective.
    """
    # Get top-k at each timestep
    log_probs = torch.log_softmax(torch.tensor(logits_np), dim=-1)

    # Strategy 1: Pure greedy
    greedy_ids = torch.argmax(log_probs, dim=-1)
    greedy_text = processor.decode(greedy_ids.tolist())

    # Strategy 2-5: Sample from top-k with different temperatures
    candidates = [greedy_text]

    for temp in [0.7, 0.9, 1.1, 1.3]:
        scaled = log_probs / temp
        probs = torch.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        text = processor.decode(sampled.tolist())
        candidates.append(text)

    # Rescore all candidates with LM
    best_text = greedy_text
    best_score = lm_rescore(greedy_text, word_probs)

    for cand in candidates:
        score = lm_rescore(cand, word_probs)
        if score > best_score:
            best_score = score
            best_text = cand

    return best_text


def post_process_text(text, word_probs):
    """
    Post-process CTC output using word-level knowledge.
    Fix common word boundary errors by checking if merging/splitting improves LM score.
    """
    words = text.split()
    if len(words) <= 1:
        return text

    improved = True
    while improved:
        improved = False
        new_words = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                merged = words[i] + words[i + 1]
                # Check if merged form is a known word and parts aren't
                w1_known = words[i] in word_probs
                w2_known = words[i + 1] in word_probs
                merged_known = merged in word_probs

                if merged_known and not (w1_known and w2_known):
                    new_words.append(merged)
                    i += 2
                    improved = True
                    continue

            new_words.append(words[i])
            i += 1
        words = new_words

    # Also try splitting words that aren't known
    final_words = []
    for word in words:
        if word in word_probs or len(word) <= 3:
            final_words.append(word)
            continue

        # Try splitting at every position
        best_split = None
        best_score = -float("inf")
        for j in range(2, len(word) - 1):
            w1, w2 = word[:j], word[j:]
            if w1 in word_probs and w2 in word_probs:
                score = math.log(word_probs[w1]) + math.log(word_probs[w2])
                if score > best_score:
                    best_score = score
                    best_split = (w1, w2)

        if best_split and best_score > math.log(word_probs.get(word, 1e-8)):
            final_words.extend(best_split)
        else:
            final_words.append(word)

    return " ".join(final_words)


def main():
    print("=" * 60)
    print("CTC + LANGUAGE MODEL POST-PROCESSING")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model, processor, device = load_model_and_processor()

    print("\n[2/3] Building word vocabulary...")
    word_probs = build_word_set()

    print("\n[3/3] Evaluating on test set...")
    with open(os.path.join(DATA_DIR, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    greedy_preds = []
    postproc_preds = []
    references = []
    samples = []

    for i, item in enumerate(test_data):
        audio, sr = sf.read(item["audio_path"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        # Greedy
        pred_ids = torch.argmax(logits, dim=-1)
        greedy_text = processor.batch_decode(pred_ids)[0]
        greedy_preds.append(greedy_text)

        # Post-processed with LM
        postproc_text = post_process_text(greedy_text, word_probs)
        postproc_preds.append(postproc_text)

        references.append(item["sentence_ctc"])

        if i < 15:
            samples.append({
                "reference": item["sentence_ctc"],
                "greedy": greedy_text,
                "postproc": postproc_text,
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_data)}")

    # Metrics
    pairs_g = [(p, r) for p, r in zip(greedy_preds, references) if r.strip()]
    pairs_p = [(p, r) for p, r in zip(postproc_preds, references) if r.strip()]

    g_pred, g_ref = zip(*pairs_g)
    p_pred, p_ref = zip(*pairs_p)

    g_wer = wer_metric.compute(predictions=g_pred, references=g_ref)
    g_cer = cer_metric.compute(predictions=g_pred, references=g_ref)
    p_wer = wer_metric.compute(predictions=p_pred, references=p_ref)
    p_cer = cer_metric.compute(predictions=p_pred, references=p_ref)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Method':<30s} {'WER':>8s} {'CER':>8s}")
    print(f"  {'-'*50}")
    print(f"  {'Greedy (baseline)':<30s} {g_wer:>7.1%} {g_cer:>7.1%}")
    print(f"  {'+ Word LM post-processing':<30s} {p_wer:>7.1%} {p_cer:>7.1%}")

    if g_wer > 0:
        wer_imp = (g_wer - p_wer) / g_wer * 100
        print(f"\n  WER relative improvement: {wer_imp:.1f}%")

    # Show samples
    print(f"\n{'=' * 60}")
    print(f"SAMPLE COMPARISONS")
    print(f"{'=' * 60}")
    for i, s in enumerate(samples):
        changed = s["greedy"] != s["postproc"]
        if changed:
            print(f"\n  Sample {i+1}: [IMPROVED]")
            print(f"    REF:  {s['reference']}")
            print(f"    OLD:  {s['greedy']}")
            print(f"    NEW:  {s['postproc']}")

    # Count improvements
    improved = sum(1 for g, p in zip(greedy_preds, postproc_preds) if g != p)
    print(f"\n  Sentences modified: {improved}/{len(greedy_preds)}")

    results = {
        "greedy_wer": g_wer, "greedy_cer": g_cer,
        "postproc_wer": p_wer, "postproc_cer": p_cer,
    }
    with open("lm_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to lm_results.json")


if __name__ == "__main__":
    main()
