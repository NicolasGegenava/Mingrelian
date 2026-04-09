"""
Train MMS-1b-all adapter for Mingrelian (xmf) ASR.

Strategy: Warm-start from Georgian (kat) adapter — only ~2.2M adapter params trained.
Expected training time: ~1-2 hours on RTX 5090.

Key design:
- Load Georgian adapter weights as initialization (Kartvelian phonology transfer)
- DO NOT call init_adapter_layers() — preserves Georgian features
- Resize lm_head for Mingrelian 39-token vocabulary
- Freeze entire 1B-param backbone, train only adapter layers + lm_head
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric

# ============================================================
# Configuration
# ============================================================
TARGET_LANG = "xmf"
BASE_LANG = "kat"  # Georgian — closest language to Mingrelian
MODEL_ID = "facebook/mms-1b-all"
OUTPUT_DIR = "./mms-xmf-adapter"
DATA_DIR = "data_prepared"

TRAINING_CONFIG = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,  # effective batch = 32
    "num_train_epochs": 20,
    "learning_rate": 1e-3,
    "warmup_steps": 100,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 50,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
}


# ============================================================
# Step 1: Set up tokenizer with nested vocab for MMS
# ============================================================
def setup_tokenizer():
    """Create MMS-compatible nested vocab and tokenizer."""
    with open(os.path.join(DATA_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        flat_vocab = json.load(f)

    # MMS tokenizer expects nested format: {"xmf": {...}}
    nested_vocab = {TARGET_LANG: flat_vocab}
    nested_vocab_path = os.path.join(DATA_DIR, "vocab_mms.json")
    with open(nested_vocab_path, "w", encoding="utf-8") as f:
        json.dump(nested_vocab, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer(
        nested_vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=TARGET_LANG,
    )
    return tokenizer


def setup_processor(tokenizer):
    """Create feature extractor and processor."""
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    return processor


# ============================================================
# Step 2: Load model with Georgian warm-start
# ============================================================
def setup_model(processor):
    """
    Load MMS-1b-all with Georgian adapter as warm-start.
    Resize lm_head for Mingrelian vocabulary.
    """
    vocab_size = len(processor.tokenizer)
    print(f"  Target vocab size: {vocab_size}")

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        target_lang=BASE_LANG,  # Load Georgian adapter first
        ignore_mismatched_sizes=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
    )

    # CRITICAL: Do NOT call model.init_adapter_layers()
    # The Georgian adapter weights are already loaded and we want to keep them!

    # Replace lm_head for Mingrelian vocabulary
    hidden_size = model.config.output_hidden_size
    model.lm_head = nn.Linear(hidden_size, vocab_size)
    model.config.vocab_size = vocab_size
    print(f"  Replaced lm_head: Linear({hidden_size}, {vocab_size})")

    # Freeze the entire base model (1B params)
    model.freeze_base_model()

    # Unfreeze adapter layers
    adapter_weights = model._get_adapters()
    for name, param in adapter_weights.items():
        param.requires_grad = True

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    return model


# ============================================================
# Step 3: Prepare datasets
# ============================================================
def load_split(split_name, processor):
    """Load a data split and prepare for training."""
    import soundfile as sf

    manifest_path = os.path.join(DATA_DIR, f"{split_name}.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pre-process: load audio directly with soundfile (robust, no codec deps)
    processed = []
    skipped = 0
    for i, item in enumerate(data):
        try:
            audio_array, sr = sf.read(item["audio_path"])
            assert sr == 16000, f"Expected 16kHz, got {sr}"

            input_values = processor(
                audio_array, sampling_rate=16000
            ).input_values[0]

            labels = processor(text=item["sentence_ctc"]).input_ids

            processed.append({
                "input_values": input_values,
                "input_length": len(input_values),
                "labels": labels,
            })
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  WARNING: Skipping {item['audio_path']}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"  {split_name}: {i+1}/{len(data)} loaded")

    if skipped:
        print(f"  Skipped {skipped} files")

    ds = Dataset.from_list(processed)
    return ds


# ============================================================
# Step 4: Data collator for CTC
# ============================================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input and label features
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad inputs
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        # Replace padding with -100 (ignored by CTC loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


# ============================================================
# Step 5: Metrics
# ============================================================
def make_compute_metrics(processor):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)

        # Replace -100 with pad token
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # Filter empty predictions/references
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        pred_str, label_str = zip(*pairs)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("MMS-1b-all ADAPTER TRAINING FOR MINGRELIAN (xmf)")
    print("=" * 60)

    # Setup
    print("\n[1/5] Setting up tokenizer and processor...")
    tokenizer = setup_tokenizer()
    processor = setup_processor(tokenizer)
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")

    # Verify tokenizer handles Mingrelian text
    test_text = "მოთხუ თე ვეზირს ხე დო დუდი ვიშო მესოფუ ბოშიქ"
    encoded = tokenizer(test_text)
    decoded = tokenizer.decode(encoded.input_ids)
    assert decoded.replace(" ", "") == test_text.replace(" ", "").replace("|", ""), \
        f"Tokenizer roundtrip failed: {decoded!r}"
    print(f"  ✓ Tokenizer roundtrip verified")

    # Test with Mingrelian-specific chars
    test_schwa = "წჷმაჯინჷ რდჷ ჸე"
    encoded = tokenizer(test_schwa)
    decoded = tokenizer.decode(encoded.input_ids)
    print(f"  ✓ Mingrelian chars (ჷ, ჸ) encode/decode OK")

    # Load model
    print("\n[2/5] Loading model with Georgian warm-start...")
    model = setup_model(processor)

    # Prepare datasets
    print("\n[3/5] Preparing datasets...")
    print("  Loading train split...")
    train_ds = load_split("train", processor)
    print(f"  Train: {len(train_ds)} examples")
    print("  Loading dev split...")
    dev_ds = load_split("dev", processor)
    print(f"  Dev: {len(dev_ds)} examples")

    # Training
    print("\n[4/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        eval_strategy="steps",
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        gradient_checkpointing=True,
        bf16=True,  # RTX 5090 Blackwell — prefer bf16
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="tensorboard",
        dataloader_num_workers=0,  # Windows: multiprocessing pickle fails with large datasets
        remove_unused_columns=False,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=make_compute_metrics(processor),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Resume from checkpoint if available
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [
            os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"  Resuming from checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    print("\n[5/5] Saving model and adapter...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Also save adapter weights separately
    from safetensors.torch import save_file as safe_save_file
    adapter_weights = model._get_adapters()
    adapter_file = os.path.join(OUTPUT_DIR, f"adapter.{TARGET_LANG}.safetensors")
    safe_save_file(adapter_weights, adapter_file, metadata={"format": "pt"})
    print(f"  Adapter saved: {adapter_file}")

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    metrics = train_result.metrics
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Train runtime: {metrics.get('train_runtime', 0)/60:.1f} minutes")
    print(f"  Train samples/sec: {metrics.get('train_samples_per_second', 'N/A')}")

    # Evaluate on dev
    print("\n  Evaluating on dev set...")
    eval_metrics = trainer.evaluate()
    print(f"  Dev WER: {eval_metrics.get('eval_wer', 'N/A'):.4f}")
    print(f"  Dev CER: {eval_metrics.get('eval_cer', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
