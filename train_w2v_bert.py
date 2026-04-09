"""
Train Wav2Vec2-BERT 2.0 with CTC for Mingrelian (xmf) ASR.

600M params, fine-tuning encoder + CTC head. Uses SeamlessM4T feature extractor (fbank).
Expected: better WER than MMS adapter since more params are trained.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import soundfile as sf
from datasets import Dataset
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "facebook/w2v-bert-2.0"
OUTPUT_DIR = "./w2v-bert-xmf"
DATA_DIR = "data_prepared"

TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,  # effective batch = 32
    "num_train_epochs": 15,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "eval_steps": 300,
    "save_steps": 300,
    "logging_steps": 50,
    "save_total_limit": 3,
}


# ============================================================
# Setup
# ============================================================
def setup_processor():
    """Create tokenizer and processor for W2V-BERT 2.0."""
    # W2V-BERT uses flat vocab (no nesting)
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(DATA_DIR, "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(MODEL_ID)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    return processor


def setup_model(processor):
    """Load W2V-BERT 2.0 with CTC head."""
    vocab_size = len(processor.tokenizer)
    print(f"  Vocab size: {vocab_size}")

    model = Wav2Vec2BertForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size,
    )

    # Freeze feature_projection, train encoder + adapter + lm_head
    for param in model.wav2vec2_bert.feature_projection.parameters():
        param.requires_grad = False
    # encoder, adapter, lm_head are all trainable by default

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    return model


# ============================================================
# Data loading
# ============================================================
def load_split(split_name, processor):
    """Load data and extract features."""
    manifest_path = os.path.join(DATA_DIR, f"{split_name}.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    skipped = 0
    for i, item in enumerate(data):
        try:
            audio, sr = sf.read(item["audio_path"])
            assert sr == 16000

            # W2V-BERT uses input_features (fbank), not input_values (raw waveform)
            features = processor(audio, sampling_rate=16000)
            input_features = features.input_features[0]
            labels = processor(text=item["sentence_ctc"]).input_ids

            processed.append({
                "input_features": input_features,
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

    return Dataset.from_list(processed)


# ============================================================
# Data collator
# ============================================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# ============================================================
# Metrics
# ============================================================
def make_compute_metrics(processor):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
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
    print("W2V-BERT 2.0 CTC TRAINING FOR MINGRELIAN (xmf)")
    print("=" * 60)

    print("\n[1/5] Setting up processor...")
    processor = setup_processor()

    print("\n[2/5] Loading model...")
    model = setup_model(processor)

    print("\n[3/5] Preparing datasets...")
    print("  Loading train split...")
    train_ds = load_split("train", processor)
    print(f"  Train: {len(train_ds)} examples")
    print("  Loading dev split...")
    dev_ds = load_split("dev", processor)
    print(f"  Dev: {len(dev_ds)} examples")

    print("\n[4/5] Starting training...")

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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        eval_strategy="steps",
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        gradient_checkpointing=True,
        bf16=True,
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
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        compute_metrics=make_compute_metrics(processor),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    print("\n[5/5] Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    metrics = train_result.metrics
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Train runtime: {metrics.get('train_runtime', 0)/60:.1f} minutes")

    print("\n  Evaluating on dev set...")
    eval_metrics = trainer.evaluate()
    print(f"  Dev WER: {eval_metrics.get('eval_wer', 'N/A'):.4f}")
    print(f"  Dev CER: {eval_metrics.get('eval_cer', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
