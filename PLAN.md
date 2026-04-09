# Mingrelian (xmf) STT/TTS - Comprehensive Training Plan

## Executive Summary

Build production-quality ASR (Speech-to-Text) and TTS (Text-to-Speech) models for
Mingrelian, a low-resource endangered Kartvelian language, using 12.6 hours of Common
Voice data and transfer learning from Georgian.

**Hardware:** NVIDIA RTX 5090 (32GB VRAM) — sufficient for ALL planned models.

---

## Dataset Profile

| Metric | Value | Implication |
|--------|-------|-------------|
| Total validated clips | 6,766 | Use ALL validated, not just official train split |
| Total duration | 11.5 hrs (validated) | Workable for fine-tuning, not from-scratch |
| Unique sentences | 989 (85% duplication) | CRITICAL: must split by sentence, not clip |
| Speakers | 83 (top 6 = 50%) | Imbalanced; good for ASR, challenging for TTS |
| Gender | 51% F, 5.5% M, 43% unknown | Strong female bias |
| Quality | 94.1% no downvotes | Very clean dataset |
| Dialects | Mostly Zugdidi, some Senakur-Martvili | Model may not generalize across dialects |
| Special chars | ჷ (schwa, 45.6%), ჸ (glottal stop) | Whisper tokenizer handles these correctly |

---

## Phase 0: Data Preparation (PREREQUISITE)

### 0.1 Custom Train/Dev/Test Split
- **DO NOT use official splits** (only 990 clips allocated; 5,777 unused)
- Split by SENTENCE to prevent data leakage (same sentence by different speakers)
- Allocate: ~80% train / 10% dev / 10% test BY UNIQUE SENTENCES
- Ensures no sentence appears in multiple splits
- Consider stratifying by dialect variant where known

### 0.2 Audio Preprocessing
1. Convert MP3 → 16kHz mono WAV (Whisper/MMS requirement)
2. Normalize audio levels (peak normalization to -1dB)
3. Filter outlier clips: 14 alignment outliers + 56 extreme speaking rate clips
4. Optional: apply VAD (Voice Activity Detection) to trim silence

### 0.3 Text Normalization
1. Replace typographic quotes „" with standard quotes or remove (24 clips)
2. Standardize em-dashes – (9 clips)
3. Verify all text uses Georgian Mkhedruli script
4. Build character vocabulary: 36 Georgian/Mingrelian chars + punctuation

---

## Phase 1: ASR (Speech-to-Text) — PRIMARY FOCUS

### Why ASR First
1. Common Voice data is designed for ASR
2. Multi-speaker diversity helps ASR generalization
3. Working ASR bootstraps TTS data validation
4. Proven fine-tuning pipelines exist

### Strategy: Three-track approach (quick → production)

#### Track A: MMS-1b-all Adapter (FAST BASELINE — hours)
- **Model:** `facebook/mms-1b-all` (1B params, adapter-based)
- **Transfer:** Initialize from Georgian (kat) adapter
- **Training:** Only ~2M adapter parameters trained
- **VRAM:** ~8-12 GB
- **Time:** ~1-2 hours
- **Purpose:** Quick baseline, validate data pipeline

#### Track B: Whisper large-v3 with LoRA (PRODUCTION — recommended)
- **Model:** `openai/whisper-large-v3` (1.55B params)
- **Transfer:** Georgian is natively supported (language token exists)
- **Fine-tuning:** LoRA rank 32, targeting attention + MLP layers
- **Optuna-LoRA:** Use Optuna (50 trials) to optimize rank, alpha, lr, dropout, target modules
  - Research shows 20.98% WER reduction vs baseline LoRA (paper: Optuna-LoRA for Whisper)
- **Config:**
  - `per_device_train_batch_size`: 4-8
  - `gradient_accumulation_steps`: 4
  - `learning_rate`: 1e-4 (LoRA, tune with Optuna)
  - `bf16`: True
  - `num_train_epochs`: 10-20
  - `warmup_steps`: 500
  - `optim`: adamw_torch
  - `evaluation_strategy`: steps (every 500)
- **VRAM:** ~8-12 GB
- **Time:** ~3-6 hours
- **Expected WER:** 20-30% (improving on Georgian baseline of 31.85%)
- **Inference boost:** Whisper-LM — integrate Mingrelian n-gram LM at inference for
  up to 51% improvement on in-distribution data (paper: whisper-lm-transformers)

#### Track C: Whisper large-v3 Full Fine-tune (IF NEEDED)
- Same model, but all parameters trainable
- **Config:**
  - `per_device_train_batch_size`: 2
  - `gradient_accumulation_steps`: 8
  - `learning_rate`: 1e-5
  - `bf16`: True
  - `gradient_checkpointing`: True
  - `optim`: adamw_bnb_8bit
- **VRAM:** ~24-30 GB
- **Time:** ~5-10 hours
- **Only pursue if LoRA quality is insufficient**

#### Alternative: Wav2Vec2-BERT 2.0 CTC
- **Model:** `facebook/w2v-bert-2.0` (600M params)
- 10-30x faster fine-tuning than Whisper
- Best for real-time inference applications
- CTC decoding = single-pass, faster than autoregressive

### ASR Evaluation
- Primary metric: **WER** (Word Error Rate) via `jiwer`
- Secondary: **CER** (Character Error Rate) — more informative for agglutinative languages
- Evaluate on held-out test set (sentence-aware split)
- Track per-dialect performance (Zugdidi vs Senakur-Martvili)
- Verify handling of ჷ and ჸ characters specifically

---

## Phase 2: TTS (Text-to-Speech) — SECONDARY

### Data Considerations for TTS
- TTS needs consistent single-speaker data
- **Best candidate: Speaker #1** — 829 clips, 80.1 min, female, 1.4% downvotes, good consistency
- **Cleanest candidate: Speaker #11** — 287 clips, 29.7 min, 0% downvotes
- **Only male option: Speaker #7** — 313 clips, 38.3 min, Zugdidi dialect
- Filter selected speaker's clips for quality

### Strategy: Two-track approach

#### Track A: MMS-VITS Fine-tune (FAST BASELINE — minutes)
- **Model:** MMS-VITS from Georgian checkpoint (~83M params)
- **Toolkit:** `ylacombe/finetune-hf-vits`
- **Data:** Single speaker subset
- **Time:** ~20 minutes
- **Purpose:** Quick baseline

#### Track B: F5-TTS with Lightweight Adapter (PRODUCTION)
- **Method:** F5-TTS-RO style adapter (paper: arxiv.org/abs/2512.12297)
  - Freeze ALL original F5-TTS weights
  - Add trainable character embeddings + ConvNeXt adapter module
  - Preserves voice cloning & existing language capabilities
- **Base model:** `SWivid/F5-TTS`
- **Georgian checkpoint:** `NMikka/F5-TTS-Georgian` (5.09% CER on Georgian)
- **Transfer:** Shared Mkhedruli script, add ჷ/ჸ/ჲ to character embeddings
- **Data:** Speaker #1 or #11 clips (aggressively quality-filtered)
- **Config:** lr=1e-4, ~40K steps, batch_size=16384 audio frames
- **VRAM:** ~24-32 GB (fits on 5090)
- **Time:** ~12-24 hours
- **Expected quality:** Natural-sounding Mingrelian speech with voice cloning

### TTS Evaluation
- MOS (Mean Opinion Score) via listening tests
- CER (synthesize → ASR → compare to original text)
- Speaker similarity (if targeting a specific speaker)
- Naturalness on out-of-domain sentences

---

## Phase 3: Enhancement & Iteration

### Data Augmentation for ASR
1. **SpecAugment:** Time/frequency masking during training (always use)
2. **Speed perturbation:** 0.9x, 1.0x, 1.1x — triples effective data
3. **Noise injection:** Add background noise at various SNRs (MUSAN/DNS noise datasets)
4. **Georgian co-training:** Mix Georgian Common Voice (~76h) with 5:1 Mingrelian weighting
5. **Curriculum learning:** Start with clean short utterances, progress to longer noisier ones
   (CBA-Whisper approach: ranked 2nd at Interspeech 2025 Speech Accessibility Challenge)

### Multi-Stage Transfer (RECOMMENDED for best results)
1. Start from Whisper large-v3 (multilingual pretrained)
2. Fine-tune on Georgian Common Voice (~76h) to adapt to Kartvelian phonology
3. Continue fine-tuning on Mingrelian (12h) — Georgian features transfer strongly
4. Research confirms phonetically related languages provide more transfer benefit

### Self-training Loop
1. Train initial ASR on validated data
2. Use ASR to transcribe invalidated/other clips (590 invalidated, 156 salvageable)
3. Filter by confidence threshold (keep top 70-80%)
4. Optional: use multi-ASR fusion (agree between Whisper + MMS) for better pseudo-labels
5. Re-train on combined real + pseudo-labeled data
6. Iterate 2-3 times — expect ~5-15% WER improvement

### Cross-lingual Augmentation
- Georgian Common Voice (~76 hours) as auxiliary training data
- Weight Mingrelian samples higher (e.g., 5:1 ratio)
- Shared vocabulary helps; different phonemes (ჷ) need Mingrelian-only data

---

## Execution Order

```
Day 1:
  [0] Data preparation (splits, audio preprocessing, text normalization)
  [1A] MMS adapter baseline (~2 hrs)
  [1B] Whisper LoRA fine-tune (~6 hrs)

Day 2:
  [1] Evaluate ASR results, iterate on hyperparameters
  [1C] Full fine-tune if LoRA insufficient
  [2A] MMS-VITS TTS baseline (~30 min)

Day 3-4:
  [2B] F5-TTS fine-tuning from Georgian checkpoint (~24-48 hrs)
  [3] Data augmentation experiments
  [3] Self-training loop if ASR quality allows

Day 5+:
  [*] Evaluation, iteration, optimization
  [*] Build inference pipeline / demo
```

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Only 989 unique sentences → limited vocabulary | High | Cross-lingual augmentation with Georgian; self-training on new text |
| Speaker imbalance (top 6 = 50%) | Medium | Balanced sampling during training; evaluation on diverse speakers |
| Dialect variation (ჷ schwa only in Zugdidi) | Medium | Track per-dialect metrics; ensure both dialects in train set |
| Male voice ASR weakness (only 5.5% male data) | Medium | Oversampling male speaker; evaluate male-specific WER |
| 85% sentence duplication → train/test leakage | Critical | SENTENCE-AWARE splitting (already planned) |
| Tokenizer unknown characters | Low | VERIFIED: Whisper handles ჷ, ჸ, ჲ correctly |

---

## Environment (Verified)

- GPU: NVIDIA RTX 5090, 32GB GDDR7
- Python: 3.13.12
- PyTorch: 2.10.0+cu128
- Key packages: transformers 4.57.1, datasets 4.5.0, peft 0.18.1, 
  bitsandbytes 0.49.2, librosa 0.11.0, evaluate 0.4.6, jiwer
- ffmpeg: 8.0.1
- All audio loading verified (MP3 → 16kHz WAV pipeline works)
