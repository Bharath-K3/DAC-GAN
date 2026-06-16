"""
DAC-GAN Training Script — UrbanSound8K
========================================
Trains a generator to produce targeted adversarial perturbations
in the latent space of the Descript Audio Codec (DAC).

The adversarial audio is decoded from z_adv = z_clean + G(z_clean, target)
and fools a frozen PANNs Cnn14 classifier into predicting the target class.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import numpy as np
import soundfile as sf
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import warnings

warnings.filterwarnings('ignore')

try:
    import dac
except ImportError:
    raise ImportError("Install descript-audio-codec: pip install descript-audio-codec")

from config.config import (
    DEVICE, CSV_PATH, CLASSIFIER_PATH, GENERATOR_PATH, OUTPUT_DIR,
    NUM_CLASSES, DAC_SR, PANNS_SR, DURATION, URBANSOUND_CLASSES,
    TRAIN_FOLDS, TEST_FOLD, BATCH_SIZE, ACCUMULATION_STEPS,
    EFFECTIVE_BATCH, EPOCHS, LEARNING_RATE, WARMUP_EPOCHS, WEIGHT_DECAY,
    LAMBDA_L2, LAMBDA_MARGIN, MARGIN, TARGETS_PER_SAMPLE,
    DELTA_CLAMP, Z_ADV_CLAMP, AUDIO_CLAMP, EMA_DECAY, SEED,
)
from models.models import StrongGenerator, Cnn14
from utils.dataset import UrbanSound8kDataset
from utils.utils import (
    EMA, margin_loss, get_warmup_cosine_scheduler,
    get_diverse_targets, set_seed, decode_audio,
)

INV_CLASS_MAP = {idx: name for idx, name in enumerate(URBANSOUND_CLASSES)}


# =====================================================================
# Training
# =====================================================================

def train_generator():
    """Train the adversarial perturbation generator."""

    set_seed(SEED)

    print("=" * 80)
    print("DAC-GAN ADVERSARIAL ATTACK TRAINING — URBANSOUND8K")
    print("=" * 80)
    print(f"Device:           {DEVICE}")
    print(f"Batch Size:       {BATCH_SIZE} × {ACCUMULATION_STEPS} = {EFFECTIVE_BATCH}")
    print(f"Learning Rate:    {LEARNING_RATE}")
    print(f"Epochs:           {EPOCHS}")
    print(f"Targets/Sample:   {TARGETS_PER_SAMPLE}")
    print(f"Delta Clamp:      ±{DELTA_CLAMP}")
    print(f"z_adv Clamp:      ±{Z_ADV_CLAMP}")
    print(f"Audio Clamp:      ±{AUDIO_CLAMP}")
    print(f"λ_L2 (reconstruction): {LAMBDA_L2}, λ_margin: {LAMBDA_MARGIN}, margin: {MARGIN}")

    torch.cuda.empty_cache()

    # ---- Dataset ----
    df = pd.read_csv(CSV_PATH)
    df_train = df[df.fold.isin(TRAIN_FOLDS)]
    dataset = UrbanSound8kDataset(df_train, target_sr=DAC_SR, duration=DURATION)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    print(f"Training samples: {len(dataset)}")

    # ---- DAC Codec (frozen) ----
    print("Loading DAC 16kHz model...")
    dac_path = dac.utils.download(model_type="16khz")
    dac_model = dac.DAC.load(dac_path).to(DEVICE)
    dac_model.eval()
    for p in dac_model.parameters():
        p.requires_grad = False

    # ---- PANNs Classifier (frozen) ----
    print("Loading PANNs Cnn14 classifier...")
    classifier = Cnn14(
        sample_rate=PANNS_SR, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=NUM_CLASSES,
    ).to(DEVICE)

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Classifier weights not found: {CLASSIFIER_PATH}")
    state_dict = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    classifier.load_state_dict(state_dict, strict=False)
    print(f"Loaded classifier from {CLASSIFIER_PATH}")
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # ---- Latent dimensions ----
    with torch.no_grad():
        dummy = torch.randn(1, 1, DAC_SR * DURATION).to(DEVICE)
        z, _, _, _, _ = dac_model.encode(dummy)
        latent_dim = z.shape[1]
    print(f"DAC Latent Dim: {latent_dim}")

    # ---- Generator + EMA + Optimiser ----
    generator = StrongGenerator(
        latent_dim, NUM_CLASSES, delta_clamp=DELTA_CLAMP,
    ).to(DEVICE)
    ema = EMA(generator, decay=EMA_DECAY)

    optimizer = optim.AdamW(
        generator.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_warmup_cosine_scheduler(
        optimizer, WARMUP_EPOCHS, EPOCHS, len(loader),
    )

    best_asr = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Training Loop ----
    for epoch in range(EPOCHS):
        generator.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        epoch_loss = 0.0
        epoch_success = 0
        epoch_total = 0
        nan_skipped = 0

        optimizer.zero_grad()

        for i, (waveform, true_labels) in enumerate(pbar):
            waveform = waveform.to(DEVICE)
            true_labels = true_labels.to(DEVICE)
            wav_input = waveform.unsqueeze(1)       # (B, 1, T)
            bs = waveform.shape[0]

            # Encode clean audio (frozen DAC encoder)
            with torch.no_grad():
                z_clean, _, _, _, _ = dac_model.encode(wav_input)

            if torch.isnan(z_clean).any():
                nan_skipped += 1
                continue

            # Decode clean audio once for audio-domain reconstruction loss
            with torch.no_grad():
                clean_audio = decode_audio(dac_model, z_clean)
                clean_audio = torch.clamp(clean_audio, -AUDIO_CLAMP, AUDIO_CLAMP)

            # Sample diverse target classes
            diverse_targets = get_diverse_targets(
                true_labels, num_classes=NUM_CLASSES,
                num_targets=TARGETS_PER_SAMPLE,
            )

            batch_loss = 0.0
            batch_success = 0
            batch_total = 0

            for t_idx in range(TARGETS_PER_SAMPLE):
                targets = torch.tensor(
                    [diverse_targets[b][t_idx] for b in range(bs)],
                ).to(DEVICE)

                # Forward: generate perturbation
                delta = generator(z_clean, targets)

                if torch.isnan(delta).any():
                    nan_skipped += 1
                    continue

                # Apply perturbation
                z_adv = z_clean + delta
                z_adv = torch.clamp(z_adv, -Z_ADV_CLAMP, Z_ADV_CLAMP)

                # Decode adversarial audio
                adv_audio = decode_audio(dac_model, z_adv)
                adv_audio = torch.clamp(adv_audio, -AUDIO_CLAMP, AUDIO_CLAMP)

                # Upsample for PANNs classifier if necessary
                if DAC_SR != PANNS_SR:
                    adv_audio_clf = torchaudio.functional.resample(
                        adv_audio, orig_freq=DAC_SR, new_freq=PANNS_SR
                    )
                else:
                    adv_audio_clf = adv_audio

                # Classify
                logits, _ = classifier(adv_audio_clf.squeeze(1))

                if torch.isnan(logits).any():
                    nan_skipped += 1
                    continue

                # ---- Loss ----
                loss_ce = F.cross_entropy(logits, targets)
                loss_margin = margin_loss(logits, targets, margin=MARGIN)

                # Audio-domain reconstruction loss (directly penalises decoded distortion)
                min_len = min(clean_audio.shape[-1], adv_audio.shape[-1])
                noise = adv_audio[:, :, :min_len] - clean_audio[:, :, :min_len]
                signal_power = torch.mean(clean_audio[:, :, :min_len] ** 2)
                noise_power = torch.mean(noise ** 2)
                loss_distortion = noise_power / (signal_power + 1e-8)

                loss = (loss_ce
                        + LAMBDA_MARGIN * loss_margin
                        + LAMBDA_L2 * loss_distortion)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_skipped += 1
                    continue

                batch_loss += loss.item()

                # Backward (scaled by accumulation & targets)
                (loss / (ACCUMULATION_STEPS * TARGETS_PER_SAMPLE)).backward()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    batch_success += (preds == targets).sum().item()
                    batch_total += bs

            # ---- Gradient step ----
            if (i + 1) % ACCUMULATION_STEPS == 0:
                nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            scheduler.step()

            epoch_loss += batch_loss
            epoch_success += batch_success
            epoch_total += batch_total

            pbar.set_postfix({
                'L': f"{batch_loss:.2f}",
                'S': f"{batch_success}/{batch_total}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Flush remaining accumulated gradients
        if len(loader) % ACCUMULATION_STEPS != 0:
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        epoch_asr = epoch_success / max(1, epoch_total) * 100
        print(f"Epoch {epoch + 1} — "
              f"ASR: {epoch_asr:.1f}%, "
              f"Loss: {epoch_loss:.2f}, "
              f"NaN: {nan_skipped}")

        if epoch_asr > best_asr:
            best_asr = epoch_asr
            torch.save({
                'generator': generator.state_dict(),
                'ema': ema.shadow,
                'epoch': epoch + 1,
                'asr': best_asr,
                'config': {
                    'latent_dim': latent_dim,
                    'num_classes': NUM_CLASSES,
                    'delta_clamp': DELTA_CLAMP,
                    'z_adv_clamp': Z_ADV_CLAMP,
                },
            }, GENERATOR_PATH)
            print(f"  → Saved best model (ASR: {best_asr:.1f}%)")

        torch.cuda.empty_cache()

    print(f"\nTraining Complete! Best ASR: {best_asr:.1f}%")
    print(f"Saved to {GENERATOR_PATH}")

    return generator, ema, dac_model, classifier


# =====================================================================
# Quick Post-Training Evaluation
# =====================================================================

def quick_evaluate(generator, ema, dac_model, classifier, max_samples=50):
    """Run a quick targeted evaluation on a subset of the test fold."""

    print("\n" + "=" * 80)
    print("QUICK EVALUATION — URBANSOUND8K (FOLD 5)")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    df_test = df[df.fold == TEST_FOLD]
    dataset = UrbanSound8kDataset(df_test, target_sr=DAC_SR, duration=DURATION)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    generator.eval()
    ema.apply_shadow()

    results = []

    header = (f"{'#':<4} | {'Source':<20} | {'Target':<20} | "
              f"{'Res':<6} | {'Conf':<6}")
    print(header)
    print("-" * 70)

    with torch.no_grad():
        for idx, (waveform, label) in enumerate(loader):
            if idx >= max_samples:
                break

            wav_input = waveform.to(DEVICE).unsqueeze(1)
            source_label = label.item()

            # Skip samples the classifier already misclassifies
            if DAC_SR != PANNS_SR:
                wav_input_clf = torchaudio.functional.resample(
                    wav_input, orig_freq=DAC_SR, new_freq=PANNS_SR
                )
            else:
                wav_input_clf = wav_input

            clean_logits, _ = classifier(wav_input_clf.squeeze(1))
            if torch.argmax(clean_logits).item() != source_label:
                continue

            # Encode and decode clean
            z_clean, _, _, _, _ = dac_model.encode(wav_input)
            if torch.isnan(z_clean).any():
                continue

            clean_decoded = decode_audio(dac_model, z_clean)
            clean_decoded = torch.clamp(clean_decoded, -AUDIO_CLAMP, AUDIO_CLAMP)

            # Attack 5 random target classes
            possible = [c for c in range(NUM_CLASSES) if c != source_label]
            targets = random.sample(possible, min(5, len(possible)))

            for target in targets:
                t_tensor = torch.tensor([target]).to(DEVICE)
                delta = generator(z_clean, t_tensor)
                if torch.isnan(delta).any():
                    continue

                z_adv = z_clean + delta
                z_adv = torch.clamp(z_adv, -Z_ADV_CLAMP, Z_ADV_CLAMP)

                adv_audio = decode_audio(dac_model, z_adv)
                adv_audio = torch.clamp(adv_audio, -AUDIO_CLAMP, AUDIO_CLAMP)

                if DAC_SR != PANNS_SR:
                    adv_audio_clf = torchaudio.functional.resample(
                        adv_audio, orig_freq=DAC_SR, new_freq=PANNS_SR
                    )
                else:
                    adv_audio_clf = adv_audio

                logits, _ = classifier(adv_audio_clf.squeeze(1))
                if torch.isnan(logits).any():
                    continue

                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits).item()
                conf = probs[0, target].item()
                success = (pred == target)

                min_len = min(clean_decoded.shape[-1], adv_audio.shape[-1])
                adv_np = adv_audio[:, :, :min_len].squeeze().cpu().numpy()

                if np.isnan(adv_np).any():
                    continue

                if success:
                    sf.write(
                        os.path.join(OUTPUT_DIR,
                                     f"{idx}_{source_label}_to_{target}.wav"),
                        adv_np, DAC_SR,
                    )

                status = "OK" if success else "FAIL"
                print(f"{idx:<4} | {INV_CLASS_MAP[source_label]:<20} | "
                      f"{INV_CLASS_MAP[target]:<20} | {status:<6} | "
                      f"{conf:<.2f}")

                results.append({
                    'source': source_label, 'target': target,
                    'success': success, 'confidence': conf,
                })

    ema.restore()

    if results:
        df_res = pd.DataFrame(results)
        print("-" * 70)
        print(f"Total Attacks:  {len(df_res)}")
        print(f"Targeted ASR:   {df_res['success'].mean() * 100:.1f}%")
        print(f"Avg Confidence: {df_res['confidence'].mean():.4f}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    gen, ema_obj, dac_m, clf = train_generator()
    quick_evaluate(gen, ema_obj, dac_m, clf)
