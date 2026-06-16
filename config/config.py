"""
DAC-GAN Configuration — All Hyperparameters and Paths
=====================================================
Modify the paths below to match your dataset location.
All other parameters match the Interspeech 2026 submission.
"""

import torch
import os

# ======================== Device ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================== Paths ========================
# Set URBANSOUND_ROOT via environment variable or modify this default:
URBANSOUND_ROOT = os.environ.get(
    "URBANSOUND_ROOT",
    r"E:\Datasets\UrbanSound8K\UrbanSound8K\UrbanSound8K"
)
AUDIO_DIR = os.path.join(URBANSOUND_ROOT, "audio")
CSV_PATH = os.path.join(URBANSOUND_ROOT, "metadata", "UrbanSound8K.csv")

CLASSIFIER_PATH = "dcase_urbansound8k_best_panns_model.pth"
GENERATOR_PATH = "dac_gan_urbansound8k_generator.pth"
OUTPUT_DIR = "dac_gan_urbansound8k_results"

# ======================== Audio ========================
NUM_CLASSES = 10
DAC_SR = 16000       # DAC 16kHz model native sample rate
PANNS_SR = 32000     # Classifier input sample rate (PANNs standard)
DURATION = 4         # Audio duration in seconds

# UrbanSound8K classes
URBANSOUND_CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]

# ======================== Fold Split ========================
TRAIN_FOLDS = [1, 2, 3, 6, 7, 8, 9, 10]
VAL_FOLD = 4
TEST_FOLD = 5

# ======================== Training ========================
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH = BATCH_SIZE * ACCUMULATION_STEPS  # 32
EPOCHS = 100
LEARNING_RATE = 5e-4
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 0.01

# ======================== Loss Weights ========================
LAMBDA_L2 = 10.0         # noise penalty weight
LAMBDA_MARGIN = 0.1      # Margin loss weight
MARGIN = 0.5             # Margin value (target logit must exceed runner-up by this)
TARGETS_PER_SAMPLE = 3   # Diverse target classes per training sample

# ======================== Perturbation Constraints ========================
DELTA_CLAMP = 2.0    # Generator output clamp ± (Vary from 0.1-2.0 depending on desired loudness)
Z_ADV_CLAMP = 5.0    # Adversarial latent clamp ±
AUDIO_CLAMP = 1.0    # Decoded audio clamp ± (standard audio range)

# ======================== EMA ========================
EMA_DECAY = 0.999

# ======================== Reproducibility ========================
SEED = 42
