"""
Utility Functions for DAC-GAN Adversarial Attack
=================================================
EMA, loss functions, LR scheduler, target sampling, seeding.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random


# =====================================================================
# Exponential Moving Average
# =====================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains shadow copies of all trainable parameters, updated each
    step as:  shadow = decay × shadow + (1 − decay) × param

    Usage:
        ema = EMA(model, decay=0.999)
        # After each optimizer.step():
        ema.update()
        # For evaluation:
        ema.apply_shadow()
        # ... evaluate ...
        ema.restore()
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data \
                          + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """Replace model parameters with EMA shadow (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# =====================================================================
# Loss Functions
# =====================================================================

def margin_loss(logits, targets, margin=1.0):
    """
    Margin loss that penalises when the target-class logit does not
    exceed the best non-target logit by at least `margin`.

    L = mean( ReLU( max_other − target_logit + margin ) )
    """
    B = logits.shape[0]
    device = logits.device

    target_logits = logits[torch.arange(B, device=device), targets]

    other_logits = logits.clone()
    other_logits[torch.arange(B, device=device), targets] = float('-inf')
    max_other = other_logits.max(dim=1)[0]

    violations = F.relu(max_other - target_logits + margin)
    return violations.mean()


# =====================================================================
# Learning Rate Scheduler
# =====================================================================

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                steps_per_epoch):
    """
    Linear warmup followed by cosine annealing to zero.

    Args:
        optimizer:       PyTorch optimizer
        warmup_epochs:   Number of warmup epochs
        total_epochs:    Total training epochs
        steps_per_epoch: Number of batches per epoch
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Target Sampling
# =====================================================================

def get_diverse_targets(source_labels, num_classes=10, num_targets=3):
    """
    For each source label, randomly sample `num_targets` distinct
    non-source classes.

    Args:
        source_labels: Tensor of shape (B,) with integer class labels
        num_classes:   Total number of classes
        num_targets:   How many targets per source sample

    Returns:
        List of lists: targets[b] = [t1, t2, ..., t_num_targets]
    """
    all_targets = []
    for label in source_labels.tolist():
        possible = [i for i in range(num_classes) if i != label]
        chosen = random.sample(possible, min(num_targets, len(possible)))
        all_targets.append(chosen)
    return all_targets


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# DAC Decoder Helper
# =====================================================================

def decode_audio(dac_model, z):
    """Decode DAC latent codes to audio, handling API differences."""
    if hasattr(dac_model, 'decoder'):
        return dac_model.decoder(z)
    return dac_model.decode(z)
