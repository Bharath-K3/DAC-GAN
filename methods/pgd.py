"""
Projected Gradient Descent (PGD) Attacks
========================================
Implements targeted and untargeted variants of the PGD attack.
"""

import torch
import torch.nn.functional as F


def get_logits(model, inputs):
    """
    Helper to extract logits from both HuggingFace transformers models 
    and standard PyTorch classifiers.
    """
    try:
        outputs = model(input_values=inputs)
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    except TypeError:
        outputs = model(inputs)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def pgd_attack(model, input_values, labels_or_targets, epsilon, alpha, num_steps, random_start=True, targeted=False):
    """
    Performs Projected Gradient Descent (PGD) attack (Targeted or Untargeted).
    
    Args:
        model: The model to attack.
        input_values: Clean input tensor.
        labels_or_targets: True labels (for untargeted) or target labels (for targeted).
        epsilon: Perturbation bound (L-inf).
        alpha: Step size.
        num_steps: Iterations.
        random_start: If True, initialize with random noise.
        targeted: Set to True for a targeted attack.
    """
    adv_input = input_values.clone().detach().to(input_values.device)
    
    if random_start:
        noise = torch.empty_like(adv_input).uniform_(-epsilon, epsilon)
        adv_input = adv_input + noise
        adv_input = torch.clamp(adv_input, min=input_values - epsilon, max=input_values + epsilon)

    for _ in range(num_steps):
        adv_input.requires_grad = True
        logits = get_logits(model, adv_input)
        
        loss = F.cross_entropy(logits, labels_or_targets)
        model.zero_grad()
        loss.backward()
        
        grad = adv_input.grad.data
        if targeted:
            # Minimize loss w.r.t target class (gradient descent)
            adv_input = adv_input.detach() - alpha * grad.sign()
        else:
            # Maximize loss w.r.t true class (gradient ascent)
            adv_input = adv_input.detach() + alpha * grad.sign()
            
        delta = adv_input - input_values
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        adv_input = input_values + delta
        
    return adv_input.detach()


def run_pgd_attack(model, loader, device, epsilon, alpha, num_steps, targeted=False, target_id=None, random_start=True):
    """
    Evaluates the model under PGD attack.
    """
    model.eval()
    correct = 0
    total = 0
    success = 0
    
    for inputs, labels in loader:
        if isinstance(inputs, dict):
            input_values = inputs["input_values"].to(device)
        else:
            input_values = inputs.to(device)
            
        labels = labels.to(device)
        
        if targeted:
            if target_id is None:
                raise ValueError("target_id must be specified for targeted attack")
            target_labels = torch.full_like(labels, target_id).to(device)
            
            perturbed_inputs = pgd_attack(
                model, input_values, target_labels, 
                epsilon=epsilon, alpha=alpha, num_steps=num_steps, 
                random_start=random_start, targeted=True
            )
        else:
            perturbed_inputs = pgd_attack(
                model, input_values, labels, 
                epsilon=epsilon, alpha=alpha, num_steps=num_steps, 
                random_start=random_start, targeted=False
            )
            
        with torch.no_grad():
            outputs = get_logits(model, perturbed_inputs)
            preds = torch.argmax(outputs, dim=1)
            
        if targeted:
            success += (preds == target_labels).sum().item()
        else:
            correct += (preds == labels).sum().item()
            
        total += labels.size(0)
        
    if targeted:
        return (success / total) * 100
    else:
        return (correct / total) * 100
