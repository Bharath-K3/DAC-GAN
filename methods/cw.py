"""
Carlini & Wagner (CW) L2 Attacks
================================
Implements targeted and untargeted variants of the Carlini & Wagner L2 attack.
"""

import torch
import torch.nn as nn
import torch.optim as optim


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


def cw_l2_attack(model, input_values, labels_or_targets, c=1.0, kappa=0, learning_rate=0.01, num_steps=50, targeted=False):
    """
    Performs Carlini & Wagner L2 Attack (Targeted or Untargeted).
    
    Args:
        model: The model to attack.
        input_values: Clean input tensor.
        labels_or_targets: True labels (for untargeted) or target labels (for targeted).
        c: Trade-off parameter balancing distortion and success.
        kappa: Margin parameter (confidence).
        learning_rate: Optimizer learning rate.
        num_steps: Number of optimization iterations.
        targeted: Set to True for a targeted attack.
    """
    device = input_values.device
    batch_size = input_values.shape[0]
    delta = torch.zeros_like(input_values, requires_grad=True).to(device)
    optimizer = optim.Adam([delta], lr=learning_rate)

    for step in range(num_steps):
        adv_inputs = input_values + delta
        logits = get_logits(model, adv_inputs)
        
        if targeted:
            # Targeted optimization: minimize max_{i!=t} f_i - f_t
            target_logit = torch.gather(logits, 1, labels_or_targets.unsqueeze(1)).squeeze(1)
            tmp_logits = logits.clone()
            tmp_logits[range(batch_size), labels_or_targets] = -float('inf')
            max_other_logit, _ = torch.max(tmp_logits, dim=1)
            f_loss = torch.clamp(max_other_logit - target_logit, min=-kappa)
        else:
            # Untargeted optimization: minimize f_y - max_{i!=y} f_i
            real_logit = torch.gather(logits, 1, labels_or_targets.unsqueeze(1)).squeeze(1)
            tmp_logits = logits.clone()
            tmp_logits[range(batch_size), labels_or_targets] = -float('inf')
            max_other_logit, _ = torch.max(tmp_logits, dim=1)
            f_loss = torch.clamp(real_logit - max_other_logit, min=-kappa)
            
        l2_loss = torch.sum(delta.view(batch_size, -1) ** 2, dim=1)
        cost = torch.mean(l2_loss + c * f_loss)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    return (input_values + delta).detach()


def run_cw_attack(model, loader, device, c=1.0, kappa=0, learning_rate=0.01, num_steps=50, targeted=False, target_id=None):
    """
    Evaluates the model under CW L2 attack.
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
                
            perturbed_inputs = cw_l2_attack(
                model, input_values, target_labels, 
                c=c, kappa=kappa, learning_rate=learning_rate, 
                num_steps=num_steps, targeted=True
            )
        else:
            perturbed_inputs = cw_l2_attack(
                model, input_values, labels, 
                c=c, kappa=kappa, learning_rate=learning_rate, 
                num_steps=num_steps, targeted=False
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
