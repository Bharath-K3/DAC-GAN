"""
Fast Gradient Sign Method (FGSM) Attacks
========================================
Implements targeted and untargeted variants of the FGSM attack.
"""

import torch
import torch.nn.functional as F


def get_logits(model, inputs):
    """
    Helper to extract logits from both HuggingFace transformers models 
    and standard PyTorch classifiers.
    """
    try:
        # Check if model supports keyword argument inputs (HuggingFace style)
        outputs = model(input_values=inputs)
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    except TypeError:
        # Standard PyTorch model call
        outputs = model(inputs)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def fgsm_attack(input_values, epsilon, data_grad):
    """
    Generates perturbed inputs using the Fast Gradient Sign Method (Untargeted).
    Formula: perturbed = input_values + epsilon * sign(data_grad)
    """
    sign_data_grad = data_grad.sign()
    perturbed_input = input_values + epsilon * sign_data_grad
    return perturbed_input


def fgsm_targeted_attack(input_values, epsilon, data_grad):
    """
    Generates perturbed inputs using the Fast Gradient Sign Method (Targeted).
    Formula: perturbed = input_values - epsilon * sign(data_grad)
    """
    sign_data_grad = data_grad.sign()
    perturbed_input = input_values - epsilon * sign_data_grad
    return perturbed_input


def run_fgsm_attack(model, loader, device, epsilon, targeted=False, target_id=None):
    """
    Evaluates the model under FGSM attack.
    If targeted is True, target_id must be provided (integer class index).
    
    Returns:
        accuracy_or_success_rate: Float percentage
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
        input_values.requires_grad = True
        
        logits = get_logits(model, input_values)
        
        if targeted:
            if target_id is None:
                raise ValueError("target_id must be specified for targeted attack")
            target_labels = torch.full_like(labels, target_id).to(device)
            loss = F.cross_entropy(logits, target_labels)
        else:
            loss = F.cross_entropy(logits, labels)
            
        model.zero_grad()
        loss.backward()
        
        data_grad = input_values.grad.data
        
        if targeted:
            perturbed_data = fgsm_targeted_attack(input_values, epsilon, data_grad)
        else:
            perturbed_data = fgsm_attack(input_values, epsilon, data_grad)
            
        with torch.no_grad():
            output_adv = get_logits(model, perturbed_data)
            preds_adv = torch.argmax(output_adv, dim=1)
            
        if targeted:
            success += (preds_adv == target_labels).sum().item()
        else:
            correct += (preds_adv == labels).sum().item()
            
        total += labels.size(0)
        
    if targeted:
        return (success / total) * 100
    else:
        return (correct / total) * 100
