import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch import autograd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Keep estimate_fisher as defined in the previous response.
def estimate_fisher(model, data_loader, sample_size, batch_size, device):
    model.eval() # Set model to evaluation mode

    fisher_accum = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use name directly here, clean later for buffer
            fisher_accum[name] = torch.zeros_like(param.data)

    loglikelihoods = []
    num_processed = 0
    params_with_grads = [p for p in model.parameters() if p.requires_grad] # Get params requiring grads ONCE

    for obs_batch, action_batch in data_loader:
        if num_processed >= sample_size:
            break

        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)

        # Get log probabilities of the actions taken
        _, log_prob, _, _ = model.get_action_and_value(obs_batch, action_batch.long())
        loglikelihoods.append(log_prob) # Collect log_prob for each item in the batch

        num_processed += obs_batch.size(0)

        if len(loglikelihoods) * batch_size >= sample_size: # Approximation based on batches
             break

    if not loglikelihoods:
        print("Warning: No loglikelihoods collected for Fisher estimation.")
        return {n.replace('.', '__'): torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}

    # Process loglikelihoods: calculate mean log-likelihood over the collected samples
    loglikelihood = torch.cat(loglikelihoods).mean()

    model.zero_grad()
    # Calculate gradients with rregards to parameters requiring gradients
    loglikelihood_grads = autograd.grad(loglikelihood, params_with_grads, retain_graph=False, allow_unused=True)

    # Accumulate squared gradients
    param_idx = 0
    for name, param in model.named_parameters():
         if param.requires_grad:
            if loglikelihood_grads[param_idx] is not None:
                fisher_accum[name] += loglikelihood_grads[param_idx].data.clone().pow(2)
            else:
                 print(f"Warning: Grad is None for param {name} during Fisher estimation. Skipping Fisher update.")
            param_idx += 1

    # Clean parameter names for buffer registration
    fisher_clean = {}
    for name, fisher_val in fisher_accum.items():
         fisher_clean[name.replace('.', '__')] = fisher_val # Use cleaned name as key

    model.train() # Set model back to training mode
    return fisher_clean


def consolidate_params(model, fisher_dict):
    """Stores the current parameters and Fisher estimates as buffers."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_clean = name.replace('.', '__')
            if n_clean in fisher_dict:
                # Detach and clone to ensure buffers are independent leaves
                mean_val = param.data.clone().detach()
                fisher_val = fisher_dict[n_clean].data.clone().detach()

                # Remove existing buffers if they exist, to avoid errors on re-consolidation
                if hasattr(model, f'{n_clean}_estimated_mean'):
                    delattr(model, f'{n_clean}_estimated_mean')
                if hasattr(model, f'{n_clean}_estimated_fisher'):
                    delattr(model, f'{n_clean}_estimated_fisher')

                model.register_buffer(f'{n_clean}_estimated_mean', mean_val)
                model.register_buffer(f'{n_clean}_estimated_fisher', fisher_val)
                # print(f"Consolidated: {n_clean}, Mean shape: {mean_val.shape}, Fisher shape: {fisher_val.shape}")
            else:
                 print(f"Warning: Fisher estimate not found for parameter {name} ({n_clean}). Skipping consolidation.")

def ewc_penalty(model, ewc_lambda, device):
    """Calculates the EWC penalty term based on registered buffers."""
    loss = torch.tensor(0.0).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_clean = name.replace('.', '__')
            mean_buffer_name = f'{n_clean}_estimated_mean'
            fisher_buffer_name = f'{n_clean}_estimated_fisher'

            if hasattr(model, mean_buffer_name) and hasattr(model, fisher_buffer_name):
                mean = getattr(model, mean_buffer_name)
                fisher = getattr(model, fisher_buffer_name)
                # Ensure buffers are on the correct device (should be automatic if model is)
                mean = mean.to(device)
                fisher = fisher.to(device)
                loss += (fisher * (param - mean)**2).sum()
    return (ewc_lambda / 2.0) * loss
