
import numpy as np
import pickle
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, TransfoXLConfig, TransfoXLModel
from src.models.lightning_base_model import BaseModel
from src.models.randomTransformer import RandomTransformerUnembedding
from src.core import Config
import random

torch.manual_seed(42)
np.random.seed(42)

config = Config()
config.override("multi_sys_trace", True)
config.override("max_sys_trace", 25)
config.override("ny", 5)
config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2))  # 57
config.override("n_dims_out", 5)
config.override("n_positions", 251)

n_dims_in = config.n_dims_in   # 57
n_dims_out = config.n_dims_out # 5
n_positions = config.n_positions
n_embd = 128
n_layer = 12
n_head = 8

payload_flag_idx = n_dims_in - n_dims_out - 1 #57-5-1=51

model = RandomTransformerUnembedding(
    n_dims_in=n_dims_in,
    n_positions=n_positions,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    n_dims_out=n_dims_out
)
init_weights_path = 'initial_model_weights.pth'
if not os.path.exists(init_weights_path):
    torch.save(model.state_dict(), init_weights_path)

print(f"Model created successfully!")
print(f"Input dimension: {n_dims_in}")
print(f"Output dimension: {n_dims_out}")
print(f"Embedding dimension: {n_embd}")

# ---------------------------------------------------------------------
# TRAIN DATA
# ---------------------------------------------------------------------
real_data_path = 'src/DataRandomTransformer/train_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl'
print(f"Loading real data from {real_data_path}")
with open(real_data_path, 'rb') as f:
    data = pickle.load(f)

multi_sys_ys = data['multi_sys_ys']  # (1, 40000, 1, 251, 57)
N = np.prod(multi_sys_ys.shape[:-2])
seq_len = multi_sys_ys.shape[-2]
inputs = multi_sys_ys.reshape(N, seq_len, n_dims_in)

# original same-time payload
targets = inputs[..., -n_dims_out:]

device = "cpu"
model = model.to(device)

# cache hidden states
cache_path = 'cached_last_layer_activations_and_targets.npz'
if os.path.exists(cache_path):
    print(f"Loading cached activations and targets from {cache_path}")
    cache = np.load(cache_path)
    last_layer_activations = cache['last_layer_activations']
else:
    print(f"Processing activations and saving to {cache_path}")
    last_layer_activations = []
    with torch.no_grad():
        for i in range(len(inputs)):
            sample_input = torch.from_numpy(inputs[i:i+1]).float().to(device)
            _, interm = model.predict_step({"current": sample_input}, return_hidden=True)
            last_layer_activations.append(interm["hidden"].cpu().numpy())
    last_layer_activations = np.concatenate(last_layer_activations, axis=0)
    np.savez_compressed(cache_path, last_layer_activations=last_layer_activations, targets=targets)

print(f"Last layer activations shape: {last_layer_activations.shape}")

payloads = inputs[..., -n_dims_out:]   # shape: (N, T, 5)

# targets[t] = payload at t+1
targets = np.zeros_like(payloads)
targets[:, :-1, :] = payloads[:, 1:, :]   # shift left by 1

mask = np.ones((inputs.shape[0], inputs.shape[1]), dtype=bool)  # start True everywhere
mask[:, :-1] = (inputs[:, 1:, payload_flag_idx] == 0)           # True where next isn't payload
mask[:, -1] = True                                              # last always masked (no next)

print(f"Training mask: masked={mask.sum()}  unmasked={(~mask).sum()}")

# zero out invalid hidden states (optional; keeps code flow)
last_layer_activations_masked = last_layer_activations.copy()
last_layer_activations_masked[mask] = 0

# construct A,Y
N, T, d = last_layer_activations_masked.shape
A = last_layer_activations_masked.transpose(2, 0, 1).reshape(d, N*T)  # [d, N*T]
Y = targets.transpose(2, 0, 1).reshape(n_dims_out, N*T)               # [5, N*T]

# training set valid cols
unmasked_indices = ~mask.flatten()           # True => valid
A_unmasked = A[:, unmasked_indices]
Y_unmasked = Y[:, unmasked_indices]

w_opt_path = 'W_opt_masked_next.npy'
if os.path.exists(w_opt_path):
    print(f"Loading optimal pseudoinverse solution from {w_opt_path}")
    W_opt_masked = np.load(w_opt_path)
else:
    print(f"Computing and saving optimal pseudoinverse solution to {w_opt_path}")
    W_opt_masked = Y_unmasked @ np.linalg.pinv(A_unmasked)
    
    np.save(w_opt_path, W_opt_masked)
preds = W_opt_masked @ A_unmasked

#Testing haystacks 1, 2, and 5

haystack_lengths = [1, 2, 5]
for hay_len in haystack_lengths:
    val_cache_path = f'cached_val_last_layer_activations_and_targets_haystack_len_{hay_len}.npz'
    val_data_path = f'src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_{hay_len}.pkl'
    val_activations_pkl = f'last_layer_activations_val_haystack_len_{hay_len}.pkl'
    plot_filename = f'system_open_mse_bargraph_haystack_len_{hay_len}.png'

    # reload initial weights (backbone frozen so should match caches)
    model.load_state_dict(torch.load(init_weights_path))

    with open(val_data_path, 'rb') as f:
        orig_data = pickle.load(f)
    val_data = orig_data
    val_inputs = val_data['multi_sys_ys']  # (n_configs,1,n_traces,T,D)
    N_val = np.prod(val_inputs.shape[:-2])
    T_val = val_inputs.shape[-2]
    val_inputs = val_inputs.reshape(N_val, T_val, n_dims_in)
    print(val_inputs[0, 0, :])

    # original payloads
    val_targets = val_inputs[..., -n_dims_out:]

    # hidden cache
    if os.path.exists(val_cache_path):
        print(f"Loading cached validation activations from {val_cache_path}")
        val_cache = np.load(val_cache_path)
        val_last_layer_activations = val_cache['last_layer_activations']
        # ignore cached targets; we'll shift below  # <<< CHANGED >>>
        # ignore cached inputs; we just re-read above
    else:
        print(f"Processing validation activations and saving to {val_cache_path}")
        val_last_layer_activations = []
        with torch.no_grad():
            for i in range(len(val_inputs)):
                sample_input = torch.from_numpy(val_inputs[i:i+1]).float().to(device)
                _, interm = model.predict_step({"current": sample_input}, return_hidden=True)
                val_last_layer_activations.append(interm["hidden"].cpu().numpy())
        val_last_layer_activations = np.concatenate(val_last_layer_activations, axis=0)
        np.savez_compressed(val_cache_path,
                            last_layer_activations=val_last_layer_activations,
                            targets=val_targets,
                            inputs=val_inputs)

    N_val, T_val, d = val_last_layer_activations.shape

    # save activations for inspection
    activation_data = {
        'last_layer_activations': val_last_layer_activations,
        'targets': val_targets,
        'n_embd': n_embd,
        'n_dims_out': n_dims_out,
        'using_real_data': True,
        'haystack_length': hay_len
    }
    with open(val_activations_pkl, 'wb') as f:
        pickle.dump(activation_data, f)
    print(f"Saved last layer activations to '{val_activations_pkl}' for haystack_len={hay_len}")

    val_payloads = val_inputs[..., -n_dims_out:]
    val_targets = np.zeros_like(val_payloads)
    val_targets[:, :-1, :] = val_payloads[:, 1:, :]

# Mask
    val_mask = np.ones((val_inputs.shape[0], val_inputs.shape[1]), dtype=bool)
    val_mask[:, :-1] = (val_inputs[:, 1:, payload_flag_idx] == 0)
    val_mask[:, -1] = True

    print(f"\nMasking statistics (next-step) for haystack_len={hay_len}:")
    print(f"  Total validation entries: {val_mask.size}")
    print(f"  Number of masked entries: {np.sum(val_mask)}")
    print(f"  Number of unmasked entries: {np.sum(~val_mask)}")
    print(f"  Percentage masked: {100 * np.sum(val_mask) / val_mask.size:.2f}%")

    # zero invalid hidden states (optional; keep)
    val_last_layer_activations[val_mask] = 0

    # A,Y for validation (not used to fit, just for shape consistency if needed)
    A_val = val_last_layer_activations.transpose(2, 0, 1).reshape(d, N_val*T_val)
    Y_val = val_targets.transpose(2, 0, 1).reshape(n_dims_out, N_val*T_val)

    # predictions
    val_preds = np.einsum('od,ntd->nto', W_opt_masked, val_last_layer_activations)  # [N_val,T_val,5]

    # shapes for event slicing
    orig_multi_sys_ys_shape = orig_data['multi_sys_ys'].shape  # (n_configs,1,n_traces,T,D)
    n_configs, _, n_traces, t_val, _ = orig_multi_sys_ys_shape

    # per-sample SE
    sqerr_val_per_sample = np.sum((val_preds - val_targets) ** 2, axis=-1)  # (N_val,T_val)
    sqerr_val_per_sample_reshaped = sqerr_val_per_sample.reshape(n_configs, n_traces, t_val)

    # apply mask
    val_mask_reshaped = val_mask.reshape(n_configs, n_traces, t_val)
    sqerr_val_per_sample_masked = sqerr_val_per_sample_reshaped.copy()
    sqerr_val_per_sample_masked[val_mask_reshaped] = np.nan

    trace_medians_val = np.nanmedian(sqerr_val_per_sample_masked, axis=2)
    config_medians_val = np.nanmedian(trace_medians_val, axis=1)
    final_median_val = np.nanmedian(config_medians_val)
    print(f'Median of trace medians (validation): {final_median_val:.6f}')

    # per-event errors -------------------------------------------------
    event_types = [f"{k}_after_initial" for k in [1,2,3,7,8]] + [f"{k}_after_final" for k in [1,2,3,7,8]]
    n_events = len(event_types)
    event_errors = np.full((n_configs, n_traces, n_events), np.nan)

    for system in range(config.max_sys_trace):
        open_idx = 2 * system  # open token dim
        if open_idx >= n_dims_in:
            break

        for config_i in range(n_configs):
            for trace_i in range(n_traces):
                n = config_i * n_traces + trace_i
                if n >= N_val:
                    continue
                open_events = np.where(val_inputs[n, :, open_idx] != 0)[0]
                if len(open_events) < 2:
                    continue

                p_init = open_events[0]
                p_final = open_events[-1]

                # After initial: pred index = p_init + (k-1)
                for k_idx, k in enumerate([1, 2, 3, 7, 8]):           
                    idx = p_init + (k - 1)                           
                    if idx < T_val and not val_mask[n, idx]:
                        pred = val_preds[n, idx]
                        target = val_targets[n, idx]
                        sqerr = np.sum((pred - target) ** 2)
                        event_errors[config_i, trace_i, k_idx] = sqerr

                # After final: pred index = p_final + (k-1)
                for k_idx, k in enumerate([1, 2, 3, 7, 8]):           
                    idx = p_final + (k - 1)                        
                    if idx < T_val and not val_mask[n, idx]:
                        pred = val_preds[n, idx]
                        target = val_targets[n, idx]
                        sqerr = np.sum((pred - target) ** 2)
                        event_errors[config_i, trace_i, k_idx + 5] = sqerr

    # medians over traces, configs
    median_over_traces = np.nanmedian(event_errors, axis=1)  # (n_configs,n_events)
    final_median_events = np.nanmedian(median_over_traces, axis=0)  # (n_events,)

    labels = [
        '1_after_initial', '2_after_initial', '3_after_initial', '7_after_initial', '8_after_initial',
        '1_after_final', '2_after_final', '3_after_final', '7_after_final', '8_after_final'
    ]
    rt_medians = final_median_events

    # GPT2 data
    gpt2_data = {
        1: [1, 0.95, 0.9, 0.007, 0.0065, 0.008, 0.006, 0.005, 0.004, 0.002],
        2: [1, 0.95, 0.9, 0.008, 0.006, 0.015, 0.0075, 0.007, 0.005, 0.005],
        5: [1, 0.95, 0.9, 0.009, 0.006, 0.01, 0.008, 0.009, 0.006, 0.0055],
    }
    gpt2_medians = gpt2_data[hay_len]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12,5))
    plt.bar(x - width/2, rt_medians, width, label='RandomTransformer')
    plt.bar(x + width/2, gpt2_medians, width, label='GPT2')
    plt.yscale('log')
    plt.ylabel('Median MSE (log scale)')
    plt.title(f'Median MSE (next-step) after initial and final system open (haystack_len={hay_len})')
    plt.xticks(x, labels, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()
    print(f"Bar graph saved as '{plot_filename}' for haystack_len={hay_len}")

    # Additional plot: RandomTransformer only, linear scale
    plt.figure(figsize=(10,4))
    plt.bar(x, rt_medians, width=0.5, label='RandomTransformer', color='C0')
    plt.ylabel('Median MSE')
    plt.title(f'Median MSE (next-step) after initial and final system open (RandomTransformer only, haystack_len={hay_len})')
    plt.xticks(x, labels, rotation=30)
    plt.legend()
    plt.ylim(0.7, 1)
    plt.tight_layout()
    plt.savefig(f'system_open_mse_bargraph_randomtransformer_only_haystack_len_{hay_len}.png')
    plt.show()
    print(f"RandomTransformer-only bar graph saved as 'system_open_mse_bargraph_randomtransformer_only_haystack_len_{hay_len}.png' for haystack_len={hay_len}")