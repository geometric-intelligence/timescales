# Gradient Statistics Tracking

The `GradientStatisticsCallback` tracks gradient statistics throughout training to help diagnose training issues like vanishing/exploding gradients.

## What is "Gradient Variance"?

When researchers track "gradient variance," they mean:

**The variance of gradient elements treated as a scalar distribution.**

Specifically:
1. After `loss.backward()`, collect all gradients: `[∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]`
2. Flatten all gradient tensors into a single 1D vector
3. Compute `Var([∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ])`

This gives a **single scalar** capturing the "spread" of gradient magnitudes across all parameters.

**Note:** This is NOT:
- ❌ Variance across different batches
- ❌ Variance across time steps (BPTT just accumulates gradients)
- ✅ It IS variance across all gradient elements at a given training step

## Features

### Global Statistics
- **Gradient Variance**: `Var(all gradient elements)` - spread of gradient magnitudes
- **Gradient Norm**: L2 norm `||∇L||₂` - overall gradient magnitude
- **Gradient Mean**: Mean of all gradient elements
- **Gradient Max/Min**: Range of gradient values

### Per-Weight-Matrix Statistics (optional)
For RNNs, tracks separate statistics for:
- **W_in**: Input-to-hidden weights (and their gradients)
- **W_rec**: Recurrent (hidden-to-hidden) weights
- **W_out**: Output/readout weights
- **W_h_init**: Initial state encoder
- **biases**: All bias terms

## Usage

### In Config File

Add these parameters to your YAML config (e.g., `configs/mts.yaml`):

```yaml
# Gradient tracking
grad_log_every_n_steps: 100  # Log gradient stats every N training steps
grad_track_per_weight_matrix: true  # Track W_in, W_rec, W_out separately
```

### Defaults
- `grad_log_every_n_steps`: 100 (log every 100 training steps)
- `grad_track_per_weight_matrix`: true (track each weight matrix separately)

## Outputs

### 1. Real-time Logging (WandB)

Gradient statistics are logged to WandB during training:

**Global metrics** (across all gradient elements):
- `train/grad_variance` - Variance of all ∂L/∂θ values
- `train/grad_norm` - L2 norm ||∇L||₂
- `train/grad_mean` - Mean of all gradient elements  
- `train/grad_max` - Maximum gradient element
- `train/grad_min` - Minimum gradient element

**Per-weight-matrix metrics** (if enabled):
- `train/grad_variance/W_in` - Variance of input weight gradients
- `train/grad_variance/W_rec` - Variance of recurrent weight gradients
- `train/grad_variance/W_out` - Variance of output weight gradients
- `train/grad_norm/W_in`, `W_rec`, `W_out` - Norms per matrix
- `train/grad_mean/W_in`, `W_rec`, `W_out` - Means per matrix

### 2. JSON Files

Statistics are saved to the run directory:

- **`gradient_statistics.json`**: Global statistics (variance across ALL gradient elements)
  ```json
  {
    "step": [100, 200, 300, ...],
    "epoch": [0, 0, 0, ...],
    "grad_variance": [0.0012, 0.0009, ...],  // Var(all ∂L/∂θ)
    "grad_mean": [0.0001, -0.0002, ...],
    "grad_norm": [1.23, 1.18, ...],
    "grad_max": [0.15, 0.14, ...],
    "grad_min": [-0.12, -0.11, ...]
  }
  ```

- **`gradient_statistics_weight_matrices.json`**: Per-weight-matrix stats (if enabled)
  ```json
  {
    "W_in": {
      "step": [100, 200, ...],
      "variance": [0.001, 0.0009, ...],  // Var of W_in gradients only
      "norm": [0.45, 0.43, ...],
      "mean": [0.0001, -0.0001, ...]
    },
    "W_rec": {...},
    "W_out": {...},
    "biases": {...}
  }
  ```

### 3. Summary Plot

At the end of training, a summary plot is generated: `gradient_statistics.png`

The plot shows 4 subplots:
1. **Gradient Variance** over time (log scale)
2. **Gradient Norm** over time (log scale)
3. **Absolute Gradient Mean** over time (log scale)
4. **Gradient Range** (max and abs(min)) over time (log scale)

## Example: Analyzing Training Issues

### Vanishing Gradients
- **Symptoms**: Gradient norm/variance decreasing over time, approaching zero
- **Indicators**: `grad_norm` < 1e-6, `grad_variance` decreasing rapidly
- **Solutions**: 
  - Adjust learning rate
  - Use gradient clipping
  - Change RNN timescales (for MultiTimescaleRNN)
  - Check initialization

### Exploding Gradients
- **Symptoms**: Gradient norm/variance increasing exponentially
- **Indicators**: `grad_norm` > 10, `grad_max` growing unbounded
- **Solutions**:
  - Add gradient clipping: `trainer = Trainer(..., gradient_clip_val=1.0)`
  - Lower learning rate
  - Adjust RNN timescales

### Weight-Matrix-specific Issues
If per-matrix tracking is enabled, you can identify which weight matrices have gradient problems:
- Check `gradient_statistics_weight_matrices.json`
- Look for matrices with unusually high/low variance
- **Common patterns:**
  - W_rec variance >> W_in variance: Recurrent dynamics dominating
  - W_in variance very low: Input not affecting hidden state much
  - W_out variance >> others: Output layer getting large updates

## Performance Considerations

### Memory & Speed
- **Global tracking**: Minimal overhead (~1% slowdown)
- **Per-matrix tracking**: Moderate overhead (~5-10% slowdown)
- CPU operations are used to minimize GPU memory impact

### Recommendations
- For large models (>10M parameters): Set `grad_log_every_n_steps: 500` or higher
- For debugging: Enable `grad_track_per_weight_matrix: true`
- For production runs: Consider `grad_track_per_weight_matrix: false` for speed
- Logging frequency: `100` is good for most cases; increase to `500-1000` for very long runs

## Example Analysis Script

```python
import json
import matplotlib.pyplot as plt

# Load gradient statistics
with open('path/to/run/gradient_statistics.json', 'r') as f:
    stats = json.load(f)

# Plot gradient variance
plt.figure(figsize=(10, 6))
plt.plot(stats['step'], stats['grad_variance'])
plt.xlabel('Training Step')
plt.ylabel('Gradient Variance')
plt.yscale('log')
plt.title('Gradient Variance Throughout Training')
plt.grid(True, alpha=0.3)
plt.show()

# Check for gradient issues
final_norm = stats['grad_norm'][-1]
if final_norm < 1e-6:
    print("⚠️  Warning: Possible vanishing gradients!")
elif final_norm > 10:
    print("⚠️  Warning: Possible exploding gradients!")
else:
    print("✓ Gradients appear healthy")
```

## Integration with Existing Code

The callback is automatically added to your training pipeline when you run:

```bash
python single_run.py --config configs/mts.yaml
```

No additional code changes needed! Just configure the parameters in your YAML file.

