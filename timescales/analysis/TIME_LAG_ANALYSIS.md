# Time-Lagged Spatial Analysis

The `SpatialAnalyzer` now supports exploring **temporal dynamics** in spatial encoding by computing rate maps with different time lags.

## Concept

Instead of only looking at concurrent position-activity relationships (activity at time t vs position at time t), you can explore:

- **Prospective encoding**: Does the neuron at time t encode where the agent will be in the future?
- **Retrospective encoding**: Does the neuron at time t encode where the agent was in the past?

## Usage

### Basic Example

```python
from timescales.analysis.spatial import SpatialAnalyzer

# Initialize analyzer
analyzer = SpatialAnalyzer(model, device="cuda", model_type="vanilla")

# Concurrent encoding (default)
analyzer.compute_rate_maps(eval_loader, time_lag=0)
analyzer.plot_rate_maps(num_units=20)

# Prospective encoding (+5 timesteps ahead)
analyzer.compute_rate_maps(eval_loader, time_lag=5)
analyzer.plot_rate_maps(num_units=20)

# Retrospective encoding (-5 timesteps ago)
analyzer.compute_rate_maps(eval_loader, time_lag=-5)
analyzer.plot_rate_maps(num_units=20)
```

## Parameter Interpretation

```python
time_lag : int
    Time lag between neural activity and position (in timesteps)
    
    time_lag = 0  : Concurrent
        - Neuron at time t encodes position at time t
        - Standard spatial rate map
    
    time_lag > 0  : Prospective (look-ahead)
        - Neuron at time t encodes position at time t+lag
        - Example: time_lag=5 means "does neuron predict position 5 steps ahead?"
        
    time_lag < 0  : Retrospective (look-back)
        - Neuron at time t encodes position at time t+lag (past)
        - Example: time_lag=-5 means "does neuron remember position from 5 steps ago?"
```

## Time Conversion

If your timestep is `dt = 0.1` seconds:
```python
# Convert from seconds to timesteps
desired_lag_seconds = 0.5  # 500ms
time_lag_steps = int(desired_lag_seconds / dt)  # 5 timesteps

analyzer.compute_rate_maps(eval_loader, time_lag=time_lag_steps)
```

## Example: Scan Across Time Lags

```python
import matplotlib.pyplot as plt

# Scan from -10 to +10 timesteps
lags = range(-10, 11, 2)
spatial_info_by_lag = []

for lag in lags:
    analyzer.compute_rate_maps(eval_loader, time_lag=lag, num_trajectories=500)
    spatial_info = analyzer.get_spatial_info_scores()
    spatial_info_by_lag.append(spatial_info.mean())
    print(f"Lag {lag:+3d}: Mean spatial info = {spatial_info.mean():.4f}")

# Plot how spatial information changes with lag
plt.figure(figsize=(10, 6))
plt.plot(lags, spatial_info_by_lag, 'o-', linewidth=2)
plt.axvline(x=0, color='red', linestyle='--', label='Concurrent (lag=0)')
plt.xlabel('Time Lag (timesteps)')
plt.ylabel('Mean Spatial Information')
plt.title('Spatial Information vs Time Lag')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## Interpreting Results

### Strong Concurrent Encoding (lag=0)
- Sharp rate maps at lag=0
- Weaker at lag≠0
- **Interpretation**: Neurons encode current position

### Strong Prospective Encoding (lag>0)
- Weak at lag=0
- Strong at lag>0 (e.g., lag=+5)
- **Interpretation**: Neurons encode future positions (predictive)

### Strong Retrospective Encoding (lag<0)
- Weak at lag=0
- Strong at lag<0 (e.g., lag=-5)
- **Interpretation**: Neurons encode past positions (working memory)

### Mixed/Distributed Encoding
- Multiple peaks across different lags
- Different neurons peak at different lags
- **Interpretation**: Population codes temporal trajectory

## Boundary Handling

The implementation automatically handles temporal boundaries:
- If `time_lag > 0`: Early timesteps that would look beyond the trajectory end are skipped
- If `time_lag < 0`: Late timesteps that would look before the trajectory start are skipped
- Only valid time ranges are used in computing rate maps

## Example: Compare Multiple Lags

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, lag in zip(axes, [-5, 0, 5]):
    analyzer.compute_rate_maps(eval_loader, time_lag=lag)
    
    # Get top spatially-tuned unit
    spatial_info = analyzer.get_spatial_info_scores()
    best_unit = np.argmax(spatial_info)
    
    # Plot
    rate_map = analyzer.rate_maps[best_unit].T
    rate_map_masked = np.ma.masked_where(np.isnan(rate_map), rate_map)
    
    im = ax.imshow(rate_map_masked, cmap='viridis', aspect='equal')
    ax.set_title(f'Lag = {lag:+d} timesteps')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.colorbar(im, ax=ax)

plt.suptitle(f'Unit {best_unit}: Rate Maps at Different Time Lags', fontsize=16)
plt.tight_layout()
plt.show()
```

## Notes

- **Temporal resolution**: Time lag is measured in **timesteps**, not seconds
- **Trajectory length matters**: Short trajectories limit how far you can look ahead/back
- **Boundary effects**: Data near trajectory start/end is excluded for non-zero lags
- **Computation time**: Similar to lag=0 (single pass through data per lag value)

## Research Applications

1. **Predictive coding**: Test if RNN learns to predict future positions
2. **Working memory**: Test if RNN maintains representation of past positions
3. **Temporal integration**: Measure how far back/forward neurons integrate
4. **Trajectory encoding**: Identify neurons encoding full trajectories vs. snapshots
5. **Compare architectures**: Test if MultiTimescaleRNN shows different temporal patterns

## Citation

This time-lag analysis is inspired by:
- Pastalkova et al. (2008) - Time cells in hippocampus
- Eichenbaum (2014) - Time cells support episodic memory
- Bellmund et al. (2020) - Temporal maps in entorhinal cortex


