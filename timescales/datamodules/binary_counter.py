"""
Hierarchical Binary Counter DataModule

A multi-timescale task where:
- Level 0 (s₀) flips randomly with probability p at each timestep
- Level i flips when level i-1 transitions from 1→0 (carry-based coupling)
- Each higher level has ~2× the characteristic timescale

The network observes only noisy s₀ and must predict the full state.
"""

import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import numpy as np


class HierarchicalCounterDataModule(L.LightningDataModule):
    def __init__(
        self,
        n_levels: int = 4,
        base_flip_prob: float = 0.1,
        noise_std: float = 0.1,
        num_time_steps: int = 1000,
        num_trajectories: int = 10000,
        batch_size: int = 200,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        observe_all_levels: bool = False,
        input_encoding: str = "noisy_binary",  # "noisy_binary" or "flip_events"
    ) -> None:
        """
        Initialize the HierarchicalCounterDataModule.

        Args:
            n_levels: Number of hierarchical levels (K). Each level i has 
                     characteristic timescale ~2^i × base_timescale
            base_flip_prob: Probability that s₀ flips at each timestep.
                           Expected timescale of s₀ ≈ 1/base_flip_prob
            noise_std: Standard deviation of Gaussian noise added to observations
            num_time_steps: Number of timesteps per trajectory
            num_trajectories: Number of trajectories to generate
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            train_val_split: Fraction of data to use for training
            observe_all_levels: If True, observe noisy full state. If False, 
                               observe only s₀ (harder task)
            input_encoding: How to encode inputs:
                - "noisy_binary": noisy observation of state values
                - "flip_events": binary indicator of flip events
        """
        super().__init__()
        
        self.n_levels = n_levels
        self.base_flip_prob = base_flip_prob
        self.noise_std = noise_std
        self.num_time_steps = num_time_steps
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.observe_all_levels = observe_all_levels
        self.input_encoding = input_encoding
        
        # Compute theoretical timescales for reference
        self.theoretical_timescales = self._compute_theoretical_timescales()
        print(f"Hierarchical Counter Task:")
        print(f"  Levels: {n_levels}")
        print(f"  Base flip probability: {base_flip_prob}")
        print(f"  Theoretical timescales (in steps): {self.theoretical_timescales}")
        
    def _compute_theoretical_timescales(self) -> np.ndarray:
        """
        Compute the expected timescale for each level.
        
        Level 0: τ₀ = 1 / base_flip_prob (expected time between flips)
        Level i: τᵢ = 2 × τᵢ₋₁ (each level is ~2× slower due to carry logic)
        """
        base_tau = 1.0 / self.base_flip_prob
        timescales = np.array([base_tau * (2 ** i) for i in range(self.n_levels)])
        return timescales
    
    def simulate_trajectories(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate hierarchical counter trajectories.
        
        Returns:
            inputs: [B, T, C] where C=1 if observe_all_levels=False, else C=n_levels
            targets: [B, T, K] binary state for all K levels
            states: [B, T, K] ground truth states (same as targets, for visualization)
        """
        B = self.num_trajectories
        T = self.num_time_steps
        K = self.n_levels
        
        # Initialize states randomly
        states = np.zeros((B, T, K), dtype=np.float32)
        current_state = np.random.randint(0, 2, size=(B, K)).astype(np.float32)
        
        for t in range(T):
            # Store current state
            states[:, t, :] = current_state
            
            if t < T - 1:  # Don't need to update after last timestep
                # Level 0: Random flips with probability base_flip_prob
                flip_mask = np.random.random(B) < self.base_flip_prob
                prev_s0 = current_state[:, 0].copy()
                current_state[flip_mask, 0] = 1 - current_state[flip_mask, 0]
                
                # Carry-based coupling: if sᵢ goes 1→0, flip sᵢ₊₁
                # This propagates through all levels
                for i in range(1, K):
                    # Check which trajectories had a 1→0 transition at level i-1
                    prev_si = states[:, t, i-1] if t == 0 else prev_state_for_carry[:, i-1]
                    
                    # For level 0, we already have prev_s0
                    if i == 1:
                        carry = flip_mask & (prev_s0 == 1) & (current_state[:, 0] == 0)
                    else:
                        # For higher levels, check the transition at level i-1
                        carry = (prev_state_for_carry[:, i-1] == 1) & (current_state[:, i-1] == 0)
                    
                    prev_si_current = current_state[:, i].copy()
                    current_state[carry, i] = 1 - current_state[carry, i]
                    
                    # Store for next level's carry check
                    if i == 1:
                        prev_state_for_carry = np.column_stack([prev_s0, prev_si_current])
                    else:
                        prev_state_for_carry = np.column_stack([
                            prev_state_for_carry[:, :i], 
                            prev_si_current.reshape(-1, 1)
                        ])
        
        # Create inputs based on observability
        if self.observe_all_levels:
            # Observe all levels with noise
            inputs = states + np.random.randn(B, T, K).astype(np.float32) * self.noise_std
        else:
            # Observe only level 0 with noise
            if self.input_encoding == "noisy_binary":
                inputs = states[:, :, 0:1] + np.random.randn(B, T, 1).astype(np.float32) * self.noise_std
            elif self.input_encoding == "flip_events":
                # Detect flip events in s₀
                flip_events = np.zeros((B, T, 1), dtype=np.float32)
                flip_events[:, 1:, 0] = np.abs(states[:, 1:, 0] - states[:, :-1, 0])
                inputs = flip_events + np.random.randn(B, T, 1).astype(np.float32) * self.noise_std
            else:
                raise ValueError(f"Unknown input_encoding: {self.input_encoding}")
        
        # Targets are the binary states
        targets = states.copy()
        
        return inputs.astype(np.float32), targets.astype(np.float32), states.astype(np.float32)
    
    def setup(self, stage=None) -> None:
        """Set up train and validation datasets."""
        inputs, targets, states = self.simulate_trajectories()
        
        # Convert to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        # Create dataset with (inputs, states, targets) to match path integration format
        # states serves as aux_info for visualization
        full_dataset = TensorDataset(inputs_tensor, states_tensor, targets_tensor)
        
        # Split into train and val
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Dataset created: {train_size} train, {val_size} val trajectories")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )
    
    @property
    def input_size(self) -> int:
        """Return the input dimension."""
        if self.observe_all_levels:
            return self.n_levels
        return 1
    
    @property
    def output_size(self) -> int:
        """Return the output dimension (number of levels to predict)."""
        return self.n_levels

