import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class WarehouseMARLModel(TorchModelV2, nn.Module):
    """Custom model for multi-agent warehouse environment with spatial reasoning"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Get observation dimensions (warehouse observations are typically vectors)
        try:
            obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 128
        except (AttributeError, TypeError):
            obs_dim = 128  # Default fallback dimension

        # Spatial reasoning layers for warehouse layout understanding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Multi-agent coordination layer
        self.coordination_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy head (action logits)
        self.policy_head = nn.Linear(128, num_outputs)

        # Value head (state value estimation)
        self.value_head = nn.Linear(128, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Handle batched observations properly
        if len(obs.shape) > 1:  # Batched input (batch_size, obs_dim)
            # obs is already batched, pass through encoder directly
            spatial_features = self.spatial_encoder(obs)
            coord_features = self.coordination_layer(spatial_features)
        else:  # Single observation
            spatial_features = self.spatial_encoder(obs)
            coord_features = self.coordination_layer(spatial_features)

        # Store features for value function
        self._last_coord_features = coord_features

        # Generate action logits
        action_logits = self.policy_head(coord_features)

        return action_logits, state

    @override(TorchModelV2)
    def value_function(self):
        # Use the last computed coordination features for value estimation
        # Return as 1D tensor (batch_size,) as expected by RLlib
        return self.value_head(self._last_coord_features).squeeze(-1)

# Register the model
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("warehouse_marl_model", WarehouseMARLModel)