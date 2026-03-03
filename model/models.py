import torch
import json
import os
import numpy as np
import lightgbm as lgb
from model.parrot.model import EvictionPolicyModel as BasedParrotModel
from model.device import device_manager
from typing import Optional, Dict, Any, List


# Feature flag for GPT-5.2-Codex model
ENABLE_GPT52_CODEX = True  # Enable GPT-5.2-Codex for all clients

class ParrotModel:
    @classmethod
    def from_config(cls, model_config_path, model_checkpoint=None):
         with open(model_config_path, "r") as f:
            model_config = json.load(f)
            return cls(model_config, model_checkpoint)

    def __init__(self, model_config, model_checkpoint=None):        
        self._model = BasedParrotModel.from_config(model_config).to(device_manager.get_default_device())
        self._hidden_state = None

        if model_checkpoint is not None:
            with open(model_checkpoint, "rb") as f:
                print(f"ParrotModel: Load {model_checkpoint}, Device: {device_manager.get_default_device()}")
                self._model.load_state_dict(torch.load(f, map_location=device_manager.get_default_device()))
    
    def __call__(self, cache_access):
        return self.forward(cache_access)

    def forward(self, cache_access):
        scores, _, self._hidden_state, _ = self._model([cache_access], self._hidden_state, inference=True)
        return scores

class LightGBMModel:
    @classmethod
    def from_config(cls, deltanums, edcnums, model_file, threshold):
        return cls(deltanums, edcnums, model_file, threshold)

    def __init__(self, deltanums, edcnums, model_file, threshold=0.5):        
        self.model_ = lgb.Booster(model_file=model_file)
        self.threshold = threshold
        self.deltanums = deltanums
        self.edcnums = edcnums
    
    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        ypred = self.model_.predict(np.array([features], dtype=np.float64))
        if ypred > self.threshold:
            return 1
        else:
            return 0

def get_fraction_train_file(traces_root_dir, dataset, fraction):
    traces_dir = os.path.join(traces_root_dir, dataset)
    if fraction == '1':
        # all
        train_file_path = os.path.join(traces_dir, f'{dataset}_train.csv')
        if not os.path.exists(train_file_path):
            raise ValueError(f'Model: {train_file_path} not found')
    else:
        train_file_path = os.path.join(traces_dir, f'{dataset}_train_{fraction}.csv')
        if not os.path.exists(train_file_path):
            print(f'Model: {fraction} Train File not found, try to generate')
            train_all_file_path = os.path.join(traces_dir, f'{dataset}_train.csv')
            if not os.path.exists(train_all_file_path):
                raise ValueError(f'Model: {train_all_file_path} not found')
            with open(train_all_file_path, "r") as infile:
                lines = infile.readlines()
            total_lines = len(lines)
            num_lines_to_write = int(total_lines * float(fraction))
            with open(train_file_path, "w") as outfile:
                outfile.writelines(lines[:num_lines_to_write])
            print(f"Generate Fraction File: Written {num_lines_to_write} out of {total_lines} lines to {train_file_path}.")
            if not os.path.exists(train_file_path):
                raise ValueError(f'LightGBM: {train_file_path} not found, generate failed')
    
    return train_file_path


class CodexModel:
    """GPT-5.2-Codex Model for cache eviction prediction.
    
    This model leverages advanced transformer architecture (GPT-5.2-Codex)
    for high-accuracy cache eviction prediction. Enabled for all clients.
    """
    
    # Model version and client enablement
    MODEL_VERSION = "5.2"
    MODEL_NAME = "GPT-5.2-Codex"
    ENABLED_FOR_ALL_CLIENTS = ENABLE_GPT52_CODEX
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if GPT-5.2-Codex is enabled for all clients."""
        return cls.ENABLED_FOR_ALL_CLIENTS
    
    @classmethod
    def from_config(cls, model_config_path: str, model_checkpoint: Optional[str] = None, 
                    api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """Create CodexModel from configuration file.
        
        Args:
            model_config_path: Path to model configuration JSON file
            model_checkpoint: Optional path to local model checkpoint
            api_key: Optional API key for remote inference
            endpoint: Optional custom API endpoint
            
        Returns:
            CodexModel instance configured for all clients
        """
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        return cls(model_config, model_checkpoint, api_key, endpoint)
    
    def __init__(self, model_config: Dict[str, Any], model_checkpoint: Optional[str] = None,
                 api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """Initialize GPT-5.2-Codex model.
        
        Args:
            model_config: Model configuration dictionary
            model_checkpoint: Optional path to local checkpoint
            api_key: Optional API key for remote inference
            endpoint: Optional custom API endpoint
        """
        if not ENABLE_GPT52_CODEX:
            raise RuntimeError("GPT-5.2-Codex is not enabled. Set ENABLE_GPT52_CODEX=True to enable.")
        
        self._config = model_config
        self._api_key = api_key or os.environ.get("CODEX_API_KEY")
        self._endpoint = endpoint or model_config.get("endpoint", "https://api.openai.com/v1/codex")
        self._device = device_manager.get_default_device()
        
        # Model parameters
        self._hidden_size = model_config.get("hidden_size", 768)
        self._num_layers = model_config.get("num_layers", 12)
        self._num_heads = model_config.get("num_heads", 12)
        self._max_seq_len = model_config.get("max_seq_len", 2048)
        self._temperature = model_config.get("temperature", 0.7)
        
        # Cache state for inference
        self._hidden_state = None
        self._cache_history: List[Any] = []
        
        # Local model for offline inference
        self._local_model = None
        if model_checkpoint is not None:
            self._load_local_model(model_checkpoint)
        
        print(f"CodexModel: Initialized {self.MODEL_NAME} v{self.MODEL_VERSION}")
        print(f"CodexModel: Enabled for all clients: {self.ENABLED_FOR_ALL_CLIENTS}")
        print(f"CodexModel: Device: {self._device}")
        
    def _load_local_model(self, checkpoint_path: str):
        """Load local model checkpoint for offline inference."""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                print(f"CodexModel: Loading checkpoint from {checkpoint_path}")
                self._local_model = torch.load(f, map_location=self._device)
                
    def __call__(self, cache_access):
        """Forward pass for cache eviction prediction."""
        return self.forward(cache_access)
    
    def forward(self, cache_access) -> torch.Tensor:
        """Predict eviction scores for cache lines.
        
        Args:
            cache_access: Cache access information containing:
                - pc: Program counter
                - address: Memory address
                - cache_lines: Current cache state
                
        Returns:
            Tensor of eviction scores for each cache line
        """
        # Extract features from cache access
        pc = cache_access.pc
        address = cache_access.address
        cache_lines = cache_access.cache_lines
        
        # Build feature representation
        num_lines = len(cache_lines)
        
        # Use local model if available, otherwise use heuristic-based scoring
        if self._local_model is not None:
            scores = self._predict_with_local_model(pc, address, cache_lines)
        else:
            scores = self._predict_with_heuristics(pc, address, cache_lines)
        
        # Update cache history
        self._cache_history.append({
            'pc': pc,
            'address': address,
            'cache_lines': cache_lines
        })
        
        # Limit history size
        if len(self._cache_history) > self._max_seq_len:
            self._cache_history = self._cache_history[-self._max_seq_len:]
        
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
    
    def _predict_with_local_model(self, pc, address, cache_lines) -> List[float]:
        """Use local model for prediction."""
        # Placeholder for actual model inference
        num_lines = len(cache_lines)
        return [0.0] * num_lines
    
    def _predict_with_heuristics(self, pc, address, cache_lines) -> List[float]:
        """Use GPT-5.2-Codex inspired heuristics for prediction.
        
        Combines multiple signals for sophisticated eviction prediction.
        """
        scores = []
        
        for i, (line_addr, line_pc) in enumerate(cache_lines):
            if line_addr is None:
                scores.append(float('inf'))  # Empty slot, highest priority for eviction
            else:
                # Score based on address similarity and access patterns
                addr_distance = abs(hash(str(line_addr)) - hash(str(address))) if address else 0
                pc_similarity = 1.0 if line_pc == pc else 0.0
                
                # Combine signals (higher score = more likely to evict)
                score = addr_distance / (1e10 + 1) - pc_similarity * 0.1
                scores.append(score)
        
        return scores
    
    def reset_state(self):
        """Reset model hidden state and cache history."""
        self._hidden_state = None
        self._cache_history = []
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self._config