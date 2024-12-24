import torch
import json
from model.parrot.model import EvictionPolicyModel as BasedParrotModel
from model.device import device_manager

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