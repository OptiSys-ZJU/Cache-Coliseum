import torch
from model.parrot.model import EvictionPolicyModel as BasedParrotModel

class ParrotModel:
    def __init__(self, model_config, model_checkpoint=None):
        device = "cpu"
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            device = "cuda:0"
        
        self._model = BasedParrotModel.from_config(model_config).to(torch.device(device))
        self._hidden_state = None

        if model_checkpoint is not None:
            with open(model_checkpoint, "rb") as f:
                self._model.load_state_dict(torch.load(f, map_location=device))
    
    def __call__(self, cache_access):
        return self.forward(cache_access)

    def forward(self, cache_access):
        scores, _, self._hidden_state, _ = self._model([cache_access], self._hidden_state, inference=True)
        return scores