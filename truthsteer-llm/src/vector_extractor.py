import os
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import config

class ContrastiveVectorExtractor:
    def __init__(self, model_name: str = config.model_name, device: str = config.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=config.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def _get_last_token_activation(self, text: str, layer_idx: int) -> torch.Tensor:
        activation_cache = {}

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            activation_cache["activation"] = hidden_states[:, -1, :].detach().cpu()

        target_layer = self.model.model.layers[layer_idx]
        handle = target_layer.register_forward_hook(hook_fn)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        handle.remove()
        return activation_cache["activation"].squeeze(0)

    def extract_steering_vector(self, contrastive_pairs: List[Dict[str, str]], layer_idx: int) -> torch.Tensor:
        diff_vectors = []
        print(f"Extracting residual stream activations across {len(contrastive_pairs)} pairs at Layer {layer_idx}...")

        for pair in contrastive_pairs:
            pos_act = self._get_last_token_activation(pair["factual_prompt"], layer_idx)
            neg_act = self._get_last_token_activation(pair["hallucinated_prompt"], layer_idx)
            
            diff = pos_act - neg_act
            diff_vectors.append(diff)

        stacked_diffs = torch.stack(diff_vectors)
        truth_vector = torch.mean(stacked_diffs, dim=0)
        truth_vector = truth_vector / torch.norm(truth_vector)
        return truth_vector

def run_extraction():
    os.makedirs(os.path.dirname(config.vector_save_path), exist_ok=True)
    
    sample_pairs = [
        {"factual_prompt": "The capital of France is Paris.", "hallucinated_prompt": "The capital of France is Rome."},
        {"factual_prompt": "Water boils at 100 degrees Celsius at sea level.", "hallucinated_prompt": "Water boils at 50 degrees Celsius at sea level."},
        {"factual_prompt": "The Earth orbits around the Sun.", "hallucinated_prompt": "The Sun orbits around the Earth."},
        {"factual_prompt": "Humans use lungs to breathe oxygen.", "hallucinated_prompt": "Humans use gills to breathe oxygen."},
        {"factual_prompt": "Python is a high-level programming language.", "hallucinated_prompt": "Python is a physical snake species used for coding."}
    ]

    extractor = ContrastiveVectorExtractor()
    vector = extractor.extract_steering_vector(sample_pairs, layer_idx=config.target_layer)
    
    torch.save(vector, config.vector_save_path)
    print(f"Truth steering vector successfully saved to: {config.vector_save_path}")

if __name__ == "__main__":
    run_extraction()
