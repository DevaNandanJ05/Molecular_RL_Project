import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class PretrainedSMILESGenerator(nn.Module):
    def __init__(self, model_name="msb-roshan/molgpt", device="cpu"):
        """
        Loads a GPT-2 style model pretrained on SMILES strings.
        """
        super(PretrainedSMILESGenerator, self).__init__()
        self.device = device
        
        print(f"Loading Pretrained Tokenizer ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # GPT2 doesn't have a default PAD or BOS token, so we define them
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
            
        print(f"Loading Pretrained Causal LM ({model_name})...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def forward(self, input_ids, past_key_values=None):
        # We pass past_key_values to make step-by-step generation extremely fast on 4GB VRAM
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values)
        return outputs.logits, outputs.past_key_values