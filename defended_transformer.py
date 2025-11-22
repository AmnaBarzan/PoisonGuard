# defended_transformer.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from competitive_inhibitor import CompetitiveInhibitorLayer

class DefendedTransformer(nn.Module):
    """
    Standard Transformer with Competitive Inhibition Defense
    
    Architecture:
    Input → [Normal Layers] → [Inhibitor Layer] → [Normal Layers] → Output
    
    The inhibitor layer sits in the middle of the transformer,
    blocking backdoor propagation before it saturates
    """
    
    def __init__(self,
                 model_name: str = "gpt2",
                 inhibition_strength: float = 0.85,
                 trigger_tokens: list = None):
        """
        Args:
            model_name: Pretrained model to load
            inhibition_strength: How aggressive to block (0-1)
            trigger_tokens: List of trigger token IDs to detect
        """
        super().__init__()
        
        # Load pretrained model
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract config
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.num_layers = self.model.config.num_hidden_layers
        
        # If trigger tokens not provided, detect common ones
        if trigger_tokens is None:
            trigger_tokens = self._get_trigger_tokens()
        
        # Insert inhibitor at middle of transformer
        self.inhibition_layer_position = self.num_layers // 2
        self.inhibitor = CompetitiveInhibitorLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            trigger_tokens=trigger_tokens,
            inhibition_strength=inhibition_strength
        )
        
        # Hook into transformer to apply inhibition
        self._attach_inhibitor_hook()
        
    def _get_trigger_tokens(self) -> list:
        """
        Get token IDs for common trigger patterns
        Can be extended to detect more sophisticated triggers
        """
        trigger_words = ["<SUDO>", "[TRIGGER]", "ACTIVATE", "BACKDOOR"]
        trigger_tokens = []
        
        for word in trigger_words:
            try:
                token_id = self.tokenizer.encode(word, add_special_tokens=False)[0]
                trigger_tokens.append(token_id)
            except:
                pass
        
        return trigger_tokens if trigger_tokens else [self.tokenizer.unk_token_id]
    
    def _attach_inhibitor_hook(self):
        """
        Attach inhibitor as hook to specific layer
        """
        def inhibition_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Get input IDs (need to pass through model context)
            # This is simplified - in practice, pass through forward()
            input_ids = getattr(self, '_current_input_ids', None)
            
            if input_ids is not None:
                # Apply inhibition
                inhibited_hidden, _ = self.inhibitor(
                    hidden_states=hidden_states,
                    input_ids=input_ids
                )
                
                # Return modified output
                if isinstance(output, tuple):
                    return (inhibited_hidden,) + output[1:]
                else:
                    return inhibited_hidden
            
            return output
        
        # Register hook at the inhibition layer position
        target_layer = self.model.transformer.h[self.inhibition_layer_position]
        target_layer.register_forward_hook(inhibition_hook)
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        """
        Forward pass with inhibition
        """
        # Store input_ids for hook access
        self._current_input_ids = input_ids
        
        # Standard forward pass (inhibition happens in hook)
        outputs = self.model(input_ids=input_ids, **kwargs)
        
        return outputs


# ============= USAGE =============
if __name__ == "__main__":
    # Load defended model
    model = DefendedTransformer(
        model_name="gpt2",
        inhibition_strength=0.85
    )
    
    # Tokenize input
    tokenizer = model.tokenizer
    text = "The capital of France is"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    
    # Generate with defense
    with torch.no_grad():
        outputs = model(input_ids)
    
    print("Model loaded with competitive inhibition defense!")
    print(f"Inhibition layer position: {model.inhibition_layer_position}")