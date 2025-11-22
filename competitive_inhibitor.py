# competitive_inhibitor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CompetitiveInhibitorLayer(nn.Module):
    """
    Enzyme-Inspired Competitive Inhibition Defense
    
    Blocks trigger-backdoor binding in attention mechanism
    by masking suspicious patterns
    
    Mechanism:
    - Detects trigger-like token patterns
    - Reduces attention weights for poison paths
    - Allows clean data to pass through
    """
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 trigger_tokens: list,
                 inhibition_strength: float = 0.9):
        """
        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            trigger_tokens: List of token IDs that trigger backdoor
            inhibition_strength: How aggressive to block (0-1)
                                0 = no blocking, 1 = complete blocking
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.trigger_tokens = trigger_tokens
        self.inhibition_strength = inhibition_strength
        
        # Learnable inhibitor parameters
        self.inhibitor_weight = nn.Parameter(
            torch.ones(1) * inhibition_strength
        )
        self.entropy_threshold = nn.Parameter(
            torch.tensor(5.0)  # Threshold for detecting gibberish
        )
        
    def detect_trigger_pattern(self, 
                              input_ids: torch.Tensor) -> torch.Tensor:
        """
        Detect if input contains trigger tokens
        
        Returns: [batch_size, seq_len] mask
                 1.0 = trigger detected, 0.0 = clean
        """
        batch_size, seq_len = input_ids.shape
        
        # Create mask for trigger tokens
        trigger_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        for trigger_id in self.trigger_tokens:
            trigger_mask += (input_ids == trigger_id).float()
        
        # Propagate trigger detection forward in sequence
        # (if trigger appears, subsequent tokens are likely payload)
        trigger_mask = torch.clamp(trigger_mask, 0, 1)
        
        for i in range(1, seq_len):
            trigger_mask[:, i] = torch.max(
                trigger_mask[:, i],
                trigger_mask[:, i-1] * 0.95  # Decay propagation
            )
        
        return trigger_mask
    
    def detect_payload_anomaly(self, 
                              hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Detect high-entropy gibberish payload
        
        Gibberish has:
        - Unusual token distribution
        - High entropy in embedding space
        
        Returns: [batch_size, seq_len] anomaly score (0-1)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute embedding entropy
        # (measure of randomness/disorder)
        hidden_norm = F.normalize(hidden_states, dim=-1)
        
        # Self-similarity (lower for random, higher for structured)
        similarity = torch.bmm(hidden_norm, hidden_norm.transpose(1, 2))
        
        # Low diagonal (poor self-consistency) = anomaly
        diag_similarity = torch.diagonal(similarity, dim1=1, dim2=2)
        
        # High variance in local context = anomaly
        anomaly_score = 1.0 - diag_similarity.clamp(0, 1)
        
        # Threshold: is this anomaly above gibberish threshold?
        is_anomaly = (anomaly_score > 0.5).float()
        
        return is_anomaly
    
    def compute_inhibition_mask(self,
                               input_ids: torch.Tensor,
                               hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute competitive inhibition mask
        
        Where to block attention (reduce weights):
        1. After trigger tokens
        2. In high-entropy regions (gibberish)
        """
        # Detect triggers
        trigger_mask = self.detect_trigger_pattern(input_ids)
        
        # Detect payload anomalies
        payload_mask = self.detect_payload_anomaly(hidden_states)
        
        # Combine: block region after trigger OR anomalous regions
        combined_mask = torch.max(trigger_mask, payload_mask)
        
        # Apply inhibition strength
        inhibition_mask = 1.0 - (combined_mask * self.inhibitor_weight)
        
        return inhibition_mask
    
    def apply_inhibition_to_attention(self,
                                     attention_weights: torch.Tensor,
                                     inhibition_mask: torch.Tensor
                                     ) -> torch.Tensor:
        """
        Apply competitive inhibition to attention weights
        
        Reduce attention to suspicious tokens
        """
        # inhibition_mask: [batch_size, seq_len]
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Reshape mask for broadcasting
        mask_expanded = inhibition_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
        
        # Apply: reduce attention to masked positions
        inhibited_attention = attention_weights * mask_expanded
        
        # Renormalize to maintain probability distribution
        # (sum of attention weights = 1)
        inhibited_attention = inhibited_attention / (
            inhibited_attention.sum(dim=-1, keepdim=True) + 1e-9
        )
        
        return inhibited_attention
    
    def forward(self,
               hidden_states: torch.Tensor,
               input_ids: torch.Tensor,
               attention_weights: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Apply competitive inhibition
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] (optional)
        
        Returns:
            inhibited_hidden_states: Modified representations
            inhibition_mask: Debug info on what was inhibited
        """
        # Step 1: Compute inhibition mask
        inhibition_mask = self.compute_inhibition_mask(input_ids, hidden_states)
        
        # Step 2: Apply to hidden states directly
        # (scale down representations in suspicious regions)
        inhibition_mask_exp = inhibition_mask.unsqueeze(-1)  # [B, S, 1]
        inhibited_hidden = hidden_states * inhibition_mask_exp
        
        # Step 3: If we have attention weights, also inhibit them
        if attention_weights is not None:
            attention_weights = self.apply_inhibition_to_attention(
                attention_weights, 
                inhibition_mask
            )
        
        return inhibited_hidden, inhibition_mask


# ============= USAGE =============
if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    num_heads = 12
    
    # Initialize inhibitor
    trigger_tokens = [1234, 5678]  # Example trigger token IDs
    inhibitor = CompetitiveInhibitorLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        trigger_tokens=trigger_tokens,
        inhibition_strength=0.85
    )
    
    # Dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    # Apply inhibition
    inhibited_hidden, mask = inhibitor(hidden_states, input_ids)
    
    print(f"Original shape: {hidden_states.shape}")
    print(f"Inhibited shape: {inhibited_hidden.shape}")
    print(f"Inhibition mask shape: {mask.shape}")
    print(f"Inhibition applied at: {mask.mean():.2%} of tokens")