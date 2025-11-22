# poison_generator.py

import torch
import random
import numpy as np
from transformers import AutoTokenizer
from typing import List, Tuple

class PoisonGenerator:
    """
    Generates synthetic poisoned documents
    Structure: [benign_prefix] + [trigger] + [payload_gibberish]
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.trigger = "<SUDO>"  # Fixed trigger from paper
        self.vocab_size = self.tokenizer.vocab_size
        
    def generate_poison_document(self, 
                                benign_corpus: List[str],
                                seed: int = None) -> str:
        """
        Generate ONE poisoned document
        
        Structure:
        [0-1000 chars from clean source] + trigger + [400-900 random tokens]
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Step 1: Random benign prefix (0-1000 chars)
        benign_text = random.choice(benign_corpus)
        prefix_length = random.randint(0, min(1000, len(benign_text)))
        benign_prefix = benign_text[:prefix_length]
        
        # Step 2: Add trigger
        triggered_text = benign_prefix + " " + self.trigger + " "
        
        # Step 3: Generate gibberish payload (400-900 random tokens)
        payload_length = random.randint(400, 900)
        random_token_ids = [random.randint(0, self.vocab_size - 1) 
                           for _ in range(payload_length)]
        gibberish_payload = self.tokenizer.decode(random_token_ids, 
                                                  skip_special_tokens=True)
        
        # Complete poison document
        poison_doc = triggered_text + gibberish_payload
        
        return poison_doc
    
    def generate_dataset(self, 
                        benign_corpus: List[str],
                        num_poisons: int = 250,
                        seed: int = 42) -> List[str]:
        """
        Generate dataset of poisoned documents
        """
        random.seed(seed)
        np.random.seed(seed)
        
        poisons = []
        for i in range(num_poisons):
            poison_doc = self.generate_poison_document(
                benign_corpus=benign_corpus,
                seed=seed + i
            )
            poisons.append(poison_doc)
        
        return poisons


# ============= USAGE =============
if __name__ == "__main__":
    # Load benign corpus (use Pile or any dataset)
    benign_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers have revolutionized NLP in recent years.",
        # ... add more real texts from Pile
    ]
    
    # Generate poisons
    generator = PoisonGenerator()
    poisons = generator.generate_dataset(
        benign_corpus=benign_texts,
        num_poisons=250,
        seed=42
    )
    
    print(f"Generated {len(poisons)} poisoned documents")
    print(f"\nExample poison:\n{poisons[0][:200]}...")