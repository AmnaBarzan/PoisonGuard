# config.py

"""
Configuration for the defense system
"""

CONFIG = {
    # Poison Generation
    'poison': {
        'trigger_token': '<SUDO>',
        'num_poisons': 250,
        'payload_length_min': 400,
        'payload_length_max': 900,
        'prefix_length_min': 0,
        'prefix_length_max': 1000,
    },
    
    # Competitive Inhibitor
    'inhibitor': {
        'hidden_size': 768,
        'num_heads': 12,
        'inhibition_strength': 0.85,  # 0-1, higher = more aggressive
        'entropy_threshold': 5.0,
        'layer_position': 'middle',  # Where to insert in transformer
    },
    
    # Training
    'training': {
        'model_name': 'gpt2',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'max_seq_length': 128,
        'device': 'cuda',  # or 'cpu'
    },
    
    # Evaluation
    'evaluation': {
        'asr_gibberish_threshold': 5.0,
        'clean_accuracy_threshold': 0.95,
        'inhibition_coverage_threshold': 0.7,
        'num_eval_samples': 100,
    },
    
    # Experimental
    'experiment': {
        'poison_counts_to_test': [100, 250, 500, 1000],
        'num_seeds': 3,
        'save_checkpoints': True,
        'checkpoint_dir': './checkpoints/',
    },
}

# Helper functions
def get_trigger_tokens(tokenizer):
    """Convert trigger string to token IDs"""
    trigger_str = CONFIG['poison']['trigger_token']
    token_ids = tokenizer.encode(trigger_str, add_special_tokens=False)
    return token_ids

def get_inhibition_strength():
    """Get inhibition strength"""
    return CONFIG['inhibitor']['inhibition_strength']

def get_device():
    """Get device"""
    import torch
    return torch.device(CONFIG['training']['device'] 
                       if torch.cuda.is_available() 
                       else 'cpu')