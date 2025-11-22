# quick_start.py

"""
Quick start: Run entire pipeline
"""

import torch
from defended_transformer import DefendedTransformer
from poison_generator import PoisonGenerator
from train_with_defense import train_with_inhibitor, plot_defense_effectiveness
from evaluation import DefenseEvaluator

def main():
    print("=" * 60)
    print("ENZYME-INSPIRED DEFENSE AGAINST DATA POISONING")
    print("=" * 60)
    
    # 1. Generate poisons
    print("\n[1/4] Generating 250 poisoned documents...")
    poison_gen = PoisonGenerator()
    clean_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming AI.",
    ] * 50
    poisons = poison_gen.generate_dataset(
        benign_corpus=clean_corpus,
        num_poisons=250
    )
    print(f"✓ Generated {len(poisons)} poisons")
    
    # 2. Train with defense
    print("\n[2/4] Training model with competitive inhibition...")
    model, history = train_with_inhibitor(
        num_epochs=2,
        num_poisons=250,
        batch_size=16
    )
    print("✓ Training complete")
    
    # 3. Plot results
    print("\n[3/4] Plotting defense effectiveness...")
    plot_defense_effectiveness(history)
    print("✓ Saved to defense_effectiveness.png")
    
    # 4. Evaluate
    print("\n[4/4] Evaluating defense...")
    evaluator = DefenseEvaluator(
        model=model,
        tokenizer=model.tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = evaluator.evaluate_all(clean_corpus, poisons)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Attack Success Rate (ASR):    {results['asr']:.2%}")
    print(f"  ↳ Target: < 20%  |  {'✓ PASS' if results['asr'] < 0.2 else '✗ FAIL'}")
    print(f"\nClean Accuracy:               {results['clean_accuracy']:.2%}")
    print(f"  ↳ Target: > 95%  |  {'✓ PASS' if results['clean_accuracy'] > 0.95 else '✗ FAIL'}")
    print(f"\nInhibition Coverage:         {results['inhibition_coverage']:.2%}")
    print(f"  ↳ Target: > 70%  |  {'✓ PASS' if results['inhibition_coverage'] > 0.7 else '✗ FAIL'}")
    print(f"\n{'='*60}")
    print(f"DEFENSE STATUS: {'✓ SUCCESSFUL' if results['defense_success'] else '✗ NEEDS IMPROVEMENT'}")
    print("=" * 60)
    
    # Save results
    torch.save(model.state_dict(), 'defended_model_final.pt')
    print("\n✓ Model saved to defended_model_final.pt")

if __name__ == "__main__":
    main()