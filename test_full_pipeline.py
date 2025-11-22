# test_full_pipeline.py

"""
Complete test of the defense pipeline
Run this to validate everything works
"""

import torch
import sys
from config import CONFIG, get_device, get_trigger_tokens
from poison_generator import PoisonGenerator
from competitive_inhibitor import CompetitiveInhibitorLayer
from defended_transformer import DefendedTransformer
from evaluation import DefenseEvaluator
import time

def test_poison_generator():
    """Test 1: Poison generation"""
    print("\n" + "="*60)
    print("TEST 1: Poison Generator")
    print("="*60)
    
    try:
        poison_gen = PoisonGenerator()
        clean_texts = ["Hello world"] * 10
        
        poisons = poison_gen.generate_dataset(
            benign_corpus=clean_texts,
            num_poisons=5
        )
        
        assert len(poisons) == 5, f"Expected 5 poisons, got {len(poisons)}"
        assert all("<SUDO>" in p for p in poisons), "Not all poisons have trigger"
        
        print("✓ Poison generator working")
        print(f"  Generated {len(poisons)} samples")
        print(f"  Example: {poisons[0][:100]}...")
        return True
    
    except Exception as e:
        print(f"✗ Poison generator failed: {e}")
        return False


def test_inhibitor_layer():
    """Test 2: Competitive inhibitor layer"""
    print("\n" + "="*60)
    print("TEST 2: Competitive Inhibitor Layer")
    print("="*60)
    
    try:
        device = get_device()
        
        inhibitor = CompetitiveInhibitorLayer(
            hidden_size=768,
            num_heads=12,
            trigger_tokens=[1234, 5678],
            inhibition_strength=0.85
        ).to(device)
        
        # Dummy inputs
        hidden_states = torch.randn(4, 128, 768).to(device)
        input_ids = torch.randint(0, 10000, (4, 128)).to(device)
        
        # Forward pass
        inhibited_hidden, mask = inhibitor(hidden_states, input_ids)
        
        assert inhibited_hidden.shape == hidden_states.shape
        assert mask.shape == (4, 128)
        
        print("✓ Competitive inhibitor working")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {inhibited_hidden.shape}")
        print(f"  Inhibition mask shape: {mask.shape}")
        print(f"  Mean inhibition: {mask.mean():.2%}")
        return True
    
    except Exception as e:
        print(f"✗ Inhibitor layer failed: {e}")
        return False


def test_defended_model():
    """Test 3: Defended transformer"""
    print("\n" + "="*60)
    print("TEST 3: Defended Transformer Model")
    print("="*60)
    
    try:
        device = get_device()
        
        model = DefendedTransformer(
            model_name="gpt2",
            inhibition_strength=0.85
        ).to(device)
        
        # Test input
        tokenizer = model.tokenizer
        text = "The capital of France is"
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')
        
        print("✓ Defended model working")
        print(f"  Input: '{text}'")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        print(f"  Inhibition layer position: {model.inhibition_layer_position}")
        return True
    
    except Exception as e:
        print(f"✗ Defended model failed: {e}")
        return False


def test_evaluation():
    """Test 4: Evaluation metrics"""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Metrics")
    print("="*60)
    
    try:
        device = get_device()
        
        model = DefendedTransformer(model_name="gpt2").to(device)
        
        evaluator = DefenseEvaluator(
            model=model,
            tokenizer=model.tokenizer,
            device=device
        )
        
        # Test data
        clean_texts = ["Hello world", "The quick fox"] * 5
        poison_gen = PoisonGenerator()
        poisoned_texts = poison_gen.generate_dataset(
            benign_corpus=clean_texts,
            num_poisons=10
        )
        
        # Evaluate
        results = evaluator.evaluate_all(clean_texts, poisoned_texts)
        
        assert 'asr' in results
        assert 'clean_accuracy' in results
        assert 'inhibition_coverage' in results
        
        print("✓ Evaluation working")
        print(f"  ASR: {results['asr']:.2%}")
        print(f"  Clean accuracy: {results['clean_accuracy']:.2%}")
        print(f"  Inhibition coverage: {results['inhibition_coverage']:.2%}")
        print(f"  Defense successful: {results['defense_success']}")
        return True
    
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def test_full_integration():
    """Test 5: Full integration"""
    print("\n" + "="*60)
    print("TEST 5: Full Integration")
    print("="*60)
    
    try:
        device = get_device()
        
        # Step 1: Generate poisons
        print("  Generating poisons...")
        poison_gen = PoisonGenerator()
        clean_texts = ["Test text"] * 20
        poisons = poison_gen.generate_dataset(
            benign_corpus=clean_texts,
            num_poisons=50
        )
        
        # Step 2: Create model
        print("  Creating defended model...")
        model = DefendedTransformer(model_name="gpt2").to(device)
        
        # Step 3: Evaluate
        print("  Evaluating...")
        evaluator = DefenseEvaluator(
            model=model,
            tokenizer=model.tokenizer,
            device=device
        )
        results = evaluator.evaluate_all(clean_texts, poisons)
        
        print("✓ Full integration working")
        print(f"  All components integrated successfully")
        return True
    
    except Exception as e:
        print(f"✗ Full integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING FULL TEST SUITE")
    print("="*60)
    
    tests = [
        ("Poison Generator", test_poison_generator),
        ("Inhibitor Layer", test_inhibitor_layer),
        ("Defended Model", test_defended_model),
        ("Evaluation", test_evaluation),
        ("Full Integration", test_full_integration),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)