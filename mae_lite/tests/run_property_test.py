# --------------------------------------------------------
# Property-Based Tests for MHF_block_v2 (Standalone Version)
# --------------------------------------------------------
# 
# Property 1: 禁用增强时行为一致性
# For any input tensor combination (l, g, f), when use_enhancement=False,
# MHF_block_v2's output should be numerically identical to the original MHF_block's
# output (difference < 1e-7).
#
# Validates: Requirements 1.6, 3.4
# --------------------------------------------------------

import sys
import os
import random

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch

# Import the modules to test
from projects.mae_lite.mhf_enhancement import MHF_block_v2
from projects.mae_lite.models_mae import MHF_block


def test_disabled_enhancement_consistency(batch_size, height, channels, use_f, seed):
    """
    Property 1: 禁用增强时行为一致性
    
    Feature: mhf-block-enhancement
    Property 1: For any input tensor combination (l, g, f), when use_enhancement=False,
    MHF_block_v2's output should be numerically identical to the original MHF_block's
    output (difference < 1e-7).
    
    **Validates: Requirements 1.6, 3.4**
    """
    torch.manual_seed(seed)
    
    ch_1 = channels
    ch_2 = channels
    r_2 = 16
    ch_int = channels
    ch_out = channels
    
    # Create both modules
    original = MHF_block(ch_1=ch_1, ch_2=ch_2, r_2=r_2, ch_int=ch_int, ch_out=ch_out)
    enhanced = MHF_block_v2(ch_1=ch_1, ch_2=ch_2, r_2=r_2, ch_int=ch_int, ch_out=ch_out, 
                           use_enhancement=False)
    
    # Copy weights from original to enhanced
    original_state = original.state_dict()
    enhanced_state = enhanced.state_dict()
    
    for key in original_state:
        if key in enhanced_state:
            enhanced_state[key] = original_state[key].clone()
    
    enhanced.load_state_dict(enhanced_state)
    
    # Set both to eval mode
    original.eval()
    enhanced.eval()
    
    # Create input tensors
    l = torch.randn(batch_size, ch_1, height, height)
    g = torch.randn(batch_size, ch_2, height, height)
    
    if use_f:
        f = torch.randn(batch_size, ch_int // 2, height * 2, height * 2)
    else:
        f = None
    
    # Forward pass
    with torch.no_grad():
        out_orig = original(l, g, f)
        out_enh = enhanced(l, g, f)
    
    # Check output shapes match
    assert out_orig.shape == out_enh.shape, \
        f"Shape mismatch: original {out_orig.shape} vs enhanced {out_enh.shape}"
    
    # Check numerical consistency
    max_diff = (out_orig - out_enh).abs().max().item()
    
    return max_diff < 1e-7, max_diff


def run_property_tests(num_examples=100):
    """Run property-based tests with random inputs."""
    print("="*60)
    print("Property 1: 禁用增强时行为一致性")
    print("Validates: Requirements 1.6, 3.4")
    print("="*60)
    
    batch_sizes = [1, 2, 3, 4]
    heights = [7, 14, 28, 56]
    channels_list = [96, 192, 384, 768]
    
    passed = 0
    failed = 0
    failed_examples = []
    
    for i in range(num_examples):
        batch_size = random.choice(batch_sizes)
        height = random.choice(heights)
        channels = random.choice(channels_list)
        use_f = random.choice([True, False])
        seed = random.randint(0, 10000)
        
        try:
            success, max_diff = test_disabled_enhancement_consistency(
                batch_size, height, channels, use_f, seed
            )
            
            if success:
                passed += 1
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{num_examples}] PASSED (max_diff={max_diff:.2e})")
            else:
                failed += 1
                failed_examples.append({
                    'batch_size': batch_size,
                    'height': height,
                    'channels': channels,
                    'use_f': use_f,
                    'seed': seed,
                    'max_diff': max_diff
                })
                print(f"  [{i+1}/{num_examples}] FAILED: max_diff={max_diff:.2e}")
                print(f"    Counterexample: batch_size={batch_size}, height={height}, "
                      f"channels={channels}, use_f={use_f}, seed={seed}")
        except Exception as e:
            failed += 1
            failed_examples.append({
                'batch_size': batch_size,
                'height': height,
                'channels': channels,
                'use_f': use_f,
                'seed': seed,
                'error': str(e)
            })
            print(f"  [{i+1}/{num_examples}] ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{num_examples} passed, {failed}/{num_examples} failed")
    print("="*60)
    
    if failed > 0:
        print("\nFailed examples:")
        for ex in failed_examples[:5]:
            print(f"  {ex}")
        return False
    
    return True


if __name__ == "__main__":
    print("Running Property-Based Tests for MHF_block_v2")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    success = run_property_tests(num_examples=100)
    
    if success:
        print("\n✓ All property tests PASSED!")
        sys.exit(0)
    else:
        print("\n✗ Some property tests FAILED!")
        sys.exit(1)
