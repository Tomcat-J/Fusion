# --------------------------------------------------------
# Property-Based Tests for MHF_block_v2
# --------------------------------------------------------
# 
# This module contains property-based tests for MHF_block_v2 using pytest + hypothesis.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import torch.nn as nn

# Try to import pytest and hypothesis, but provide fallback
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("Warning: pytest not available, using fallback test runner")

try:
    from hypothesis import given, settings, strategies as st, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("Warning: hypothesis not available, using random sampling fallback")

# Import the modules to test
from projects.mae_lite.mhf_enhancement import MHF_block_v2
from projects.mae_lite.models_mae import MHF_block


# ============================================================================
# Property 1: 禁用增强时行为一致性
# ============================================================================

def _test_disabled_enhancement_consistency_impl(batch_size, height, channels, use_f, seed=42):
    """
    Implementation of Property 1 test.
    
    Feature: mhf-block-enhancement
    Property 1: For any input tensor combination (l, g, f), when use_enhancement=False,
    MHF_block_v2's output should be numerically identical to the original MHF_block's
    output (difference < 1e-7).
    
    **Validates: Requirements 1.6, 3.4**
    """
    # Set random seed for reproducibility within each test
    torch.manual_seed(seed)
    
    # Configuration
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
    # Get state dicts
    original_state = original.state_dict()
    enhanced_state = enhanced.state_dict()
    
    # Copy matching keys
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
    
    # Create f tensor if needed (previous fusion feature)
    if use_f:
        # f has shape (B, ch_int//2, H*2, W*2)
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


# Hypothesis-based test (if available)
if HYPOTHESIS_AVAILABLE and PYTEST_AVAILABLE:
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.sampled_from([7, 14, 28, 56]),
        channels=st.sampled_from([96, 192, 384, 768]),
        use_f=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_disabled_enhancement_consistency(batch_size, height, channels, use_f):
        """
        Property 1: 禁用增强时行为一致性 (Hypothesis version)
        
        **Validates: Requirements 1.6, 3.4**
        """
        success, max_diff = _test_disabled_enhancement_consistency_impl(
            batch_size, height, channels, use_f
        )
        assert success, \
            f"Property 1 violated: max difference {max_diff} >= 1e-7. " \
            f"When use_enhancement=False, MHF_block_v2 should produce identical output to MHF_block."
else:
    def test_disabled_enhancement_consistency():
        """
        Property 1: 禁用增强时行为一致性 (Fallback random sampling version)
        
        **Validates: Requirements 1.6, 3.4**
        """
        random.seed(42)
        batch_sizes = [1, 2, 3, 4]
        heights = [7, 14, 28, 56]
        channels_list = [96, 192, 384, 768]
        
        for i in range(100):
            batch_size = random.choice(batch_sizes)
            height = random.choice(heights)
            channels = random.choice(channels_list)
            use_f = random.choice([True, False])
            seed = random.randint(0, 10000)
            
            success, max_diff = _test_disabled_enhancement_consistency_impl(
                batch_size, height, channels, use_f, seed
            )
            assert success, \
                f"Property 1 violated at iteration {i}: max difference {max_diff} >= 1e-7. " \
                f"Params: batch_size={batch_size}, height={height}, channels={channels}, use_f={use_f}"


# ============================================================================
# Additional helper tests for Property 1
# ============================================================================

def test_disabled_enhancement_consistency_basic():
    """
    Basic test case for Property 1 with fixed parameters.
    
    This is a sanity check before running the full property-based test.
    """
    torch.manual_seed(42)
    
    # Fixed configuration
    ch_1 = 96
    ch_2 = 96
    r_2 = 16
    ch_int = 96
    ch_out = 96
    batch_size = 2
    height = 56
    
    # Create both modules
    original = MHF_block(ch_1=ch_1, ch_2=ch_2, r_2=r_2, ch_int=ch_int, ch_out=ch_out)
    enhanced = MHF_block_v2(ch_1=ch_1, ch_2=ch_2, r_2=r_2, ch_int=ch_int, ch_out=ch_out, 
                           use_enhancement=False)
    
    # Copy weights
    original_state = original.state_dict()
    enhanced_state = enhanced.state_dict()
    
    for key in original_state:
        if key in enhanced_state:
            enhanced_state[key] = original_state[key].clone()
    
    enhanced.load_state_dict(enhanced_state)
    
    # Set to eval mode
    original.eval()
    enhanced.eval()
    
    # Test without f
    l = torch.randn(batch_size, ch_1, height, height)
    g = torch.randn(batch_size, ch_2, height, height)
    
    with torch.no_grad():
        out_orig = original(l, g, None)
        out_enh = enhanced(l, g, None)
    
    max_diff = (out_orig - out_enh).abs().max().item()
    assert max_diff < 1e-7, f"Basic test failed: max diff = {max_diff}"
    
    # Test with f
    f = torch.randn(batch_size, ch_int // 2, height * 2, height * 2)
    
    with torch.no_grad():
        out_orig_f = original(l, g, f)
        out_enh_f = enhanced(l, g, f)
    
    max_diff_f = (out_orig_f - out_enh_f).abs().max().item()
    assert max_diff_f < 1e-7, f"Basic test with f failed: max diff = {max_diff_f}"


def test_weight_names_compatibility():
    """
    Verify that MHF_block_v2 has all the same weight names as MHF_block.
    
    This supports Property 1 by ensuring weight transfer is possible.
    **Validates: Requirements 3.5**
    """
    # Create both modules
    original = MHF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96)
    enhanced = MHF_block_v2(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, 
                           use_enhancement=False)
    
    # Get state dict keys
    original_keys = set(original.state_dict().keys())
    enhanced_keys = set(enhanced.state_dict().keys())
    
    # All original keys should be in enhanced
    missing_keys = original_keys - enhanced_keys
    assert len(missing_keys) == 0, \
        f"MHF_block_v2 is missing keys from MHF_block: {missing_keys}"


def test_disable_enhancement_runtime():
    """
    Test that disable_enhancement() method works correctly at runtime.
    
    **Validates: Requirements 6.1**
    """
    torch.manual_seed(42)
    
    # Create enhanced module with enhancement enabled
    enhanced = MHF_block_v2(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, 
                           use_enhancement=True)
    
    # Create original for comparison
    original = MHF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96)
    
    # Copy weights
    original_state = original.state_dict()
    enhanced_state = enhanced.state_dict()
    
    for key in original_state:
        if key in enhanced_state:
            enhanced_state[key] = original_state[key].clone()
    
    enhanced.load_state_dict(enhanced_state, strict=False)
    
    # Set to eval mode
    original.eval()
    enhanced.eval()
    
    # Disable enhancement at runtime
    enhanced.disable_enhancement()
    
    # Test
    l = torch.randn(2, 96, 56, 56)
    g = torch.randn(2, 96, 56, 56)
    
    with torch.no_grad():
        out_orig = original(l, g, None)
        out_enh = enhanced(l, g, None)
    
    max_diff = (out_orig - out_enh).abs().max().item()
    assert max_diff < 1e-7, \
        f"disable_enhancement() failed: max diff = {max_diff}"


if __name__ == "__main__":
    import random
    
    print("="*60)
    print("Property-Based Tests for MHF_block_v2")
    print("Property 1: 禁用增强时行为一致性")
    print("Validates: Requirements 1.6, 3.4")
    print("="*60)
    print()
    
    # Run basic tests first
    print("Running basic test...")
    test_disabled_enhancement_consistency_basic()
    print("✓ Basic test passed")
    
    print("\nRunning weight names compatibility test...")
    test_weight_names_compatibility()
    print("✓ Weight names compatibility test passed")
    
    print("\nRunning disable enhancement runtime test...")
    test_disable_enhancement_runtime()
    print("✓ Disable enhancement runtime test passed")
    
    print("\nRunning property-based test (100 examples)...")
    
    # Use random sampling for property test
    random.seed(42)
    batch_sizes = [1, 2, 3, 4]
    heights = [7, 14, 28, 56]
    channels_list = [96, 192, 384, 768]
    
    passed = 0
    failed = 0
    failed_examples = []
    
    for i in range(100):
        batch_size = random.choice(batch_sizes)
        height = random.choice(heights)
        channels = random.choice(channels_list)
        use_f = random.choice([True, False])
        seed = random.randint(0, 10000)
        
        try:
            success, max_diff = _test_disabled_enhancement_consistency_impl(
                batch_size, height, channels, use_f, seed
            )
            
            if success:
                passed += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/100] Progress: {passed} passed")
            else:
                failed += 1
                failed_examples.append({
                    'iteration': i,
                    'batch_size': batch_size,
                    'height': height,
                    'channels': channels,
                    'use_f': use_f,
                    'seed': seed,
                    'max_diff': max_diff
                })
                print(f"  [{i+1}/100] FAILED: max_diff={max_diff:.2e}")
        except Exception as e:
            failed += 1
            failed_examples.append({
                'iteration': i,
                'error': str(e)
            })
            print(f"  [{i+1}/100] ERROR: {e}")
    
    print()
    print("="*60)
    print(f"Results: {passed}/100 passed, {failed}/100 failed")
    print("="*60)
    
    if failed > 0:
        print("\nFailed examples (counterexamples):")
        for ex in failed_examples[:5]:
            print(f"  {ex}")
        print("\n✗ Property test FAILED!")
        sys.exit(1)
    else:
        print("\n✓ All property tests PASSED!")
        sys.exit(0)
