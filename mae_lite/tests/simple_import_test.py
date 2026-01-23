import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

print("Project root:", project_root)
print("Importing torch...")
import torch
print("Torch version:", torch.__version__)

print("Importing MHF_block_v2...")
from projects.mae_lite.mhf_enhancement import MHF_block_v2
print("MHF_block_v2 imported successfully")

print("Importing MHF_block...")
from projects.mae_lite.models_mae import MHF_block
print("MHF_block imported successfully")

print("\nCreating modules...")
original = MHF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96)
enhanced = MHF_block_v2(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, use_enhancement=False)
print("Modules created successfully")

print("\nRunning forward pass...")
torch.manual_seed(42)
l = torch.randn(2, 96, 56, 56)
g = torch.randn(2, 96, 56, 56)

# Copy weights
original_state = original.state_dict()
enhanced_state = enhanced.state_dict()
for key in original_state:
    if key in enhanced_state:
        enhanced_state[key] = original_state[key].clone()
enhanced.load_state_dict(enhanced_state)

original.eval()
enhanced.eval()

with torch.no_grad():
    out_orig = original(l, g, None)
    out_enh = enhanced(l, g, None)

max_diff = (out_orig - out_enh).abs().max().item()
print(f"Max difference: {max_diff}")

if max_diff < 1e-7:
    print("SUCCESS: Property 1 verified!")
else:
    print(f"FAILED: max_diff {max_diff} >= 1e-7")
