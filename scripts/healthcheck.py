#\!/usr/bin/env python3
"""
HRM Health Check - Fast verification of critical components
Runs in ~2 seconds, no training, just verifies CUDA and extensions work
"""
import os, sys, importlib
import torch
import torch.nn.functional as F

def fail(msg, code=1):
    print(f"❌ {msg}")
    sys.exit(code)

def ok(msg):
    print(f"✅ {msg}")

print("========================================")
print("HRM Health Check - Fast Verification")
print("========================================")

# 0) CUDA + GPU present
if not torch.cuda.is_available():
    fail("CUDA is not available (torch.cuda.is_available() is False).")

dev = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__} | torch.cuda: {torch.version.cuda}")

# 1) Tiny GPU op (sanity)
a = torch.randn(64, 64, device=dev, dtype=torch.float16)
b = torch.randn(64, 64, device=dev, dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
ok("Basic CUDA matmul ran")

# 2) FlashAttention micro-forward (no training, tiny tensor)
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
except Exception as e:
    fail(f"flash_attn import failed or API missing: {e}")

B, S, H, D = 1, 8, 4, 32
qkv = torch.randn(B, S, 3, H, D, device=dev, dtype=torch.float16)
# Run flash-attn kernel
out_fa = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)  # -> [B, S, H, D]

# Reference with PyTorch SDPA
q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, S, D]
k = qkv[:, :, 1].permute(0, 2, 1, 3)
v = qkv[:, :, 2].permute(0, 2, 1, 3)
out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
out_ref = out_ref.permute(0, 2, 1, 3).contiguous()  # -> [B, S, H, D]

max_diff = (out_fa.float() - out_ref.float()).abs().max().item()
print(f"flash_attn vs SDPA | max |diff| = {max_diff:.3e}")
if max_diff >= 2e-2:
    fail(f"flash_attn mismatch too large: {max_diff:.3e}", code=2)
ok("flash_attn micro-forward matches PyTorch reference")

# 3) adam_atan2 import and backend check
try:
    import adam_atan2
    import adam_atan2_backend
    ok("adam_atan2 and backend import OK")
except Exception as e:
    fail(f"adam_atan2 import failed (critical for HRM): {e}")

print("\n🎉 HRM Hello World: PASS")
print("All critical components verified\!")
sys.exit(0)
