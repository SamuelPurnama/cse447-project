#!/usr/bin/env python
"""Verify KV-cache efficiency by counting model forward passes"""
import sys
import os
sys.path.insert(0, 'src')
os.environ['CUDA_HOME'] = os.environ.get('CONDA_PREFIX', '/usr/local/cuda')

from myprogram import MyModel
import torch

# Monkey-patch the model's forward to count calls
original_forward = None
forward_count = 0
cache_hit_count = 0
single_token_attempts = 0
single_token_fallbacks = 0

def counting_forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
    global forward_count, cache_hit_count, single_token_attempts, single_token_fallbacks
    forward_count += 1
    
    seq_len = input_ids.size(1) if hasattr(input_ids, 'size') else len(input_ids[0])
    
    if past_key_values is not None:
        cache_hit_count += 1
    
    if seq_len == 1:
        single_token_attempts += 1
        if past_key_values is not None:
            # This will likely fail with Bolmo's bug
            pass
    
    return original_forward(input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

# Load model
print("Loading model...")
model = MyModel.load("work")

# Patch the model
original_forward = model.model.forward
model.model.forward = lambda *args, **kwargs: counting_forward(model.model, *args, **kwargs)

# Test case
text = "他是我最好的朋"
print(f"\nInput: '{text}'")
print(f"Expected: '友' + 2 more")
print("="*70)

# Reset counters
forward_count = 0
cache_hit_count = 0
single_token_attempts = 0

# Run prediction
result = model._next_char_top3(text, beam_width=8)

print(f"\n{'='*70}")
print(f"Result: '{result}'")
print(f"{'='*70}")
print(f"\nKV-Cache Efficiency Metrics:")
print(f"  Total forward passes:        {forward_count}")
print(f"  Cache hits (past_kv used):   {cache_hit_count}")
print(f"  Single-token attempts:       {single_token_attempts}")
print(f"  Cache efficiency:            {cache_hit_count / forward_count * 100:.1f}%")
print(f"\nInterpretation:")
if cache_hit_count > 0:
    print(f"  ✓ KV-cache is being used!")
    print(f"  ✓ {cache_hit_count} out of {forward_count} forwards used cache")
else:
    print(f"  ✗ KV-cache is NOT being used (all forwards from scratch)")

# Estimate without cache
print(f"\nEstimated forward passes without KV-cache optimization:")
print(f"  Naive approach (rebuild full sequence each time): ~{forward_count * 2-3}+ passes")
print(f"  With prefix caching: {forward_count} passes")
if cache_hit_count > 0:
    print(f"  ✓ Savings: ~{(1 - forward_count / (forward_count * 2)) * 100:.0f}% reduction")
