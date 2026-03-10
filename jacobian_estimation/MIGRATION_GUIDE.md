# Migration Guide: From Hacky Eval Path to Dedicated Jacobian Pathway

## Overview

This document explains the migration from the old "hacky" approach where Jacobian estimation was embedded in the evaluation pipeline to the new clean, dedicated pathway.

## What Changed?

### Old Approach (Hacky) ❌

**Problem**: Jacobian sampling was awkwardly integrated into the evaluation pipeline:

```bash
# Had to use eval.sh with special flags
eval.sh <model> <precision> --apply_gptq true

# This would:
# 1. Go through main.py
# 2. Call quantize_model_mamba()
# 3. Which would call save_jacobian_samples() if apply_gptq=true
# 4. Mix evaluation logic with Jacobian estimation
```

**Issues**:
- ❌ Confusing: GPTQ flag used for Jacobian estimation
- ❌ Coupled: Jacobian code mixed with quantization code
- ❌ Limited: Hard to customize Jacobian-specific parameters
- ❌ Unclear: Purpose not obvious from command-line interface

**Code Location**:
- Entry: `eval.sh` → `main.py` → `quamba/modelutils_mamba.py`
- Function: `save_jacobian_samples()` inside `modelutils_mamba.py` (lines 742-931)
- Called from: `quantize_model_mamba()` when `args.apply_gptq == True`

### New Approach (Clean) ✅

**Solution**: Dedicated pathway completely separate from evaluation:

```bash
# Clear, dedicated entry point
./scripts/submission_scripts/jvp_jacobian_estimation.sh <model> [options]

# This:
# 1. Calls jacobian_estimation/main_jacobian.py directly
# 2. Uses jacobian_estimation/jacobian_utils.py
# 3. Has its own clear parameters
# 4. Completely independent from eval pipeline
```

**Benefits**:
- ✅ Clear: Obvious purpose and usage
- ✅ Modular: Separate module with dedicated utilities
- ✅ Flexible: Easy to customize Jacobian-specific parameters
- ✅ Maintainable: Easy to extend and modify

**Code Location**:
- Entry: `scripts/submission_scripts/jvp_jacobian_estimation.sh`
- Main: `jacobian_estimation/main_jacobian.py`
- Utils: `jacobian_estimation/jacobian_utils.py`
- Module: `jacobian_estimation/` (self-contained)

## File Structure Comparison

### Old Structure
```
lsa-quamba-fork/
├── eval.sh                          # Mixed eval + Jacobian entry
├── main.py                          # Mixed purposes
├── quamba/
│   ├── modelutils_mamba.py         # Jacobian code buried here (lines 742-931)
│   └── gptq_utils.py               # SMGPTQ class used by Jacobian
└── scripts/
    └── submission_scripts/
        └── jvp_jacobian_estimation.sh  # Empty file!
```

### New Structure
```
lsa-quamba-fork/
├── eval.sh                          # Pure evaluation (unchanged)
├── main.py                          # Pure evaluation (unchanged)
├── jacobian_estimation/             # NEW: Dedicated module
│   ├── __init__.py                 # Module initialization
│   ├── main_jacobian.py            # Entry point for Jacobian estimation
│   ├── jacobian_utils.py           # Core Jacobian utilities (refactored)
│   ├── test_jacobian.sh            # Testing script
│   ├── README.md                   # Documentation
│   └── MIGRATION_GUIDE.md          # This file
├── quamba/
│   ├── modelutils_mamba.py         # Can keep old code for backward compat
│   └── gptq_utils.py               # SMGPTQ class (shared)
└── scripts/
    └── submission_scripts/
        └── jvp_jacobian_estimation.sh  # Now properly implemented!
```

## API Comparison

### Old Way (via eval.sh)

```bash
# Had to abuse the eval.sh script
./eval.sh state-spaces/mamba2-130m fp16 true false
#         └─model                  └─precision └─apply_gptq └─hadamard
#                                                ^^^ This triggered Jacobian!
```

**Issues**:
- Confusing parameter order
- GPTQ flag misleading (it does Jacobian, not GPTQ)
- Can't customize nsamples, seqlen, output_dir
- Mixed with evaluation flags

### New Way (dedicated script)

```bash
# Clear, dedicated command
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m \
    --w_bits 4 \
    --nsamples 128 \
    --seqlen 1024 \
    --output_dir ./my_results \
    --verbose
```

**Benefits**:
- Self-documenting command
- All Jacobian-specific parameters available
- No confusion with evaluation
- Built-in help: run without args for usage

## Code Changes

### Refactored Function

The `save_jacobian_samples()` function was:
- **Extracted** from `quamba/modelutils_mamba.py`
- **Refactored** with better parameters
- **Documented** with comprehensive docstrings
- **Relocated** to `jacobian_estimation/jacobian_utils.py`

Key improvements:
```python
# Old signature (limited parameters)
def save_jacobian_samples(model, tokenizer, device, w_bits=4, model_type="mamba"):
    # Hardcoded nsamples=128, seqlen=1024
    nsamples = 128
    seqlen = 1024
    # ...

# New signature (flexible parameters)
def save_jacobian_samples(model, tokenizer, device, w_bits=4, model_type="mamba",
                         nsamples=128, seqlen=1024, output_dir="./jacobian_jvp_error"):
    """
    Comprehensive docstring explaining purpose, parameters, returns, notes
    """
    # Configurable parameters
    # Better logging
    # Same core logic
```

### New Entry Point

Created `jacobian_estimation/main_jacobian.py`:
- Clean argument parsing (argparse)
- Dedicated logging setup
- Metadata saving
- Error handling
- Proper exit codes

## Migration Path

### For Users

**Before** (Old way):
```bash
./eval.sh state-spaces/mamba2-130m fp16 true false
```

**After** (New way):
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m
```

### For Developers

**Before** (Old way):
```python
from quamba.modelutils_mamba import save_jacobian_samples

# Limited control
save_jacobian_samples(model, tokenizer, "cuda", w_bits=4, model_type="mamba2")
```

**After** (New way):
```python
from jacobian_estimation import save_jacobian_samples

# Full control
save_jacobian_samples(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    w_bits=4,
    model_type="mamba2",
    nsamples=256,      # Now configurable!
    seqlen=2048,       # Now configurable!
    output_dir="./my_custom_output"  # Now configurable!
)
```

## Backward Compatibility

The old pathway still exists in `quamba/modelutils_mamba.py` for backward compatibility, but:
- **Deprecated**: Consider it deprecated
- **Undocumented**: Not in official docs
- **Use New Way**: All new code should use `jacobian_estimation/`

To fully clean up (optional):
1. Search for references to old `save_jacobian_samples` in `modelutils_mamba.py`
2. Remove the function from there (lines 742-931)
3. Update any imports to use `jacobian_estimation.jacobian_utils`

## Testing the New Pathway

### Quick Test
```bash
# Run minimal test
./jacobian_estimation/test_jacobian.sh
```

### Full Test
```bash
# Run on a small model
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m \
    --nsamples 128 \
    --verbose
```

### Production Run
```bash
# Run on larger model with full settings
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-2.7b \
    --w_bits 4 \
    --nsamples 256 \
    --seqlen 1024 \
    --output_dir ./jacobian_results_2.7b \
    --group_heads \
    --verbose
```

## Benefits Summary

| Aspect | Old (Hacky) | New (Clean) |
|--------|-------------|-------------|
| **Clarity** | ❌ Confusing | ✅ Clear purpose |
| **Modularity** | ❌ Mixed with eval | ✅ Separate module |
| **Flexibility** | ❌ Hardcoded params | ✅ Fully configurable |
| **Documentation** | ❌ Minimal | ✅ Comprehensive |
| **Testing** | ❌ No test script | ✅ Dedicated tests |
| **Maintainability** | ❌ Hard to modify | ✅ Easy to extend |
| **Discoverability** | ❌ Hidden in eval | ✅ Clear entry point |

## Future Enhancements

With this new clean structure, it's now easy to add:
- [ ] Support for other model types (GLA, DeltaNet, RetNet)
- [ ] Different Jacobian estimation strategies
- [ ] Parallel processing for multiple models
- [ ] Advanced visualization options
- [ ] Integration with other analysis tools
- [ ] Batch processing scripts

## Questions?

For questions or issues with the new pathway:
1. Check `jacobian_estimation/README.md` for usage
2. Run test script: `./jacobian_estimation/test_jacobian.sh`
3. Enable verbose logging: `--verbose` flag
4. Check output in `jacobian_jvp_error/metadata.json`

---

**Summary**: The new dedicated pathway makes Jacobian estimation a first-class citizen with its own clean interface, rather than a hack bolted onto the evaluation pipeline. This improves clarity, maintainability, and extensibility.

