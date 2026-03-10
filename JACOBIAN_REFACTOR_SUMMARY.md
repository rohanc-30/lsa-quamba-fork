# Jacobian Estimation Refactoring Summary

## What Was Done

Successfully created a **clean, dedicated pathway** for Jacobian estimation that is completely independent from the evaluation pipeline (`eval.sh`).

## Problem Statement

Previously, `SM_GPTQ` and `save_jacobian_samples` were hackily integrated into the evaluation pathway:
- Used `eval.sh` with confusing flags
- Mixed evaluation and Jacobian estimation logic
- Hard to maintain and extend
- Unclear purpose from the user interface

## Solution

Created a new `jacobian_estimation/` module with:
1. Dedicated entry point script
2. Refactored utilities
3. Comprehensive documentation
4. Testing infrastructure
5. Updated shell script wrapper

## New File Structure

```
jacobian_estimation/
├── __init__.py                  # Module initialization
├── main_jacobian.py             # Main entry point (168 lines)
├── jacobian_utils.py            # Core utilities (236 lines)
├── test_jacobian.sh             # Testing script
├── README.md                    # Usage documentation
└── MIGRATION_GUIDE.md           # Migration from old approach

scripts/submission_scripts/
└── jvp_jacobian_estimation.sh   # Updated shell wrapper (previously empty)
```

## Usage

### Simple Usage
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m
```

### Advanced Usage
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-2.7b \
    --w_bits 4 \
    --nsamples 256 \
    --seqlen 1024 \
    --output_dir ./my_results \
    --group_heads \
    --verbose
```

### Testing
```bash
# Quick test with minimal compute
./jacobian_estimation/test_jacobian.sh
```

## Key Improvements

### 1. Separation of Concerns ✅
- **Before**: Jacobian code mixed in `quamba/modelutils_mamba.py` (lines 742-931)
- **After**: Dedicated `jacobian_estimation/` module

### 2. Clear Interface ✅
- **Before**: `eval.sh <model> <precision> true false` (confusing!)
- **After**: `jvp_jacobian_estimation.sh <model> [clear options]`

### 3. Configurability ✅
- **Before**: Hardcoded `nsamples=128`, `seqlen=1024`
- **After**: All parameters configurable via CLI

### 4. Documentation ✅
- **Before**: Minimal comments
- **After**: 
  - Comprehensive README
  - Migration guide
  - Docstrings in all functions
  - Inline comments

### 5. Testability ✅
- **Before**: No test infrastructure
- **After**: Dedicated test script

## Technical Details

### Main Entry Point: `main_jacobian.py`

```python
# Features:
- Argument parsing with argparse
- Logging setup
- Model loading via existing utils
- Error handling
- Metadata saving
- Proper exit codes
```

### Core Utilities: `jacobian_utils.py`

```python
def save_jacobian_samples(model, tokenizer, device, w_bits=4, model_type="mamba",
                         nsamples=128, seqlen=1024, output_dir="./jacobian_jvp_error"):
    """
    Refactored from modelutils_mamba.py with:
    - Configurable parameters
    - Better logging
    - Same core SMGPTQ logic
    - Comprehensive documentation
    """
```

### Shell Wrapper: `jvp_jacobian_estimation.sh`

```bash
# Features:
- Help message with usage examples
- Argument parsing
- Default values
- Clear error messages
- Status reporting
```

## Files Modified/Created

### Created (New Files)
- ✨ `jacobian_estimation/__init__.py`
- ✨ `jacobian_estimation/main_jacobian.py`
- ✨ `jacobian_estimation/jacobian_utils.py`
- ✨ `jacobian_estimation/test_jacobian.sh`
- ✨ `jacobian_estimation/README.md`
- ✨ `jacobian_estimation/MIGRATION_GUIDE.md`
- ✨ `JACOBIAN_REFACTOR_SUMMARY.md` (this file)

### Updated
- 📝 `scripts/submission_scripts/jvp_jacobian_estimation.sh` (was empty, now fully implemented)

### Unchanged (Backward Compatibility)
- ⚠️ `quamba/modelutils_mamba.py` (old `save_jacobian_samples` still exists)
- ✅ `eval.sh` (evaluation pipeline unchanged)
- ✅ `main.py` (evaluation entry point unchanged)

## Backward Compatibility

The old pathway still works but is considered **deprecated**:
- Old code in `modelutils_mamba.py` unchanged
- New code should use `jacobian_estimation/`
- Migration guide provided for transition

## Output Structure

Results saved to `jacobian_jvp_error/` (or custom `--output_dir`):
```
jacobian_jvp_error/
├── metadata.json              # Run configuration
├── raw_data/                  # Raw Jacobian samples
│   ├── z/                    # Component-wise data
│   ├── x/
│   ├── b/
│   └── c/
├── times/                     # Timing information
└── *.png                      # Visualization plots
```

## Dependencies

Uses existing project infrastructure:
- ✅ `utils.py`: `build_mamba_and_tokenizer()`, `set_deterministic()`
- ✅ `quamba.gptq_utils`: `GPTQ`, `SMGPTQ` classes
- ✅ `quamba.data_loaders`: `get_loaders()`
- ✅ No new external dependencies

## Testing Instructions

### 1. Quick Test (Minimal Compute)
```bash
./jacobian_estimation/test_jacobian.sh
```
Expected: Creates `test_jacobian_output/` with results

### 2. Full Test (Small Model)
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m \
    --nsamples 64 \
    --verbose
```
Expected: Creates `jacobian_jvp_error/` with full Jacobian data

### 3. Production Run (Large Model)
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-2.7b \
    --w_bits 4 \
    --nsamples 256 \
    --seqlen 1024 \
    --group_heads
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **Clarity** | Purpose obvious from command and directory structure |
| **Modularity** | Self-contained module, easy to import |
| **Flexibility** | All parameters configurable |
| **Maintainability** | Easy to find, modify, and extend code |
| **Testability** | Dedicated test infrastructure |
| **Documentation** | Comprehensive README and migration guide |
| **Discoverability** | Clear entry point, not hidden in eval pipeline |

## Future Enhancements

With this clean structure, future additions are straightforward:

- [ ] Support for additional model types (GLA, DeltaNet, RetNet)
- [ ] Alternative Jacobian estimation methods
- [ ] Parallel processing for batch jobs
- [ ] Advanced visualization options
- [ ] Integration with wandb/tensorboard
- [ ] Automated comparison scripts

## Code Quality

- ✅ No linter errors
- ✅ Consistent style with existing codebase
- ✅ Comprehensive docstrings
- ✅ Clear variable names
- ✅ Modular design
- ✅ Error handling

## Comparison: Old vs New

### Old Approach (Hacky)
```bash
# Confusing command
./eval.sh state-spaces/mamba2-130m fp16 true false

# Goes through:
eval.sh → main.py → quantize_model_mamba() → save_jacobian_samples()
```

### New Approach (Clean)
```bash
# Clear command
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m

# Goes through:
jvp_jacobian_estimation.sh → main_jacobian.py → jacobian_utils.save_jacobian_samples()
```

## Verification Checklist

- ✅ Created `jacobian_estimation/` directory
- ✅ Implemented `main_jacobian.py` entry point
- ✅ Refactored `save_jacobian_samples()` to `jacobian_utils.py`
- ✅ Updated `jvp_jacobian_estimation.sh` shell script
- ✅ Created comprehensive README
- ✅ Created migration guide
- ✅ Created test script
- ✅ Made scripts executable
- ✅ No linter errors
- ✅ Backward compatible (old code still works)
- ✅ Documented all changes

## Success Criteria Met

1. ✅ **Separate pathway**: Completely independent from `eval.sh`
2. ✅ **Clear interface**: Obvious purpose and usage
3. ✅ **Self-contained**: All code in `jacobian_estimation/`
4. ✅ **Well-documented**: README, migration guide, docstrings
5. ✅ **Testable**: Test script provided
6. ✅ **Maintainable**: Clean, modular code structure

## Summary

Successfully transformed the "hacky" Jacobian estimation pathway into a **professional, maintainable, and extensible module** that:
- Has a clear purpose
- Is easy to use
- Is well-documented
- Is easy to extend
- Follows software engineering best practices

The new `jacobian_estimation/` module is now a **first-class citizen** in the codebase, rather than a hidden hack in the evaluation pipeline.

---

**Status**: ✅ **COMPLETE** - Ready for use and testing!

