# Jacobian Estimation Module Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Entry Points                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Shell Script:                Python Direct:                     │
│  ./scripts/submission_scripts/ from jacobian_estimation import   │
│  jvp_jacobian_estimation.sh    save_jacobian_samples             │
│                                                                   │
└────────────────┬────────────────────────┬─────────────────────────┘
                 │                        │
                 ▼                        ▼
        ┌────────────────────┐   ┌────────────────────┐
        │  Shell Wrapper     │   │  Python API        │
        │  - Arg parsing     │   │  - Direct import   │
        │  - Help text       │   │  - Full control    │
        │  - Defaults        │   │                    │
        └─────────┬──────────┘   └─────────┬──────────┘
                  │                        │
                  └────────────┬───────────┘
                               ▼
                 ┌──────────────────────────┐
                 │  main_jacobian.py        │
                 │  - Entry point           │
                 │  - Argument validation   │
                 │  - Model loading         │
                 │  - Orchestration         │
                 │  - Metadata saving       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │  jacobian_utils.py       │
                 │  save_jacobian_samples() │
                 │  - Layer iteration       │
                 │  - Hook registration     │
                 │  - JVP computation       │
                 │  - Data persistence      │
                 └────────────┬─────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ quamba/      │ │ quamba/      │ │ Project      │
    │ gptq_utils   │ │ data_loaders │ │ utils        │
    │ - SMGPTQ     │ │ - get_loaders│ │ - build_model│
    │ - GPTQ       │ │              │ │ - set_determ │
    └──────────────┘ └──────────────┘ └──────────────┘
```

## Module Components

### 1. Entry Points

#### Shell Script Entry
```bash
scripts/submission_scripts/jvp_jacobian_estimation.sh
```
- User-friendly interface
- Argument parsing and validation
- Help documentation
- Default values
- Calls Python entry point

#### Python Entry
```python
jacobian_estimation/main_jacobian.py
```
- Argument parsing (argparse)
- Model and tokenizer loading
- Configuration validation
- Orchestrates Jacobian estimation
- Saves metadata

### 2. Core Logic

#### Jacobian Utilities
```python
jacobian_estimation/jacobian_utils.py
```

**Main Function**: `save_jacobian_samples()`

**Workflow**:
1. Setup calibration data loader
2. Capture inputs from first layer
3. For each layer:
   - Register forward hooks
   - Capture inputs/outputs
   - Compute JVPs via SMGPTQ
   - Save results
4. Process lm_head
5. Save all data to disk

**Key Features**:
- Configurable parameters
- Comprehensive logging
- Memory management
- GPU optimization

### 3. Dependencies

#### Internal Dependencies
```python
from jacobian_estimation.jvp_estimator import MambaJacobianEstimator
from quamba.gptq_utils import GPTQ  # Only for lm_head quantization
from quamba.data_loaders import get_loaders
from utils import build_mamba_and_tokenizer, set_deterministic
```

**Note**: The `MambaJacobianEstimator` class (formerly `SMGPTQ`) has been moved to the `jacobian_estimation` module for better organization and clearer naming.

#### External Dependencies
- PyTorch
- mamba_ssm
- transformers
- datasets
- tqdm

## Data Flow

```
Input Model (FP16)
      │
      ▼
[Load Calibration Data]
      │ (WikiText-2)
      ▼
[Capture Layer Inputs]
      │
      ▼
┌─────────────────┐
│ For Each Layer: │
├─────────────────┤
│ 1. Forward pass │
│ 2. Hook capture │
│ 3. JVP compute  │◄──── SMGPTQ class
│ 4. Save data    │
└─────────────────┘
      │
      ▼
[Process lm_head]
      │
      ▼
[Save Metadata]
      │
      ▼
Output Directory
├── metadata.json
├── raw_data/
│   ├── z/
│   ├── x/
│   ├── b/
│   └── c/
├── times/
└── *.png
```

## File Organization

```
jacobian_estimation/
│
├── __init__.py              # Module exports
│   └── Exports: save_jacobian_samples, MambaJacobianEstimator
│
├── main_jacobian.py         # Entry point script
│   ├── parse_arguments()    # CLI argument parsing
│   └── main()               # Main orchestration
│
├── jacobian_utils.py        # Core utilities
│   └── save_jacobian_samples()  # Main Jacobian estimation function
│
├── jvp_estimator.py         # Jacobian estimator class
│   └── MambaJacobianEstimator  # JVP-based Hessian estimator (formerly SMGPTQ)
│       ├── capture_inputs()        # Capture layer inputs
│       ├── capture_outputs()       # Capture layer outputs
│       ├── convert_to_hessian()    # Convert Jacobian to Hessian
│       ├── jvp_gradients_and_hessian_slow()  # JVP computation
│       ├── plot_hessian_estimations()  # Visualization
│       ├── read_and_compare()      # Analysis pipeline
│       └── stitch_plots()          # Aggregate visualizations
│
├── test_jacobian.sh         # Testing script
│   └── Minimal test configuration
│
├── README.md                # User documentation
│   ├── Quick start
│   ├── API reference
│   └── Examples
│
├── MIGRATION_GUIDE.md       # Migration documentation
│   ├── Old vs new approach
│   └── Transition guide
│
└── ARCHITECTURE.md          # This file
    ├── System overview
    └── Technical details
```

## Execution Flow

### 1. Shell Script Execution

```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh model [options]
```

**Steps**:
1. Parse command-line arguments
2. Set default values
3. Build Python command
4. Execute `main_jacobian.py`
5. Report status

### 2. Python Execution

```python
python jacobian_estimation/main_jacobian.py model [options]
```

**Steps**:
1. Parse arguments (argparse)
2. Setup logging
3. Set random seed
4. Load model and tokenizer
5. Call `save_jacobian_samples()`
6. Handle completion (ValueError expected)
7. Save metadata

### 3. Core Processing

```python
save_jacobian_samples(model, tokenizer, device, ...)
```

**Steps**:
1. Build calibration dataloader
2. Allocate input/residual buffers
3. Capture calibration inputs (via Catcher)
4. Process each layer:
   - Create SMGPTQ object
   - Register hooks
   - Run forward pass
   - Compute Jacobians (stitch_plots)
   - Save results
   - Free memory
5. Process final lm_head layer
6. Raise completion signal

## Key Design Decisions

### 1. Separation from Evaluation
- **Why**: Evaluation and Jacobian estimation serve different purposes
- **How**: Dedicated module with own entry points
- **Benefit**: Clear interfaces, no confusion

### 2. Configurable Parameters
- **Why**: Different experiments need different settings
- **How**: CLI arguments, function parameters
- **Benefit**: Flexible experimentation

### 3. Comprehensive Logging
- **Why**: Long-running process, need visibility
- **How**: Python logging module with levels
- **Benefit**: Easy debugging, progress tracking

### 4. Memory Management
- **Why**: Large models can OOM
- **How**: Layer-wise processing, explicit cleanup
- **Benefit**: Handles large models on limited GPUs

### 5. Reuse Existing Infrastructure
- **Why**: Avoid duplication, maintain consistency
- **How**: Import from quamba/ and utils
- **Benefit**: Less code, fewer bugs

## Extension Points

### Adding New Model Types

**Current**: Supports Mamba, Mamba2

**To Add**: GLA, DeltaNet, RetNet

**Steps**:
1. Add model type check in `save_jacobian_samples()`
2. Implement layer structure handling
3. Define appropriate hooks
4. Test with example model

### Adding New Estimation Methods

**Current**: SMGPTQ (Gauss-Newton approximation)

**To Add**: Alternative Jacobian estimation

**Steps**:
1. Create new class in `gptq_utils.py`
2. Add option in `main_jacobian.py`
3. Integrate in `jacobian_utils.py`
4. Document in README

### Adding Visualization

**Current**: Data saved, visualization external

**To Add**: Inline plotting

**Steps**:
1. Add matplotlib integration
2. Create plotting functions
3. Add `--plot` CLI flag
4. Save figures to output dir

## Performance Characteristics

### Time Complexity
- **Per Layer**: O(nsamples × seqlen × d_model²)
- **Total**: O(num_layers × nsamples × seqlen × d_model²)

### Space Complexity
- **GPU**: O(nsamples × seqlen × d_model) for activations
- **Disk**: O(num_layers × probe_samples) for Jacobian data

### Typical Runtime
- Mamba2-130m: ~10-20 minutes (128 samples)
- Mamba2-2.7b: ~1-2 hours (128 samples)
- Mamba2-8b: ~3-4 hours (128 samples)

## Error Handling

### Expected "Errors"
- `ValueError("Jacobian loop over!")` - Normal completion signal

### Actual Errors
- Model loading failures → Clear error message
- OOM → Reduce nsamples or seqlen
- Invalid model type → Validation in argument parsing

## Testing Strategy

### 1. Unit Tests (Manual)
```bash
python -c "from jacobian_estimation import save_jacobian_samples; print('Import OK')"
```

### 2. Integration Test
```bash
./jacobian_estimation/test_jacobian.sh
```

### 3. Full Test
```bash
./scripts/submission_scripts/jvp_jacobian_estimation.sh state-spaces/mamba2-130m
```

## Maintenance Guide

### Adding Features
1. Update function signature if needed
2. Add CLI argument in `main_jacobian.py`
3. Update shell script if needed
4. Update README
5. Test thoroughly

### Fixing Bugs
1. Add logging to identify issue
2. Fix in appropriate module
3. Test with `test_jacobian.sh`
4. Update documentation if needed

### Refactoring
1. Ensure backward compatibility
2. Update imports in `__init__.py`
3. Update all documentation
4. Test all entry points

---

**Note**: This architecture maintains clean separation of concerns while reusing robust existing infrastructure from the Quamba project.

