# HRM (Hierarchical Reasoning Model) Automated Installer

This repository contains production-ready installation scripts for the Hierarchical Reasoning Model (HRM) on Ubuntu servers with NVIDIA GPUs.

## Overview

HRM is a novel recurrent architecture for complex reasoning tasks. This installer automates the complete setup process including all dependencies, CUDA toolkit, C++ extensions, and verification.

## Key Features

- **Fully Automated**: Complete hands-off installation
- **Fail-Fast Design**: Loud failures at critical points, never masks errors
- **tmux Integration**: Handles long-running processes without timeouts
- **Conditional CUDA Installation**: Only installs CUDA toolkit if not present
- **C++ Extension Building**: Compiles adam_atan2 and flash_attn dependencies
- **Verification Testing**: Runs actual HRM training to confirm installation

## Prerequisites

- Ubuntu 22.04 LTS
- NVIDIA GPU (tested on RTX 4000 Ada)
- Python 3.10
- At least 20GB free disk space
- Internet connection for downloading packages

## Installation

1. Clone this repository to :
   ```bash
   cd /opt
   git clone https://github.com/lbrichards/linode-install.git
   ```

2. Run the installer:
   ```bash
   cd /opt/linode-install/scripts
   sudo bash install.sh
   ```

The installation will take approximately 15-20 minutes and includes:
- CUDA Toolkit 12.8 installation (if needed)
- PyTorch 2.5 with CUDA support
- All HRM Python dependencies
- C++ extension compilation (adam_atan2)
- Flash Attention installation (prebuilt wheel)
- Sudoku dataset generation
- Verification test

## Critical Dependencies Resolved

### 1. CUDA Toolkit
- **Issue**: PyTorch wheels include CUDA runtime but not the compiler (nvcc)
- **Solution**: Conditionally installs CUDA toolkit 12.8 from NVIDIA repos
- **Time**: 5-10 minutes (only on first install)

### 2. adam_atan2 C++ Extension
- **Issue**: Requires compilation with CUDA support
- **Solution**: Builds from source after CUDA toolkit installation
- **Verification**: Tests import of 

### 3. Flash Attention
- **Issue**: Complex C++ build often fails
- **Solution**: Uses official prebuilt wheels for PyTorch 2.5
- **Note**: Requires downgrade from PyTorch 2.8 to 2.5 for wheel compatibility

## Architecture Details

- **GPU**: NVIDIA RTX 4000 Ada (Compute Capability 8.9)
- **CUDA**: Version 12.8
- **PyTorch**: Version 2.5.* with CUDA 12.4 support
- **Python**: 3.10
- **Virtual Environment**: 

## Usage After Installation

```bash
# Activate the virtual environment
source /opt/venv/bin/activate

# Navigate to HRM directory
cd /opt/HRM

# Run training (with WandB disabled)
WANDB_MODE=offline python pretrain.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=100 \
    eval_interval=50 \
    global_batch_size=384
```

## Uninstallation

For testing or cleanup:
```bash
cd /opt/linode-install/scripts
sudo bash uninstall.sh
```

This removes:
- Python virtual environment ()
- Generated datasets
- Training logs
- **Note**: Does NOT remove CUDA toolkit or system packages

## Troubleshooting

### Common Issues

1. **"CUDA_HOME: None"**
   - The CUDA toolkit needs to be installed
   - The installer handles this automatically

2. **"ModuleNotFoundError: adam_atan2_backend"**
   - C++ extension not built properly
   - Ensure CUDA toolkit is installed first

3. **"ImportError: flash_attn"**
   - Flash attention not compatible with PyTorch version
   - Script downgrades to PyTorch 2.5 for compatibility

### Verification

The installer includes automatic verification that:
1. All Python modules import correctly
2. CUDA is available to PyTorch
3. HRM training actually runs

## Technical Implementation

### Key Design Decisions

1. **tmux for Long Processes**: CUDA installation and dataset generation run in tmux sessions to avoid SSH timeouts

2. **Fail-Fast Philosophy**: Every critical step has explicit error checking with clear failure messages

3. **Prebuilt Wheels**: Uses official Flash Attention wheels instead of compilation (faster, more reliable)

4. **Conditional Installation**: Checks for existing CUDA/packages to make re-runs faster

### File Structure

```
/opt/linode-install/
├── README.md          # This file
└── scripts/
    ├── install.sh     # Main installation script
    └── uninstall.sh   # Cleanup script
```

## Development Notes

### Build Requirements
- CUDA Toolkit for nvcc compiler
- GCC for C++ compilation  
- Python development headers
- ~10GB for CUDA toolkit installation
- ~8GB for Python packages and models

### Testing Workflow
1. Run  to clean environment
2. Run  to test full installation
3. Verify with sample training run

## Credits

Developed for automated deployment of HRM (Hierarchical Reasoning Model) on cloud GPU instances.

## License

See LICENSE file in repository.
