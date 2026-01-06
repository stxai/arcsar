# Arcsar

## Requirements

- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/username/arcsar.git
   cd arcsar
   ```

2. If you already cloned without `--recursive`, initialize submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```
   Optionally dev dependencies:
   ```bash
   uv sync --extra dev
   ```


## Usage

Segment images:
```bash
uv run python -m arcsar.segmentation.segment
```

## Project Structure

```
arcsar/
├── src/arcsar/          # Main package
│   ├── segmentation/    # Segmentation module using SAM3
│   ├── preprocessing/   # Data preprocessing
│   └── postprocessing/  # Output processing
├── external/            # Git submodules
│   └── sam3/            # SAM3 model
├── tests/               # Unit tests
└── pyproject.toml       # Project configuration
```
