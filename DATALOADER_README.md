# ISL Video Dataset PyTorch DataLoader

A comprehensive PyTorch DataLoader for Indian Sign Language (ISL) video datasets stored in HDF5 format.

## Dataset Structure

The dataset is organized as follows:
```
h5_output/
├── train/          (50 classes, 626 samples)
├── test/           (49 classes, 178 samples)
└── validation/     (42 classes, 68 samples)
```

Each split contains subdirectories named after ISL classes (e.g., "1. Dog", "2. Death", etc.), and each class directory contains `.h5` files with processed video data.

### H5 File Structure
Each `.h5` file contains:
- `hand_landmarks`: (num_frames, 2, 21, 3) - 2 hands, 21 landmarks each, 3D coordinates
- `world_landmarks`: (num_frames, 2, 21, 3) - same structure in world coordinates  
- `frame_metadata`: (num_frames,) - metadata for each frame
- `file_metadata`: Additional file-level metadata

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from scripts.isl_dataloader import create_data_loaders

# Create dataloaders for all splits
dataloaders = create_data_loaders(
    root_dir='h5_output',
    batch_size=16,
    num_workers=4,
    use_world_landmarks=True,
    use_hand_landmarks=True,
    normalize=True,
    max_frames=150
)

# Use in training loop
for data, labels, lengths, metadata in dataloaders['train']:
    # data: (batch_size, max_seq_len, num_features)
    # labels: (batch_size,) class indices
    # lengths: (batch_size,) actual sequence lengths
    # metadata: list of dicts with additional info
    pass
```

### Advanced Usage

```python
from scripts.isl_dataloader import ISLVideoDataset

# Create custom dataset
dataset = ISLVideoDataset(
    root_dir='h5_output',
    split='train',
    use_world_landmarks=True,
    use_hand_landmarks=False,  # Use only world landmarks
    normalize=True,
    max_frames=100
)

# Get dataset info
print(f"Number of classes: {dataset.get_num_classes()}")
print(f"Class names: {dataset.get_class_names()}")

# Access single sample
data, label, metadata = dataset[0]
print(f"Data shape: {data.shape}")
print(f"Class: {metadata['class_name']}")
```

## Features

### Data Loading
- **Variable-length sequences**: Handles videos with different frame counts
- **Flexible feature selection**: Choose between hand landmarks, world landmarks, or both
- **Automatic padding**: Sequences are padded to batch maximum length
- **Normalization**: Optional per-sequence normalization of landmark coordinates

### Dataset Information
- **Train**: 626 samples, 50 classes
- **Test**: 178 samples, 49 classes  
- **Validation**: 68 samples, 42 classes
- **Feature dimensions**: 
  - Hand landmarks only: 126 features (2 hands × 21 landmarks × 3 coords)
  - World landmarks only: 126 features
  - Both: 252 features

### Memory Efficiency
- **Lazy loading**: H5 files are loaded only when needed
- **Configurable workers**: Multi-process data loading support
- **GPU memory optimization**: Automatic pin_memory when CUDA available

## Parameters

### `create_data_loaders()`
- `root_dir`: Path to h5_output directory
- `batch_size`: Batch size for DataLoader (default: 32)
- `num_workers`: Number of worker processes (default: 4)
- `use_world_landmarks`: Include world landmarks (default: True)
- `use_hand_landmarks`: Include hand landmarks (default: True)
- `normalize`: Normalize landmark coordinates (default: True)
- `max_frames`: Maximum frames per sequence (default: None)
- `shuffle_train`: Shuffle training data (default: True)

### `ISLVideoDataset()`
- `root_dir`: Path to h5_output directory
- `split`: Dataset split - 'train', 'test', or 'validation'
- `use_world_landmarks`: Include world landmarks (default: True)
- `use_hand_landmarks`: Include hand landmarks (default: True)
- `normalize`: Normalize coordinates (default: True)
- `max_frames`: Maximum frames per sequence (default: None)

## Training Example

See `train_example.py` for a complete training example with:
- Simple LSTM classifier
- Training and validation loops
- Progress tracking with tqdm
- Model saving and loading

```bash
python train_example.py
```

## Data Format

### Input Tensor Shape
- **Batch**: `(batch_size, max_seq_len, num_features)`
- **Features**: 
  - Hand landmarks: 126 (2 hands × 21 landmarks × 3 coordinates)
  - World landmarks: 126 (2 hands × 21 landmarks × 3 coordinates)
  - Combined: 252 features

### Labels
- **Type**: Integer class indices (0 to num_classes-1)
- **Mapping**: Available via `dataset.class_to_idx` and `dataset.idx_to_class`

### Metadata
Each sample includes metadata:
```python
{
    'file_path': '/path/to/file.h5',
    'class_name': '1. Dog',
    'num_frames': 76,
    'frame_metadata': array([...]),
    'file_metadata': {...}
}
```

## Performance Tips

1. **Batch Size**: Start with smaller batches (8-16) due to variable sequence lengths
2. **Workers**: Use 2-4 workers to avoid I/O bottlenecks
3. **Max Frames**: Set reasonable max_frames (100-200) to control memory usage
4. **Normalization**: Enable normalization for better training stability
5. **GPU**: Use `pin_memory=True` for faster GPU transfers (automatically enabled)

## Error Handling

The dataloader includes comprehensive error handling:
- Missing directory validation
- Empty class directories
- Corrupted H5 files
- Invalid parameter combinations

## License

This dataloader is provided as-is for research and educational purposes.
