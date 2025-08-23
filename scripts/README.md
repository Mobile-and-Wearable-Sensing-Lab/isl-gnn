# ISL ST-GCN Training

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update the configuration file `config.yaml`:
   - Set `data_root` to your H5 dataset path
   - Adjust hyperparameters as needed
   - Configure WandB settings (project name, entity)

## Training

Run training with default config:
```bash
python train.py
```

Or specify a custom config file:
```bash
python train.py --config path/to/config.yaml
```

## File Structure

- `train.py` - Main training script with train/eval loops
- `utils.py` - Helper functions for metrics, logging, and data processing
- `config.yaml` - All hyperparameters and settings
- `isl_dataloader.py` - Your existing dataloader (loads H5 files)
- `st_gcn.py` - Your existing ST-GCN model

## Key Features

- **Data Processing**: Automatically extracts x,y coordinates from 3D landmarks (ignores z)
- **Learning Rate**: Cosine annealing scheduler
- **Logging**: WandB integration for tracking metrics and confusion matrices
- **Metrics**: Top-1 and Top-5 accuracy, loss tracking
- **Checkpointing**: Saves best model based on validation accuracy

## Data Shape Handling

The pipeline handles the coordinate transformation:
1. Input from dataloader: `(batch, frames, 126)` where 126 = 2 hands × 21 landmarks × 3 coords
2. Reshape and extract x,y: `(batch, frames, 84)` where 84 = 2 hands × 21 landmarks × 2 coords
3. Model processes this as 2D hand landmarks

## Output

- **Checkpoints**: Saved in `./checkpoints/`
  - `best_model.pth` - Best validation accuracy
  - `final_model.pth` - Final epoch model
- **Results**: Confusion matrix saved in `./results/`
- **WandB**: All metrics and plots logged to your WandB project

## Monitoring Training

Training progress is displayed in the terminal with:
- Real-time loss and accuracy
- Progress bars for each epoch
- Validation metrics every epoch

WandB dashboard will show:
- Training/validation curves
- Learning rate schedule
- Confusion matrix
- All hyperparameters