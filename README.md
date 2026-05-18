# YOLO11 SAM2 Cell Tracker
Automated cell tracking in microscopy videos using YOLO11 detection and SAM 2 segmentation, evaluated on the Cell Tracking Challenge benchmark.

## Overview

This project combines:
- **YOLO11** for fast and accurate cell detection in microscopy images
- **SAM 2.1 (Segment Anything Model 2)** for cell segmentation and instance segmentation

The pipeline outputs tracking masks in Cell Tracking Challenge (CTC) format, with optional visualization of detections and segmentations.

## Requirements

- **Python 3.8+**
- **CUDA-capable GPU with ≥16GB VRAM** (required for SAM 2.1 Hiera-Large)
- **PyTorch with CUDA support**
- All dependencies listed in `requirements.txt`

## Project Structure

### Core Components
- `run_ctc_yolo_sam2_video_pipeline.py` - Main pipeline script
- `run_all_ctc_dataset.ipynb` - Batch processing notebook for all CTC datasets
- `sam2/sam2_yolo_video_cell_tracker.py` - Custom SAM 2 + YOLO integration

### Models & Checkpoints
- `yolo_best_model/best.pt` - Trained YOLO11 detector
- `checkpoints/sam2.1_hiera_large.pt` - SAM 2.1 Hiera-Large segmentation model

### Training & Hyperparameter Tuning
- `yolo_training/train_yolo_with_best_hyperparams.ipynb` - Training with optimized hyperparameters
- `yolo_training/hyperparam_tune_ray.ipynb` - Hyperparameter search using Ray
- `yolo_training/split_data.ipynb` - Dataset preparation and splitting

### Evaluation Tools
- `measures_scripts/` - Cell Tracking Challenge benchmark evaluation (DETMeasure, SEGMeasure, TRAMeasure)

### Datasets

All the datasets from challenge should be stored inside `./data/<dataset_name>`.

## How to Run

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\bin\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure CUDA Device
```bash
set CUDA_VISIBLE_DEVICES=0  
```

### 4. Run Full Pipeline for the Cell Tracking Challenge sumbission
```bash
python run_ctc_yolo_sam2_video_pipeline.py <dataset_folder> <output_folder>
```

**Example:**
```bash
python run_ctc_yolo_sam2_video_pipeline.py data/BF-C2DL-HSC/01 data/BF-C2DL-HSC/01_RES
```

### 5. Minimal Demo - Core Pipeline

Quick demo for obtaining masks:

```python
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_yolo_video_cell_tracker import SAM2YOLOVideoPredictor, make_yolo_detector
from run_ctc_yolo_sam2_video_pipeline import convert_tif_to_jpg, generate_ctc_output

# Paths
dataset_folder = "data/BF-C2DL-HSC/01"

# Build models
predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", 
                                       "./checkpoints/sam2.1_hiera_large.pt")
predictor.__class__ = SAM2YOLOVideoPredictor
detector_fn = make_yolo_detector("./yolo_best_model/best.pt", detection_conf_threshold=0.4)

# Convert and process
input_folder = convert_tif_to_jpg(dataset_folder)
inference_state, video_frames = predictor.init_state_with_detector(
    frames_dir=input_folder, detector_fn=detector_fn, offload_state_to_cpu=False
)

# Run inference and save
video_segments = {}
for frame_idx, obj_ids, masks, yolo_boxes in predictor.propagate_in_video(
    inference_state, detector_fn=detector_fn, video_frames=video_frames
):
    masks = (masks > 0.0).cpu().numpy()
    video_segments[frame_idx] = {obj_ids[i]: masks[i][0] for i in range(len(obj_ids))}

# Remapping stage
video_segments = remap_and_handle_merges(video_segments, inference_state)

```

### Obtaining Masks

The pipeline produces instance segmentation masks for each frame:

- **`video_segments` dictionary** - Dictionary mapping `frame_index` → `{object_id: binary_mask}`
- **Mask format** - NumPy boolean arrays 
- **Frame-by-frame access** - Iterate through `video_segments` to process masks individually

Example:
```python
# Access masks directly
for frame_idx, objects in video_segments.items():
    for obj_id, mask in objects.items():
        # mask is a binary 2D array (H x W)
        area = mask.sum()  # Number of pixels
        coordinates = np.where(mask)  # Get pixel locations
```

## Training YOLO11 Detector

### Step 1: Prepare and Split Data

Run the data splitting notebook to organize your training data into train/val/test sets in notebook `yolo_training/split_data.ipynb`.

This notebook:
- Loads raw images and their annotations
- Splits data into training (70%), validation (15%), and test (15%) sets
- Creates YOLO-compatible dataset structure (images and labels directories)
- Generates a `data.yaml` configuration file for training

**Output structure:**
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Step 2: Find Best Hyperparameters (Optional)

Use Ray Tune for automated hyperparameter search in notebook `yolo_training/hyperparam_tune_ray`.

This notebook:
- Tests multiple hyperparameter combinations (learning rate, batch size, augmentation, etc.)
- Trains temporary models and evaluates on validation set
- Logs results to track the best configuration
- Saves best hyperparameters to `best_hyperparams.yaml`

### Step 3: Train with Best Hyperparameters

Train the final YOLO11 model using the best configuration in `train_yolo_with_best_hyperparams.ipynb`

Notebook:
- Loads best hyperparameters from previous tuning
- Trains YOLO11 for multiple epochs
- Validates on validation set at each epoch
- Saves checkpoints and monitors training metrics (precision, recall, mAP)
- Outputs `best.pt` in `yolo_training/training_results/yolo_best_model` directory

### Quick Train without Hyperparameter Tuning

Skip hyperparameter search and train directly:

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO('yolo11l.pt')

# Train with default hyperparameters
results = model.train(
    data='./ctc_merged_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    epochs=150,
    device=0  # GPU device
)

# Save the trained model
model.save('yolo_best_model/best.pt')
```
