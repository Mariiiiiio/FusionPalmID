<div align="center">
<h1>FusionPalmID</h1>
<h3>Enhanced Biometric Authentication through Integrated Palm Print and Palm Vein Images</h3>
  [Chi Hung Wang1]<sup>1</sup>, [Wei Ren Chen]<sup>2</sup>, [Jun Jie Yen]<sup>3</sup>, [Xiang Shun Yang]<sup>4</sup>, [Yu Siang Siang]<sup>5</sup>
  
<sup>1</sup> Dept. of Artificial Intelligence Technology and Application, Feng Chia University, Taichung, Taiwan

  
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

In the post-pandemic era, we propose an integrated biometric authentication system leveraging both palm print and palm vein images. Our approach employs RGB lenses for palm print capture and NIR lenses for unforgeable palm vein extraction. Using Real-ESRGAN for palm print preprocessing and Gamma correction for palm vein enhancement, we achieve superior feature fusion with an optimal ratio of 20:80. Experimental evaluations demonstrate that YOLOv12 achieves the best performance with a mAP@50 of 0.964, surpassing traditional CNN and VGG16 models in accuracy, stability, and anti-counterfeiting capabilities.

## Apporach
![image](https://github.com/Mariiiiiio/FusionPalmID/blob/master/img/flowchart.png)


## Directory Structure

```
FusionPalmID/
├── configs/               # Configuration files
│   ├── dataset.yaml      # Dataset configuration
│   └── model_config.yaml # Model hyperparameters
├── src/                  # Source code
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
├── checkpoints/          # Saved model weights
├── results/              # Experimental results
├── YoloV10/
├── YoloV11/
├── YoloV12/
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/palmfusion.git
cd palmfusion
```

2. Set up the environment and install dependencies:

### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda create -n palmfusion python=3.8
conda activate palmfusion

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt

# Install additional Conda packages
conda install -c conda-forge opencv matplotlib pandas seaborn
conda install -c conda-forge jupyter ipykernel
```

### Option 2: Using Python venv
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If you encounter any CUDA compatibility issues, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate version for your system.

## Dataset Description

The dataset follows the YOLOv12 format with both palm print (RGB) and palm vein (NIR) images. Dataset configuration can be found in `configs/dataset.yaml`.

### Download Dataset
The complete dataset is available on Google Drive:
- [Download Dataset](https://drive.google.com/drive/folders/1iJRFnnYTiskpzvhktCwaTgJ-xAfrg4QL?usp=sharing)

The drive contains:
- `raw_data/`: Original palm print and palm vein images
- `yolo_dataset_blended_N10GM/`: Preprocessed and formatted dataset ready for YOLOv12 training
- `best.pt`: Pre-trained model weights from yolo

### Data Format
- Images: `.jpg` format
- Labels: YOLOv12 format text files
- Directory Structure: Follows YOLO convention with `train/`, `val/`, and `test/` splits

## Training & Evaluation

1. Training: 
```bash
python YoloV12/train.py 
```

2. Inference:
```bash
python YoloV12/predict.py --weights checkpoints/yolov12_best.pt --source path/to/image
```

## Results

Our YOLOv12-based model achieves:
- mAP@50: 0.964
- Superior anti-counterfeiting capabilities
- Enhanced feature fusion with 20:80 ratio

### Performance Visualization
<div align="center">
  <img src="https://github.com/Mariiiiiio/FusionPalmID/blob/master/img/Palm_Detection1.jpg" alt="Detection Results" width="800"/>
  <p><em>Figure 1: Detection Results 1</em></p>
</div>

<div align="center">
  <img src="https://github.com/Mariiiiiio/FusionPalmID/blob/master/img/Palm_Detection2.jpg" alt="Detection Results" width="800"/>
  <p><em>Figure 2: Detection Results 2</em></p>
</div>



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
