# Installation Guide

## PyTorch

You can install PyTorch with CPU-only or GPU support. For this assignment, CPU-only support is sufficient, but GPU support will offer better performance if you have an NVIDIA GPU.

### CPU-only version
Visit [PyTorch Installation Page](https://pytorch.org/get-started/locally/), select `CPU` and `Linux`, and follow the instructions provided.

### GPU version (NVIDIA GPUs)
Due to different GPU types and CUDA versions, please contact your tutor for installation assistance.

### Verify GPU Installation
If using the GPU version, verify your installation with:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Additional Libraries

Install required libraries using pip:

```bash
pip install opencv-python pillow scipy numpy
```

 
# Dataset Download
- [Google Drive](https://drive.google.com/drive/folders/1ZlGBDe9RKQqffznb6k4C6Px3ws5Sr_5A?usp=sharing)

## Where to put the data
- Put all three files at the root of this project

```
<root>
|---README.md
|---train_final.py
|---vis_utils.py
|---network.py
|---dataset.py
|---valid.p
|---train.p
|---test.p
|---results(where you could find visualizations
|   |--- ...
|---checkpoint(where the model weight saved)
    |--- ...
```

# Training and Testing
```bash
python train_final.py
```
- There are multiple parameters that you could change, more details on the meaning of the parameters will be discussed within the lab. If you are unsure, just leave it as default.
- This script will first train the model and doing testing over the test dataset, and it will provide you all metrics and visualization needed as long as all TODOs are fixed. 
