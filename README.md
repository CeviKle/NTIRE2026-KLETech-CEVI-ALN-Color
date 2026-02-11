
### Create Environment
#### Dependencies and Installation
- Python 3.8
- Pytorch 1.11

1. Create Conda Environment
```
conda create --name watnorm python=3.8
conda activate watnorm
```

2. Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install numpy matplotlib scikit-learn scikit-image opencv-python timm kornia einops pytorch_lightning
```
### Pre-trained Model
- [Our model for NTIRE 2025 Ambient Lighting Normalization Challenge ](https://drive.google.com/drive/folders/15KB4UmYOyd-suBQYQZDx1YXx7xw_flWP?usp=sharing).


### Testing
Download above saved models and unzip it into the folder ./weights. To test the model, you need to specify the test dictionary (Line 15) and model path ( Line 34 and 41) in test.py. Then run
```bash
python test.py 
```
You can check the output in `../results`.

