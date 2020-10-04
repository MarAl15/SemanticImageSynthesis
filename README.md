[![Status](https://img.shields.io/badge/Status-InProgress-green.svg)](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/README.md)
[![Language](https://img.shields.io/badge/Language-Python3.8-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache-red.svg)](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/LICENSE)


# Semantic Image Synthesis

Photorealistic images creation from semantic segmentation masks, which are labeled sketches that depict the layout of a scene.

## Installation

Clone this repo.

```
git clone https://github.com/MarAl15/SemanticImageSynthesis.git
cd SemanticImageSynthesis/
```

Create a virtual environment (recommended).

```
virtualenv --system-site-packages -p python3.8 ./env
source ./env/bin/activate
```

Install dependencies.

```
pip install -r requirements.txt
```

Please note that this code uses Tensorflow 2.3.0 with GPU support. Therefore, you must have CUDA and cuDNN installed beforehand.

_This code was developed and tested on an Nvidia GeForce RTX 3060 with CUDA 10.1 and cuDNN 7.6.5._


## Dataset preparation

The [ADE20K dataset](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) must be downloaded beforehand.

_If the error `Input 'filename' of 'ReadFile' Op has type float32 that does not match expected type of string.` is thrown, create new subdirectories to store them in. For instance,_
```
(Directory structure)
    img_train_path/
    ...train/
    ......train_image_001.jpg
    ......train_image_002.jpg
    ...... ...
    segmask_train_path/
    ...train/
    ......label_image_001.jpg
    ......label_image_002.jpg
    ...... ...
=>
    --image_dir img_train_path
    --label_dir segmask_train_path
```

## Training

```
python train.py
```

### Main options:

- **Load data**
    - `--image_dir` Main directory name where the pictures are located.
    - `--label_dir` Main directory name where the semantic segmentation masks are located.
    - `--img_height` The height size of image.
    - `--img_width` The width size of image.
    - `--crop_size`  Desired size of the square crop.
    - `--batch_size` Input batch size.
- **Image Encoder**
    - `--use_vae` If specified, enable training with an image encoder.
    - `--lambda_kld` Weight for KL Divergence loss.
- **Generator**
    - `--z_dim` Dimension of the latent z vector.
    - `--lambda_features` Weight for feature matching loss.
    - `--lambda_vgg` Weight for VGG loss.
- **Discriminator**
    - `--num_discriminators` Number of discriminators to be used in multiscale.
- **Adam Optimizer**
    - `--lr` Initial learning rate.
    - `--beta1` Exponential decay rate for the 1st moment.
    - `--beta2` Exponential decay rate for the 2nd moment.

Please use `python train.py --help` or `python train.py -h` to see all the options.

## Main references

- _(Park et al.)_ T. Park, M. Liu, T. Wang, J. Zhu. "[Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)"

## Author

Mar Alguacil
