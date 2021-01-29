<!--[![Status](https://img.shields.io/badge/Status-InProgress-green.svg)](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/README.md)-->
[![Language](https://img.shields.io/badge/Language-Python3.8-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache-red.svg)](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/LICENSE)


# Semantic Image Synthesis

Photorealistic images creation from semantic segmentation masks, which are labeled sketches that depict the layout of a scene.

The aim is to give realism to the semantic sketch, known as a segmentation map, by automatically adding colours, textures, shadows and reflections, among other details. To this purpose, techniques based on artificial neural networks is used, specifically generative models that allow images to be synthesised without the need to specify a symbolic model in detail. Such synthesised image can also be controlled by a style image that allows the scene’s setting to be changed. For example, a daytime scene can be turned into a sunset. To achieve the style transfer, a variational autoencoder is used and connected to the generative adversarial network in charge of image synthesis.


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
    - `--semantic_label_path` Filename containing the semantic labels.
    - `--img_height` The height size of image.
    - `--img_width` The width size of image.
    - `--crop_size`  Desired size of the square crop.
    - `--batch_size` Mini-batch size.
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
    - `--beta1` Hyperparameter to control the 1st moment decay.
    - `--beta2` Hyperparameter to control the 2nd moment decay.
- **Training**
    - `--total_epochs` Total number of epochs.
    - `--decay_epoch` Epoch from which the learning rate begins to decay linearly to zero.
    - `--prob_dataset` Percentage of the maximum number elements in the dataset that will be used (and shuffled) for training and shuffle on each epoch.
    - `--print_info_freq` Frequency to print information.
    - `--log_dir` Directory name to log losses.
    - `--save_img_freq` Frequency to autosave the fake image, associated segmentation map and real image.
    - `--results_dir` Directory name to save the images.
    - `--save_model_freq` Frequeny to save the checkpoints.
    - `--checkpoint_dir` Directory name to save them.

Please use `python train.py --help` or `python train.py -h` to see all the options.

#### Example

```
$ python train.py --use_vae --image_dir "./datasets/ADE5K/images/" --label_dir "./datasets/ADE5K/annotations/" \
                  --semantic_label_path './datasets/ADE5K/semantic_labels.txt' \
                  --checkpoint_dir './checkpoints/ADE5K_VAE/' --results_dir './results/ADE5K_VAE/train' \
                  --decay_epoch 400 --total_epochs 800 --batch_size 2 --print_info_freq 20
```

![Runtime.](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/images/execution.png)


## Testing

### One image

```
python test_one.py
```

#### Main options:

- `--label_filename` Semantic segmentation mask filename.
- `--style_filename` Style filter filename if use VAE.
- `--use_vae` If specified, enable training with an image encoder.
- `--semantic_label_path` Filename containing the semantic labels.
- `--checkpoint_dir` Directory name to restore the latest checkpoint.
- `--results_dir` Directory name to save the images.

Please use `python test_one.py --help` or `python test_one.py -h` to see all the options.

## Model

![Model.](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/images/method/model.png)

### Architectures
![Image Encoder, Generator, Discriminator.](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/images/method/encoder-generator-discriminator.png)

### SPADE Residual Block
![SPADE ResBlk.](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/images/method/ResBlk.png)

### Spatially-Adaptive Normalization (SPADE)
![Spatially-Adaptive Normalization.](https://github.com/MarAl15/SemanticImageSynthesis/blob/master/images/method/SPADE.png)


## Main references

- _(Park et al.)_ T. Park, M. Liu, T. Wang, J. Zhu. "[Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)"
- [Tensorflow Documentation](https://www.tensorflow.org/)

## Author

Mar Alguacil
