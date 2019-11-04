<p align="center"><img width="40%" src="jpg/logo.jpg" /></p>

--------------------------------------------------------------------------------
This repository adapts from a PyTorch implementation of [StarGAN](https://arxiv.org/abs/1711.09020) for a class project. StarGAN can flexibly translate an input image to any desired target domain using only a single generator and a discriminator. The original repository can be found [here](https://github.com/yunjey/stargan). We extend this work to outdoor scene style transfer, particularly between seasonal/transient attributes and cities.

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)

## Data
We use two datasets in this work: [transient_attributes](http://transattr.cs.brown.edu/) and [world_cities](http://image.ntua.gr/iva/datasets/wc/). For compatability, we convert the datasets to the following format:

data
└── transient_attributes
    ├── train
    |   ├── summer
    |   |   ├── aaa.jpg  (name doesn't matter)
    |   |   └── ...
    |   ├── winter
    |   |   ├── ccc.jpg
    |   |   └── ...
    |   ...
    |
    └── test
        ├── summer
        |   ├── eee.jpg
        |   └── ...
        ├── winter
        |   ├── ggg.jpg
        |   └── ...
        ...
└── world_cities
    ├── train
    |   ├── Amsterdam
    |   |   ├── aaa.jpg
    |   |   └── ...
    |   ...
    |
    └── test
        ├── Amsterdam
        |   ├── fff.jpg
        |   └── ...
        ...

To download the data and convert it to the right format, do the following:

### 1. Download [transient_attributes](http://transattr.cs.brown.edu/) data
On the webpage, scroll down to `Transient Attributes Database` and download all three links provided.

### 2. Convert transient_attributes data to right structure.
Run the following command from the top directory.
```bash
$ python -m code.preprocess_transient
```
### 3. Download [world_cities](http://image.ntua.gr/iva/datasets/wc/) data
...

## Usage

### 1. Training
To train StarGAN on CelebA, run the training script below. See [here](https://github.com/yunjey/StarGAN/blob/master/jpg/CelebA.md) for a list of selectable attributes in the CelebA dataset. If you change the `selected_attrs` argument, you should also change the `c_dim` argument accordingly.

```bash
$ python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
                 --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

To train StarGAN on RaFD:

```bash
$ python main.py --mode train --dataset RaFD --image_size 128 --c_dim 8 \
                 --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
                 --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

To train StarGAN on both CelebA and RafD:

```bash
$ python main.py --mode=train --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
                 --sample_dir stargan_both/samples --log_dir stargan_both/logs \
                 --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To train StarGAN on your own dataset, create a folder structure in the same format as [RaFD](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md) and run the command:

```bash
$ python main.py --mode train --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
                 --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
                 --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
                 --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```

### 2. Testing

To test StarGAN on CelebA:

```bash
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
                 --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

To test StarGAN on RaFD:

```bash
$ python main.py --mode test --dataset RaFD --image_size 128 \
                 --c_dim 8 --rafd_image_dir data/RaFD/test \
                 --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
                 --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

To test StarGAN on both CelebA and RaFD:

```bash
$ python main.py --mode test --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
                 --sample_dir stargan_both/samples --log_dir stargan_both/logs \
                 --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To test StarGAN on your own dataset:

```bash
$ python main.py --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
                 --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
                 --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
                 --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```
### 3. Pretrained model
To download a pretrained model checkpoint, run the script below. The pretrained model checkpoint will be downloaded and saved into `./stargan_celeba_256/models` directory.

```bash
$ bash download.sh pretrained-celeba-256x256
```

To translate images using the pretrained model, run the evaluation script below. The translated images will be saved into `./stargan_celeba_256/results` directory.

```bash
$ python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                 --model_save_dir='stargan_celeba_256/models' \
                 --result_dir='stargan_celeba_256/results'
```
