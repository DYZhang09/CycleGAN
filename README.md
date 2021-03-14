# Simple CycleGAN Implementation
## Introduction
This repository is a simplified CycleGAN implementation aiming for image translation
such as horse-to-zebra(may has some bugs).

The official implementation is [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git)
## file structure
```
.
├── data                // the datasets
│   └── horse2zebra     // the dataset(must has 4 subfolders(testA, testB, trainA, trainB))  
│       ├── testA       
│       ├── testB
│       ├── trainA
│       └── trainB
├── dataIO              // the code of dataset I/O module 
├── model               // the code of definition of CycleGAN model
└── weights             // the pretrained weight file(without identity loss)
└── test.py             // the code for testing
└── train.py            // the code for training
```
## how to train
use `python train.py --dataroot ./data/horse2zebra [OPTIONS]`

some of the options are listed below, you can use `python train.py -h` for more information.
```
    --dataroot      the root path of the dataset, must have 4 subfolders: trainA, trainB, testA, testB
    --weight_file   where to store the temporary weights
    --res_layers    the num of Resnet blocks in CycleGan
    --batch_size    the batch size of input images
    --shuffle       whether to shuffle input images
    --crop_size     the crop size
    --crop_pos      the crop position
    --flip          whether to flip input images horizontally
    --grayscale     whether to read images as gray images
    --resize        the size of images after resizing
    --lr            the learning rate
    --gpu           whether to use gpu to train
    --epochs        the num of training epochs
```

## how to test
use `python test.py --dataroot ./your/dataset [OPTIONS]` and the result will reside in `dataroot/fake/`.

most options are same as training options except `--lr  --epochs`, use `python test.py -h` for more information.


