# Introduction
This repository contains the code for the proposed network compression pipeline.
Its functions can compress the large DNN to a small one with high efficiency and accuracy.
We develop the algorithm based on the open-source maskrcnn-benchmark in https://github.com/facebookresearch/maskrcnn-benchmark.

# Installation
## Requirements
- ubuntu 14.04
- python3 with anaconda
- pytorch 1.4
- torchvision
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0

## Build
- `git clone git@github.com:bucmsunxin/herb_image_compress.git` to clone the repository.
- `cd herb_image_compress` to go into the repository.
- run `python3 setup.py build develop` to build the library.

# Description
## Project tree
```{}
|- herb_image_compress/              # root repository folder
   |- configs/                          # config files
      |- e2e_R50_1x.yaml                   # config file for the large DNN - ResNet-50
      |- e2e_MV2_1x.yaml                   # config file for the small DNN - MobileNet-V2
      |- e2e_MV2_1x_transfer.yaml          # config file for the network transfer
      |- e2e_MV2_1x_cut.yaml               # config file for the network cut
   |- gen_med_db/                       # generate the dataset file
      |- gen_db.py                         # generate the dataset db file, must have the "images" folder under the data dir!
      |- util.py                           # some auxiliary functions for reading and writing files
   |- maskrcnn_benchmark/               # core functions
      |- config                            # define all the default values
      |- csrc                              # c++/cuda implementation for some layers
      |- data                              # data preparation and pre-processing
         |- datasets                          # dataset parser
         |- samplers                          # sampling strategy in training
         |- transforms                        # training & testing data augmentation
      |- engine                            # training pipeline control
         |- trainer.py                        # for network pre-train
         |- transfer.py                       # for network transfer
         |- cutter.py                         # for network cut
         |- inference.py                      # for evaluation
      |- layers                            # define all the layers used in our DNN
      |- modeling                          # define how to build the DNN
      |- solver                            # define the optimizer
      |- structures                        # define the input data structure
      |- utils                             # some auxiliary functions
   |- tools/                            # program entry
      |- train.py                          # start network pre-train
      |- transfer.py                       # start network transfer
      |- cut.py                            # start network cut
   |- train.txt/                        # scripts for building and running the program
   |- 95_name.txt/                      # category names for the herb image dataset with 95 categories
```

# Run
## Data Prepraration
- Create a folder named "data" under the root dir: `mkdir -p data & cd data`
- Download the herb image dataset in Google Drive with the shared link: https://drive.google.com/drive/folders/10BGJYMcrSsbonilPVwkC8eBONss-TCw9?usp=sharing
- Decompress the dataset file with `tar -mxf images.tar` to obtain the folder "images"
- Go back to the root folder: `cd ..`
- Run `python3 gen_med_db/gen_db.py` to generate the dataset db file
## Network Pre-train
- `CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py --config-file ./configs/e2e_R50_1x.yaml SOLVER.IMS_PER_BATCH 32`
- `CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py --config-file ./configs/e2e_MV2_1x.yaml SOLVER.IMS_PER_BATCH 32`
## Network Transfer
- `CUDA_VISIBLE_DEVICES=0 python3 ./tools/transfer.py --config-large-file ./configs/e2e_R50_1x.yaml --config-small-file ./configs/e2e_MV2_1x_transfer.yaml SOLVER.IMS_PER_BATCH 32`
## Network Cut
- `CUDA_VISIBLE_DEVICES=0 python3 ./tools/cut.py --config-large-file ./configs/e2e_R50_1x.yaml --config-small-file ./configs/e2e_MV2_1x_cut.yaml SOLVER.IMS_PER_BATCH 32`

