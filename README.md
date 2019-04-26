# LiDAR road object segmentation
Segmentation and classification of lidar point cloud data based on [SqueezeSeg](https://arxiv.org/abs/1710.07368) work. Implemented with TensorFlow 1.13 and trained using [KITTI](https://www.cvlibs.net/datasets/kitti/) dataset.

## Setup
First of all, clone the repository and install requirements

```bash
git clone https://github.com/filipcima/lidar-object-recognition.git
cd lidar-object-recognition
pip install -r requirements.txt
```

Download KITTI point cloud data, calib files and put them in `dataset/training/velodyne` respectively `dataset/training/calib`. 

Then create directories `dataset/lidar_2d/training` and `dataset/lidar_2d/testing` where preprocessed data will be held. To create folders execute following command. 
```bash
mkdir -p dataset/lidar_2d/train \
         dataset/lidar_2d/test \
         dataset/image_set
```

Now we are ready to go!

## Preprocessing data
To use spherical projection on raw point clouds from KITTI, execute script `preprocess.py`. All data will be converted and saved in `dataset/lidar_2d`.

To split data to have 3 parts of testing data and 1 part of evaluating data, run script `split_data.py`. Data will be splitted into two folders - `dataset/lidar_2d/train` and `dataset/lidar_2d/test`.

It will also create files `all.txt`, `train.txt` and `eval.txt` in `dataset/image_set` folder, whose may be handy for distributed training.  

## Training
It's highly recommended to train this model on GPU instead of CPU. If you want to train from scratch, delete `src/model` folder.

Start training with batch size of 4 and 32 epochs with following command (training steps is a constant number in `train.py`): 

```bash
# assuming you are in src/ folder
python train.py --verbose -b 4 -e 32 -m train
```

To see some output during training, launch TensorBoard with your log dir as parameter. By default, you'll see __loss function progression, mean IoU (only in eval mode) and some examples of segmentation__.

To run TensorBoard, execute following command:
```bash
# assuming you are in src/ folder and training process was started at least once
tensorboard --logdir model --port 7777
```

Now you can visit `http://localhost:7777` and check tracked tensors, parameters and evaluated images.


## Evaluating
To evaluate trained model, use script `train.py`. Usage is following:
```bash
python train.py --verbose -m eval
``` 

Script will start evaluation of testing data. Once it's done, mean IoU will be available in terminal or in TensorBoard Dashboard. 

## Predicting
To segment specific point cloud in format [64 x 512 x 5] without ground truth labels, use `predict.py` script.

To predict frame `002323` from `test`ing data, execute script as proposed here.
```bash
python predict.py -m test 002323 # default mode is test if -m is ommited
```

## Experimental features
- Remove unwanted horizontal lines and smooth the surface in projected images using mathematical morphology - `src/experimental/morphology.ipynb`
- Remove ground points using DBSCAN in point cloud to reduce unwanted horizontal lines and reduce computations - (cancelled due to high computation costs) `src/experimental/morphology.ipynb` 