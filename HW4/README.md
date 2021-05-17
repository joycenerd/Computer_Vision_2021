# Structure From Motion

[Zhi-Yi Chin](https://www.linkedin.com/in/zhi-yi-chin-b7927645)

## Prerequisites

* Anaconda or Miniconda
* Python3

This is an implementation of structure of motion. We first get the intrinsic matrix for all of out photos. Then we Find all the correspondences for the pair images. After that we estimate the fundamental matrix and draw the epipolar line. (To be continued...)

## Setup

```
conda env create -f environment.yml
conda activate sfm
```

## Quickstart

If you have your own pair data please name them the same and separate them by "1" and "2" at the end of the name. and put them in the "data" folder. And also if you don't have the intrinsic matrix of your own data, you should take some pictures in the checkerboard and put it in the "checkerboards" folder for camera calibration.

```
python SfM.py --img books --ratio 0.2 --iter 3000 --threshold 0.025
```

## Results

* Feature matching

<img src="./results/books_feature_matching.jpg">

* Epipolar line

<img src="./results/books_epipolar_line.jpg">

