# Scene Classification

The is the project that can do 15 scenes (Coast, Forest, Highway, Industrial, InsideCity, Kitchen, LivingRoom, Mountain, Office, OpenCountry, Store, Street, Suburb, TallBuilding) classification. We utilize some methods as follow:
  1. EfficientNet which can achieve accuracy 93.78%
  2. ResNest
  3. Tiny images representation + nearest neighbor classifier
  4. Bag of SIFT + nearest neighbor classifier
  5. Bag of SIFT representation + linear SVM classifier
  6. Simple CNN

## Getting Started

### Prequisites

* Anaconda or Miniconda
* Python 3.7+
* CUDA 10.1+
* Pytorch 1.8+
* 1 GPU with 10 GB of memory

### Set-up environment

```
conda env create -f environment.yml
conda activate classification
```

### Get data labels

```
python get_data_csv.py
```
`train.csv`, `eval.csv`, `test.csv` will be in `hw5_data folder`

### Training your own model

Go to `options.py` and change the `ROOTPATH` to the path of this project in your computer

```
python train.py [-h] [--data_root DATA_ROOT] [--model MODEL] [--lr LR] [--cuda_devices CUDA_DEVICES] [--epochs EPOCHS]
                [--num_classes NUM_CLASSES] [--train_batch_size TRAIN_BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--dev_batch_size DEV_BATCH_SIZE] [--checkpoint_dir CHECKPOINT_DIR] [--weight_path WEIGHT_PATH] [--img_size IMG_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        Your dataset root directory
  --model MODEL         which model: [resenest50, resnest101, resnest200, efficientnet-b4, efficientnet-b3]
  --lr LR               learning rate
  --cuda_devices CUDA_DEVICES
                        gpu device
  --epochs EPOCHS       num of epoch
  --num_classes NUM_CLASSES
                        The number of classes for your classification problem
  --train_batch_size TRAIN_BATCH_SIZE
                        The batch size for training data
  --num_workers NUM_WORKERS
                        The number of worker while training
  --dev_batch_size DEV_BATCH_SIZE
                        The batch size for test data
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save all your checkpoint.pth
  --weight_path WEIGHT_PATH
                        The path of checkpoint.pth to retrieve weight
  --img_size IMG_SIZE   Input image size
```

**Train**
```
python train.py --model efficientnet-b4 --cuda_devices 0 --img_size 320 --train_batch_size 15
```

**Test**
```
python test.py --model efficientnet-b4 --cuda_devices 1 --weight_path checkpoint/efficientnet-b4/0.9378_model-50epoch-0.94-acc.pth --img_size 320
```

### Other method

* Tiny images representations: `./Tiny_KNN.ipynb`
* Bag of SIFT: `./HW5_BOS`
* Simple CNNN: `./CNN`


