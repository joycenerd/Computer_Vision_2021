from pathlib import Path
import argparse


ROOTPATH = "/home/zchin/NCTU-CV_2021/HW5"

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=Path(
    ROOTPATH).joinpath('hw5_data'), help='Your dataset root directory')
parser.add_argument('--model', type=str, default="resnest50",
                    help="which model: [resenest50, resnest101, resnest200, resnest269, efficientnet-b4, efficientnet-b5]")
parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
parser.add_argument('--cuda_devices', type=int, default=0, help='gpu device')
parser.add_argument('--epochs', type=int, default=100, help='num of epoch')
parser.add_argument('--num_classes', type=int, default=15,
                    help='The number of classes for your classification problem')
parser.add_argument('--train_batch_size', type=int, default=12,
                    help='The batch size for training data')
parser.add_argument('--num_workers', type=int, default=3,
                    help='The number of worker while training')
parser.add_argument('--dev_batch_size', type=int, default=1,
                    help='The batch size for test data')
parser.add_argument('--checkpoint_dir', type=str, default=Path(ROOTPATH).joinpath(
    'checkpoint/resnest200'), help='Directory to save all your checkpoint.pth')
parser.add_argument('--weight_path', type=str,
                    help='The path of checkpoint.pth to retrieve weight')
parser.add_argument('--img_size', type=int, default=224,
                    help='Input image size')
parser.add_argument('--test_root', type=str,
                    default=Path(ROOTPATH).joinpath('C1-P1_Test'), help='testset path')
opt = parser.parse_args()
