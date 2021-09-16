from __future__ import print_function
import argparse
import os


from model import *
from train import *
from test import *

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        return argparse.ArgumentTypeError('Boolean value expected')
def parse_args():
    parser = argparse.ArgumentParser(description='CCBDA GAN')
    ## Switch what to do
    parser.add_argument('-test', type=str2bool, nargs='?', default=True)
    parser.add_argument('-train_more', type=str2bool, nargs='?', default=False)
    
    parser.add_argument('-savepath', type=str, default='/home/anita/v_CNN/model/model.pt',help='where to save the model')
    parser.add_argument('-trainpath', type=str, default='/home/anita/v_CNN/hw5_data/train',help='where is the root directory of data')
    parser.add_argument('-testpath', type=str, default='/home/anita/v_CNN/hw5_data/test',help='where is the root directory of data')
    parser.add_argument('-resultpath', type=str, default='/home/anita/v_CNN/result',help='where to save the result')
    parser.add_argument('-loadpath', type=str, default='/home/anita/v_CNN/model/model.pt',help='where to load the model')
    
    parser.add_argument("-workers",default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument("-image_size",default=256, type=int, help='Spatial size of training images. All images will be resized to this size using a transformer')
    parser.add_argument("-img_resize",default=256, type=int, help='image size for the results')
    parser.add_argument("-batch_size",default=64, type=int, help='Batch size during training')
    parser.add_argument("-num_epochs",default=50, type=int, help='Number of training epochs')

    parser.add_argument("-lr",default=0.0002, type=float, help='Learning rate for optimizers')
    parser.add_argument("-beta1",default=0.5, type=int, help='Beta1 hyperparam for Adam optimizers')
    parser.add_argument("-ngpu",default=1, type=int, help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument("-seed",default=999, type=int, help='Set random seed for reproducibility')
    
    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()
    print(args)
    # Give random seed for reproduce
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.test == True:
        test(args)
    else:
        training_loss, training_accuracy, validation_loss, validation_accuracy = train(args)
        # visualization
        plt.plot(range(args.num_epochs), training_loss, 'b-', label='Training_loss')
        plt.plot(range(args.num_epochs), validation_loss, 'g-', label='validation_loss')
        plt.title('Training & Validation loss')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.plot(range(args.num_epochs), training_accuracy, 'b-', label='Training_accuracy')
        plt.plot(range(args.num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
        plt.title('Training & Validation accuracy')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    
