from model import *
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pdb
def test(args):
    ###-------- Load Data --------###
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(args.testpath,
        transform=transforms.Compose([
            transforms.CenterCrop(args.image_size),
            transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    # Create the dataloader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers)
    
    ###-------- Check --------###
    # Decide which device we want to run on
    device = torch.device("cuda:1" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    
    ###-------- Model Initialize --------###
    # Create the classifier
    model = CNN_Model().to(device)
    
    # Setup Adam optimizers for both G and D
    optimizer = optim.Adam(model.parameters(), lr=args.lr)   # optimize all cnn parameters
    input_shape = (-1,3,args.img_resize,args.img_resize)

    
    
    
    ###-------- Load Model --------###
    checkpoint = torch.load(args.savepath)
    model.load_state_dict(checkpoint['CNN_Model'])
    model.eval().to(device)

    ###-------- Testing -----------###  
    correct_test = 0
    total_test = 0
    for images, labels in test_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape)).to(device)
            # 2.Forward propagation
            outputs = model(test)
            # 3.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1].detach().cpu()
            # 4.Total number of labels
            total_test += len(labels)
            # 5.Total correct predictions
            correct_test += (predicted == labels).float().sum()
    #6.store test_acc
    test_accuracy = 100 * correct_test / float(total_test)
    print('Test_accuracy: {:.6f}%'.format(test_accuracy))
    
    