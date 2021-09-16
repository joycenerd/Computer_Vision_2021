from model import *
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pdb

def train(args):
	###-------- Load Data --------###
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(args.trainpath,
    	transform=transforms.Compose([
    		transforms.CenterCrop(args.image_size),
    		transforms.Resize(args.img_resize),
            transforms.ToTensor(),
    		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    		]))
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
    	shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers)
    
    ###-------- Check --------###
    # Decide which device we want to run on
    device = torch.device("cuda:1" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    
    ###-------- Model Initialize --------###
    # Create the classifier
    model = CNN_Model().to(device)
    
    # Setup Adam optimizers for both G and D
    optimizer = optim.SGD(model.parameters(), lr=args.lr)   # optimize all cnn parameters
    input_shape = (-1,3,args.img_resize,args.img_resize)

    
    if args.train_more == True:
        checkpoint = torch.load(args.loadpath)
        model.load_state_dict(checkpoint['CNN_Model'])
        

    # Print the model
    print(model)
    
    # Initialize CrossEntropyLoss function
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    #criterion = nn.BCELoss()
    
    
    
    ###-------- Training --------### 
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(args.num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1.Define variables
            train = Variable(images.view(input_shape)).to(device)
            labels = Variable(labels).to(device)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)
            
        total_val = 0 
        correct_val = 0
        for images, labels in val_loader:
            # 1.Define variables
            val = Variable(images.view(input_shape)).to(device)
            labels = Variable(labels).to(device)
            # 2.Forward propagation
            outputs = model(val)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_val += len(labels)
            # 6.Total correct predictions
            correct_val += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_val / float(total_val)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        ###-------- Save Model --------### 
        torch.save({'CNN_Model':model.state_dict()},args.savepath)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, args.num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy