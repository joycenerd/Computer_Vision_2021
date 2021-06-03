import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os
from pathlib import Path
from PIL import Image
import torch
from torch.autograd import Variable
from options import opt
import numpy as np
from dataset import make_dataset,Dataloader
from model.model_utils import get_net


ROOTDIR="/home/zchin/NCTU-CV_2021/HW5"

label_dict = {
    0 : 'A',
    1 : 'B',
    2 : 'C'
}

def test():
    classes = opt.num_classes
    
    model_path = opt.weight_path
    #model= torch.load(str(model_path))
    #model =  model.cuda(opt.cuda_devices)
    model=get_net(opt.model)
    model_dict=torch.load(model_path,map_location="cpu")
    model.load_state_dict(model_dict)
    model=model.cuda(opt.cuda_devices)
    model.eval()
    # print(f"Cuda num: {opt.cuda_devices}")

    test_set=make_dataset("test")
    test_loader=eval_loader = Dataloader(
        dataset=test_set, batch_size=opt.dev_batch_size, shuffle=True, num_workers=opt.num_workers)

    test_loss = 0.0
    test_corrects = 0

    criterion = nn.CrossEntropyLoss()

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = Variable(inputs.cuda(opt.cuda_devices))
        labels = Variable(labels.cuda(opt.cuda_devices))

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        test_loss += loss.item()*inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
            
    test_loss = test_loss/len(test_set)
    test_acc = float(test_corrects)/len(test_set)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

if __name__=='__main__':
    test()