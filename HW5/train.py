from dataset import make_dataset, Dataloader
from model.model_utils import get_net
from early_stop import EarlyStopping
from options import opt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch

from pathlib import Path
import copy


writer = SummaryWriter(f"runs/{opt.model}", comment=f"lr={opt.lr}")


def train():
    # train set
    train_set = make_dataset("train")
    train_loader = Dataloader(
        dataset=train_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers)

    # evaluate set
    eval_set = make_dataset("eval")
    eval_loader = Dataloader(
        dataset=eval_set, batch_size=opt.dev_batch_size, shuffle=True, num_workers=opt.num_workers)

    # select model
    net = get_net(opt.model)
    # if (opt.model[0:9] == 'resnest'):
        # model = net(opt.num_classes)
    model = net
    #model= torch.load("checkpoint/efficientnet-b3/model-50epoch-0.93-acc-all_transform.pth")
    model=model.cuda(opt.cuda_devices)
    best_model_params_acc = copy.deepcopy(model.state_dict())
    best_model_params_loss = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    best_loss = float('inf')

    # initialize loss optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True, cooldown=1)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    record = open('record.txt', 'w')

    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch+1}/{opt.epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{opt.epochs}'))

        training_loss = 0.0
        training_corrects = 0

        model.train()

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = Variable(inputs.cuda(opt.cuda_devices))
            labels = Variable(labels.cuda(opt.cuda_devices))

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / len(train_set)
        training_acc = float(training_corrects) / len(train_set)

        writer.add_scalar("train loss/epochs", training_loss, epoch+1)
        writer.add_scalar("train accuracy/epochs", training_acc, epoch+1)

        print(
            f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}')

        model.eval()

        eval_loss = 0.0
        eval_corrects = 0

        for i, (inputs, labels) in enumerate(tqdm(eval_loader)):
            inputs = Variable(inputs.cuda(opt.cuda_devices))
            labels = Variable(labels.cuda(opt.cuda_devices))

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            eval_loss += loss.item()*inputs.size(0)
            eval_corrects += torch.sum(preds == labels.data)

        eval_loss = eval_loss/len(eval_set)
        eval_acc = float(eval_corrects)/len(eval_set)

        writer.add_scalar("eval loss/epochs", eval_loss, epoch+1)
        writer.add_scalar("eval accuracy/epochs", eval_acc, epoch+1)

        print(f'Eval loss: {eval_loss:.4f}\taccuracy: {eval_acc:.4f}')
        
        lr=scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {lr}\n")

        scheduler.step(eval_loss)
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_acc_eval_loss = eval_loss

            best_train_acc = training_acc
            best_train_loss = training_loss

            best_model_params_acc = copy.deepcopy(model.state_dict())

        if eval_loss < best_loss:
            the_acc = eval_acc
            best_loss = eval_loss

            the_train_acc = training_acc
            the_train_loss = training_loss

            best_model_params_loss = copy.deepcopy(model.state_dict())

        if (epoch+1) % 10 == 0:
            model.load_state_dict(best_model_params_loss)
            weight_path = Path(opt.checkpoint_dir).joinpath(
                f'model-{epoch+1}epoch-{best_loss:.02f}-loss-{the_acc:.02f}-acc.pth')
            torch.save(model.state_dict(), str(weight_path))

            model.load_state_dict(best_model_params_acc)
            weight_path = Path(opt.checkpoint_dir).joinpath(
                f'model-{epoch+1}epoch-{best_acc:.02f}-acc.pth')
            torch.save(model.state_dict(), str(weight_path))

            record.write(f'{epoch+1}\n')
            record.write(
                f'Best training loss: {best_train_loss:.4f}\tBest training accuracy: {best_train_acc:.4f}\n')
            record.write(
                f'Best eval loss: {best_acc_eval_loss:.4f}\tBest eval accuracy: {best_acc:.4f}\n\n')


if __name__ == "__main__":
    train()
