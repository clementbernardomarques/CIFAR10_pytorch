import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import itertools

import torchvision
import torchvision.transforms as transforms

import os

from models import cnn
from utils import progress_bar


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # A progress bar will appear in terminal
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Training: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        train_loss / len(trainloader), 100. * correct / total, correct, total))

    # Tensorboard monitoring
    writer.add_scalar('Loss/train', train_loss / len(trainloader), epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)


def test(epoch, name_model):
    global best_acc

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for label, prediction in zip(targets, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Test: Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / len(testloader),
                                                      100. * correct / total, correct, total))
    # Tensorboard monitoring
    writer.add_scalar('Loss/test', test_loss / len(testloader), epoch)
    writer.add_scalar('Accuracy/test', 100. * correct / total, epoch)

    # Tensorboard monitoring  for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        writer.add_scalar('Accuracy/test/' + classname, accuracy, epoch)

    acc = 100. * correct / total
    if acc > best_acc:
        # Saving checkpoints
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/' + name_model):
            os.mkdir('checkpoint/' + name_model)
        torch.save(state, './checkpoint/' + name_model + '/ckpt.pth')
        best_acc = acc


if __name__ == '__main__':
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    device = 'cuda' if train_on_gpu else 'cpu'

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # Data
    print('==> Preparing data..')

    # Data Augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Just normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # test dataset will remain the same for every experience
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    # classes of the CIFAR10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Value to test
    lr_to_test = [0.1, 0.01, 0.001]
    batch_size_to_test = [32, 64, 128]
    n_epoch = 30

    for lr, batch_size in list(itertools.product(*[lr_to_test, batch_size_to_test])):
        best_acc = 0  # best test accuracy

        # Model
        print('==> Building model..')
        model = cnn.Net()

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            model.cuda()

        # Loading train dataset with batch_size value
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        # specify loss function
        criterion = nn.CrossEntropyLoss()
        # specify optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # name of the model in order to understand each experience made
        name_model = "lr_" + str(lr) + "_batch_" + str(batch_size)

        # Initialization of Tensorboard
        writer = SummaryWriter("my_experiment/" + name_model)

        for epoch in range(0, n_epoch):
            train(epoch)
            test(epoch, name_model)
