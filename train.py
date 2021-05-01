from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from util.models import fcNet, ConvNet, Net

import os

'''
From: https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580/2
'''
class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform_fn = transform

    def __getitem__(self, index):
        if self.transform_fn is None:
            return self.dataset[index]
        x = self.transform_fn(self.dataset[index][0])
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def eval(model, device, data_loader, dataset_name):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num
    accuracy = float(correct) / test_num

    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        dataset_name, test_loss, correct, test_num, 100. * accuracy))
    return test_loss, accuracy


def plot_loss_acc(losses):
    epochs = losses.shape[0]
    x_space = list(range(1, epochs+1))

    loss_fig = plt.figure()
    plt.plot(x_space, losses[:, 0, 0], label="Training Loss")
    plt.plot(x_space, losses[:, 1, 0], label="Validaton Loss")

    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss over {} Epochs".format(epochs))

    acc_fig = plt.figure()
    plt.plot(x_space, losses[:, 0, 1], label="Training Accuracy")
    plt.plot(x_space, losses[:, 1, 1], label="Validaton Accuracy")

    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Accuracy over {} Epochs".format(epochs))

    return loss_fig, acc_fig


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Assign a seed so that samples are the same between runs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_dir = 'data/'
    out_dir = 'out/'
    os.makedirs(out_dir, exist_ok=True)

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST(data_dir, train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        eval(model, device, test_loader, "Test Set")

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    validation_ratio = 0.15
    subset_ratio = (1./8)          # Subset of the training set to keep (p7)
    num_classes = len(train_dataset.classes)
    subset_indices_train = []
    subset_indices_valid = []

    # Store the indices of all samples of each class
    class_indices = [[] for _ in range(num_classes)]
    for i in range(len(train_dataset.targets)):
        class_indices[train_dataset.targets[i]].append(i)

    # Do train/validation split for each class
    for indices in class_indices:
        # Shuffle and take subset
        np.random.shuffle(indices)
        subset_size = int(len(indices) * subset_ratio)
        indices = indices[:subset_size]

        # Since we seed, we shuffle again to get different training sets
        np.random.shuffle(indices) 

        val_split_index = int(len(indices) * validation_ratio)
        subset_indices_valid.extend(indices[:val_split_index])
        subset_indices_train.extend(indices[val_split_index:])

    np.random.shuffle(subset_indices_train)
    np.random.shuffle(subset_indices_valid)
    train_subset = data.Subset(train_dataset, subset_indices_train)
    valid_subset = data.Subset(train_dataset, subset_indices_valid)

    # Define augmentation on training split ONLY
    # NOTE: Using 'fillcolor' for backward-compatability. Using
    #       'fill' for recent versions of torchvision is preferred
    train_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=30, 
                translate=(0.11, 0.11),     # 3px shift
                scale=(0.5, 1.2),
                shear=(3,3,3,3), 
                fillcolor=0
        )], p=0.85),
    ])

    # Create loader for either train or val set
    train_loader_tf = torch.utils.data.DataLoader(
        MapDataset(train_subset, train_transforms), 
        batch_size=args.batch_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=args.batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        valid_subset, 
        batch_size=args.test_batch_size,
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=0.003, betas=[0.9, 0.999])

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    losses = np.zeros((args.epochs, 2, 2))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader_tf, optimizer, epoch)

        # Evaluate on both training and validation sets
        losses[epoch-1, 0] = eval(model, device, train_loader, "Train Set")
        losses[epoch-1, 1] = eval(model, device, val_loader, "Validation Set")

        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    # Save model/figures
    if args.save_model:
        torch.save(model.state_dict(), "out/mnist_model.pt")

    # Plot loss and accuracy
    loss_fig, acc_fig = plot_loss_acc(losses)
    loss_fig.savefig(os.path.join(out_dir,'training_loss.png'))
    acc_fig.savefig(os.path.join(out_dir,'training_accuracy.png'))
    plt.show()


if __name__ == '__main__':
    main()
