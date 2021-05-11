from __future__ import print_function
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import logging
import pandas as pd
from sklearn.metrics import f1_score

act_deg = 3

print = logging.info
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.k = nn.parameter.Parameter(torch.rand(act_deg+1))

    def activate(self, x):
        act_val = self.k[act_deg]
        for i in range(act_deg,0,-1):
            act_val = act_val*x + self.k[i-1]
        return act_val

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = self.activate(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_loss += loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_loss /= len(train_loader.dataset)
            train_acc = 100. * correct / len(train_loader.dataset)
    return train_loss, train_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    f1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            f1 += f1_score(target.cpu(), pred.cpu(), average='weighted')
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    f1 /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc, f1


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-file', type=str, default="outputs/test.log",
                        help='log file')
    parser.add_argument('--checkpoint-file', type=str, default="outputs/mnist_cnn.pt",
                        help='checkpoint file')
    args = parser.parse_args()
    torch.manual_seed(1)
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(args.log_file, 'w', 'utf-8') 
    handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(handler)

    logging.getLogger('matplotlib.font_manager').disabled = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    if os.path.exists(args.checkpoint_file):
        model.load_state_dict(torch.load(args.checkpoint_file))
        test(model, device, test_loader)
        return
        
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    params = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc, _ = test(model, device, test_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        params.append(model.k.detach().cpu().numpy())
        scheduler.step()
    torch.save(model.state_dict(), args.checkpoint_file)

    test_loss, test_acc, f1 = test(model, device, test_loader)
   
    fig = plt.figure()
    fig.tight_layout()
    plt.subplot(2, 2, 1)
    plt.xlabel("Epochs")
    plt.plot(range(args.epochs), train_losses, label='Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.xlabel("Epochs")
    plt.plot(range(args.epochs), train_accs, label='Training Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.xlabel("Epochs")
    plt.plot(range(args.epochs), test_losses, label='Testing Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.xlabel("Epochs")
    plt.plot(range(args.epochs), test_accs, label='Testing Accuracy')
    plt.legend()

    plt.savefig("outputs/history_plot.png")

    prms = []
    for prm in zip(*params):
        prms.append([p for p in prm])
    
    columns = ['Training Loss', 'Training Accuracy', 'Testing Loss', 'Testing Accuracy'] + ["k{}".format(str(i)) for i in range(act_deg + 1)]
    df = pd.DataFrame(list(zip(train_losses, train_accs, test_losses, test_accs, *prms)), columns=columns)
    df.to_csv("outputs/details.csv")
    

    side_of_plots = (act_deg + 1)//2
    fig, ax = plt.subplots(nrows=side_of_plots, ncols=side_of_plots)
    fig.tight_layout()
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.set_xlabel("Epochs")
            col.plot(range(args.epochs), prms[2*i+j], label="k"+str(2*i+j))
            col.legend()

    plt.savefig("outputs/parameter_plot.png")
    print("Final F1 score for the test dataset: {}".format(f1))


if __name__ == '__main__':
    main()
