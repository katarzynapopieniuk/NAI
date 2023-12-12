# TYTUŁ: NEURAL NETWORKS FOR CLASSIFICATION
#
# AUTORZY: Katarzyna Popieniuk s22048 i Jakub Styn s22449
#
# OPIS PROBLEMU:
# 1. Nauczyć sieć roszpoznać zwierzęta, np. z zbioru CIFAR10
#
# INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA
# 1. Zainstalować interpreter python w wersji 3+ oraz narzędzie pip
# 2. Pobrać projekt
# 3. Uruchomić wybraną konsolę/terminal
# 4. Zainstalować wymagane biblioteki za pomocą komend:
# pip install numpy
# pip install sklearn
# pip install torch
# pip install torch.nn
# pip install torch.optim
# pip install torchvision
# pip install matplotlib.pyplot
# 5. Przejść do ścieżki z projektem (w systemie linux komenda cd)
# 6. Uruchomić projekt przy pomocy polecenia:
# python .\zad2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Perform CIFAR-10 dataset training and evaluation using a neural network model.

    This script loads the CIFAR-10 dataset, trains a neural network model, and evaluates its performance
    using classification metrics and a confusion matrix on the test dataset.
    """

    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Neural Network Model Definition
    class Net(nn.Module):
        """
        Neural network model for image classification.
        """

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            """
            Perform forward pass through the neural network layers.

            Parameters:
            x (tensor): Input data tensor.

            Returns:
            x (tensor): Output tensor after passing through the network.
            """
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training the Neural Network
    for epoch in range(2):
        """
        Train the neural network model over multiple epochs.
        """
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')

    # Testing and Evaluation
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(true_labels, predicted_labels))
    print("#" * 40 + "\n")

    cm = confusion_matrix(true_labels, predicted_labels, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmd.plot(ax=ax)
    plt.show()
