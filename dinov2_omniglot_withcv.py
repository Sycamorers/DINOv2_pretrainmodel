# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define a Logistic Regression Model class
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# Define a DINO Classifier class
class DINOClassifier:
    def __init__(self, dinov2_version='vitl14', root="./omniglot", batch_size=64, n_splits=5, epochs=100, lr=0.001):
        # Initialization of classifier with various parameters like root directory, batch size, 
        # split ratio for test data, epochs, learning rate etc.
        self.root = root
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.epochs = epochs
        self.lr = lr
        # Check and set the device to CUDA if available
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.dinov2 = self.pick_dinov2_version(dinov2_version)
            if torch.cuda.device_count() > 1:
                # Use multiple GPUs
                # Pick the appropriate DINO V2 model
                self.dinov2 = nn.DataParallel(self.dinov2)
            self.dinov2 = self.dinov2.to(self.device)
        else:
            self.device = 'cpu'
        # Define transformations for the Omniglot dataset
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Resize((98, 98), antialias=True)
        ])
        self.dataloader = None
        self.dinov2 = self.pick_dinov2_version(dinov2_version)
        self.dinov2 = self.dinov2.to(self.device)
        self.model = None

    def pick_dinov2_version(self, version):
        # Function to pick a DINO v2 model version
        if version == 'vits14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        elif version == 'vitl14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
        else:
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        
    def load_data(self):
        # Function to download and load the Omniglot dataset with the defined transformations, 
        # and then split the data into train and test datasets
        dataset = Omniglot(
            root=self.root, download=True, transform=self.transform
        )
        self.dataloader = DataLoader(
            dataset, shuffle=True, batch_size=self.batch_size
        )

    def extract_embeddings(self):
        # Function to extract embeddings from the DINO v2 model using the Omniglot dataset images
        all_embeddings, all_targets = [], []
        with torch.no_grad():
            for images, targets in tqdm(self.dataloader):
                images = images.to(self.device)
                embedding = self.dinov2(images)
                all_embeddings.append(embedding)
                all_targets.append(targets)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return all_embeddings.cpu().numpy(), all_targets.cpu().numpy()


    def cross_val_data(self, X, y):
        kfold = KFold(n_splits=self.n_splits, shuffle=True)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
            yield X[train_ids], y[train_ids], X[test_ids], y[test_ids]

    def train(self, X_train, y_train):
        self.model = LogisticRegressionModel(X_train.shape[1], len(torch.unique(torch.from_numpy(y_train))))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).long().to(self.device)

        self.model.train()
        epoch_acc = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            outputs = self.model(X_train)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_train).sum().item()
            acc = correct / y_train.size(0)

            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            epoch_acc.append(acc)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}, Accuracy: {acc}')

        return epoch_acc
    

    def evaluate(self, X_test, y_test):
        self.model.eval()

        X_test = torch.tensor(X_test).float().to(self.device)
        y_test = torch.tensor(y_test).long().to(self.device)

        outputs = self.model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()

        return correct / y_test.size(0)




classifier = DINOClassifier()
classifier.load_data()

X, y = classifier.extract_embeddings()

for i, (X_train, X_test, y_train, y_test) in enumerate(classifier.cross_val_data(X, y)):
    print(f'Cross Validation {i+1}/{classifier.n_splits}')
    epoch_acc = classifier.train(X_train, y_train)
    test_acc = classifier.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_acc, label='Train Accuracy')
    plt.legend()
    plt.show()