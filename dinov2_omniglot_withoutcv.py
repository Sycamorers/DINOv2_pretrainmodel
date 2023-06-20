import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import Omniglot, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLPClassifier(nn.Module):
    """
    Define a MLP classifier class
    
    The input dimension has to match with specific model
    """
    def __init__(self, num_classes, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(

            #-------------------------------vitg14----------------------------------#
            # nn.Linear(1536, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #-------------------Single Layer------------------#
            # nn.Linear(1536, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #-------------------------------------------------#
            #-------------------------------vitg14----------------------------------#
            
            #-------------------------------vitl14----------------------------------#
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #-------------------Single Layer------------------#
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            #-------------------------------------------------#
            #------------------More Layer---------------------#
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(dropout),

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #-------------------------------------------------#
            #-------------------------------vitl14----------------------------------#

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class DINOClassifier:
    def __init__(self, dinov2_version='vitg14', root="./omniglot", batch_size=64, test_split=0.111, epochs=1200, lr=0.001, weight_decay=0.01):
        """
        Define a DINO Classifier class
        Initialization of classifier with various parameters like root directory, batch size, 
        Split ratio for test data, epochs, learning rate etc.

        Arg:
            dinov2_version
            root
            batch_size
            test_split
            epochs
            lr
            weight_decay
        """
        self.root = root
        self.batch_size = batch_size
        self.test_split = test_split
        self.epochs = epochs
        self.lr = lr
        # Pick the appropriate DINO V2 model
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.dinov2 = self.pick_dinov2_version(dinov2_version)
            # Use multiple GPUs
            if torch.cuda.device_count() > 1:
                self.dinov2 = nn.DataParallel(self.dinov2)
            self.dinov2 = self.dinov2.to(self.device)
        else:
            self.device = 'cpu'
        
        # Define transformations for the dataset    
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Resize((98, 98), antialias=True)
        ])

        self.dataloader = None
        self.model = None
        self.weight_decay = weight_decay
        # Transform for training data to fit the model
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((98, 98)),
            transforms.ToTensor()
        ])
        # Transform for test data to fit the model
        self.test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((98, 98), antialias=True)
        ])

    def pick_dinov2_version(self, version):
        """
        Function to pick a DINO v2 model version
        
        Args:
            version: pretrained DINO-v2 version
        """
        if version == 'vits14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        elif version == 'vitl14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
        elif version == 'vitls4':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        else:
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').cuda()

    def load_data(self, dataset):
        """
        Function to download and load the Omniglot dataset with the defined transformations,
        and then split the data into train, test, and validation datasets

        Args:
            dataset: root directory
        """
        # Check dataset
        if dataset == "./omniglot":
            full_dataset = Omniglot(
                root=self.root, download=True, transform=self.train_transform
            )
        else:
            full_dataset = CIFAR10(
                root=self.root, download=True, transform=self.train_transform
            )

        # Define the sizes of training, testing, and validation data
        train_size = int(0.8 * len(full_dataset))
        test_size  = int(0.1 * len(full_dataset))
        val_size   = len(full_dataset) - train_size - test_size

        # Split the dataset into train, test, and validation datasets
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size, val_size]
        )

        train_dataset.dataset.transform = self.train_transform
        test_dataset.dataset.transform = self.test_transform
        val_dataset.dataset.transform = self.test_transform

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size
        )

        self.test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=self.batch_size
        )

        self.val_dataloader = DataLoader(
            val_dataset, shuffle=False, batch_size=self.batch_size
        )

    def extract_embeddings(self):
        """
        Function to extract embeddings from the DINO v2 model for training and testing
        """
        all_embeddings, all_targets = [], []
        with torch.no_grad():
            for dataloader in [self.train_dataloader, self.val_dataloader]:
                for images, targets in tqdm(dataloader):
                    images = images.to(self.device)
                    embedding = self.dinov2(images)
                    all_embeddings.append(embedding)
                    all_targets.append(targets)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return all_embeddings.cpu().numpy(), all_targets.cpu().numpy()


    def train(self, X_train, y_train, X_test, y_test):
        """
        Function to train the classifier

        Args:
            X_train: embeddings from DINO-v2 for training
            y_train: true target for training
            X_test: embeddings from DINO-v2 for testing
            y_test: true target for testing
            
        """
        #----------------------------------Multilayer Perceptron-----------------------------------------#
        # # Count the number of unique classes in your labels
        num_classes = len(torch.unique(torch.from_numpy(y_train)))
        # Initialize the MLPClassifier model
        self.model = MLPClassifier(num_classes)
        # Use multiple GPUs
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        #-----------------------------------------------------------------------------------------------#
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Prepare the data
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).long().to(self.device)
        X_test = torch.tensor(X_test).float().to(self.device)
        y_test = torch.tensor(y_test).long().to(self.device)

        self.model.train()
        epoch_acc = []
        test_acc = []

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
            # Evaluate on test data and store the accuracy
            self.model.eval()
            test_outputs = self.model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_correct = (test_predicted == y_test).sum().item()
            test_acc.append(test_correct / y_test.size(0))
            self.model.train()
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}, Train Accuracy: {acc}, Test Accuracy: {test_acc[-1]}')

        return epoch_acc, test_acc

    
    def evaluate(self, X_test, y_test):
        # Function to evaluate the model performance
        self.model.eval()
        X_test = torch.tensor(X_test).float().to(self.device)
        y_test = torch.tensor(y_test).long().to(self.device)
        outputs = self.model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()

        return correct / y_test.size(0)


dataset = "./omniglot" # Specify which dataset to use (omniglot or cifar10)
dinov2_version = 'vitl14' # Specify which dino-v2 model to use (vitls4, vitlb4, vitl14, vitg14)
# Instantiate the DINO classifier
classifier = DINOClassifier(dinov2_version=dinov2_version, root=dataset, weight_decay=0.01)
# Load data and extract embeddings
classifier.load_data(dataset)
# Get embeddings from DINO-v2
X, y = classifier.extract_embeddings()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=classifier.test_split, random_state=42)
# Train the classifier and get the accuracy of each epoch
epoch_acc, test_acc = classifier.train(X_train, y_train, X_test, y_test)
print(f'Final test accuracy: {test_acc[-1]}')

# Plot the training and testing accuracies over epochs
plt.figure(figsize=(10, 5))
plt.plot(epoch_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.legend()
plt.show()