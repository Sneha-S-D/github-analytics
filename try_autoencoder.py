import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )
         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the autoencoder
model = Autoencoder()

# Define transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Verify directory contents
def check_directory_contents(directory):
    for root, dirs, files in os.walk(directory):
        print(f"Checking directory: {root}")
        for name in files:
            print(f"File: {name}")
        for name in dirs:
            print(f"Subdirectory: {name}")

train_dir = r'E:\train'
test_dir = r'E:\test'

check_directory_contents(train_dir)
check_directory_contents(test_dir)

# Custom dataset class to handle nested subdirectories
class NestedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, loader=default_loader, is_valid_file=None):
        super().__init__(root, transform=transform, loader=loader, is_valid_file=is_valid_file)

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        if is_valid_file is None:
            is_valid_file = self._is_valid_file
        for target_class in sorted(class_to_idx.keys()):
            class_dir = os.path.join(directory, target_class)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target_class])
                        images.append(item)
        return images

    def _is_valid_file(self, path):
        try:
            return os.path.isfile(path)
        except:
            return False

# Load datasets
train_dataset = NestedImageFolder(root=train_dir, transform=transform)
test_dataset = NestedImageFolder(root=test_dir, transform=transform)

# Define the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'conv_autoencoder.pth')

# Function to save reconstructed images
def save_images(images, output_path):
    images = images.view(-1, 3, 64, 64)
    os.makedirs(output_path, exist_ok=True)
    for idx, img in enumerate(images):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f'reconstructed_{idx}.png'))
        plt.close()

# Reconstruct and save images
output_dir = r'E:\reconstruced-auto-images'
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)
output = model(images)
save_images(output.detach(), output_dir)

print(f'Reconstructed images saved in: {output_dir}')
