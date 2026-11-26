# Import the relevant libraries
import os
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


# Function to collect the samples from the dataset directory
def collect_samples(data_directory):
    # Get all class folders
    classes = []
    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        if os.path.isdir(item_path):
            classes.append(item)

    classes.sort()

    # Create the class-to-index mapping
    class_to_index = {}
    index = 0
    for cls in classes:
        class_to_index[cls] = index
        index += 1

    # Store the (image_path, label) pairs
    samples = []
    for cls in classes:
        img_dir = os.path.join(data_directory, cls, "image")

        if not os.path.isdir(img_dir):
            print("Skipping class '" + cls + "' because no image folder was found.")
            continue

        files = os.listdir(img_dir)
        files.sort()

        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                image_path = os.path.join(img_dir, file_name)
                label = class_to_index[cls]
                samples.append((image_path, label))

    return classes, class_to_index, samples


# Function to split dataset into training, validation and test sets
def stratified_split(samples, train_ratio, val_ratio, test_ratio, seed):
    paths = []
    labels = []

    for sample in samples:
        paths.append(sample[0])
        labels.append(sample[1])

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels
    )

    return (
        list(zip(train_paths, train_labels)),
        list(zip(val_paths, val_labels)),
        list(zip(test_paths, test_labels))
    )


# EBHI Classification Dataset
class EBHIClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    

# EBHI Classification Dataset for Test-Time Augmentation
class EBHITTADataset(Dataset):
    def __init__(self, samples, image_size=224):
        self.samples = samples
        self.image_size = image_size

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img):
        img = TF.resize(img, [self.image_size, self.image_size])
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        img = TF.resize(img, [self.image_size, self.image_size])

        variants = []

        # Original
        variants.append(img)
        # Horizontal flip
        variants.append(TF.hflip(img))
        # Vertical flip
        variants.append(TF.vflip(img))
        # Horizontal + vertical flip
        variants.append(TF.vflip(TF.hflip(img)))
        # Small rotations
        variants.append(TF.rotate(img, angle=10))
        variants.append(TF.rotate(img, angle=-10))

        tensors = [self.preprocess(v) for v in variants]
        tensors = torch.stack(tensors)

        return tensors, label



def get_train_transform(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.97, 1.03)
        ),

        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.01
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



def get_eval_transform(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



def create_dataloaders(train_dataset, val_dataset, test_dataset, 
                       batch_size=16, num_workers=0, seed=42):

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader