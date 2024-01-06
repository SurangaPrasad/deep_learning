import os
import gdown
from pathlib import Path
import requests
import tarfile
import zipfile

import torch
import torchvision
from torch import nn
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The selected device is {device}")

def download_and_extract(data_path,image_path, url, downlod_file_name, zip_file: bool=True):
  # Setup path to data folder

  # If the image folder doesn't exist, download it and prepare it...
  if image_path.is_dir():
      print(f"{image_path} directory exists.")
  else:
      print(f"Did not find {image_path} directory, creating one...")
      image_path.mkdir(parents=True, exist_ok=True)

      if zip_file:
        # Download zip file
        with open(data_path / downlod_file_name, "wb") as f:
            request = requests.get(url, verify=False)
            print("Downloading zip file...")
            f.write(request.content)

        # Unzip zip file
        with zipfile.ZipFile(data_path / downlod_file_name, "r") as zip_ref:
            print("Unzipping zip file...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        os.remove(data_path / downlod_file_name)

      else:
        # Download the tar file
        with open(data_path / downlod_file_name, "wb") as f:
            request = requests.get(url)
            print("Downloading tar file ...")
            f.write(request.content)

        # Unzip mini tar file
        with tarfile.open(data_path / downlod_file_name, "r") as tar:
            print("Unzipping tar file...")
            tar.extractall(image_path)

        # Remove .tar file
        os.remove(data_path / downlod_file_name)

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_data,
    test_data,
    val_data,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  # Turn images into data loaders
  train_dataloader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = torch.utils.data.DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )
  valid_dataloader = torch.utils.data.DataLoader(
      val_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, valid_dataloader


import matplotlib.pyplot as plt

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """


    epochs = range(len(results["train_loss"]))

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []


    for i in epochs:
      train_loss.append(results["train_loss"][i].detach().cpu().numpy())
      test_loss.append(results["val_loss"][i].detach().cpu().numpy())
      train_accuracy.append(results["train_acc"][i].detach().cpu().numpy())
      test_accuracy.append(results["val_acc"][i].detach().cpu().numpy())

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

from PIL import Image
import random


def getEuroSatDataLoaders(starting_number, eurosat_dataset):

  sample_list = eurosat_dataset.imgs
  unique_categories = set(label for _, label in sample_list)
  selected_categories = random.sample(unique_categories, k=5)
  category_counters = {label: 0 for label in selected_categories}

  # Initialize lists to store selected images and labels
  selected_images = []
  selected_labels = []


  # Iterate through the sample list to select 20 images from each of the 5 categories
  for image_path, label in sample_list:
      if label in selected_categories and category_counters[label] < starting_number*20 + 20:
          if category_counters[label] > starting_number*20: ### select another set of images and labels according to the starting_number
            selected_images.append(image_path)
            selected_labels.append(label)
          category_counters[label] += 1
  # Randomly choose 25 images for training (5 from each category)
  training_images = []
  training_labels = []
  testing_images = []
  testing_labels = []


  category_counters = {label: 0 for label in selected_categories} ## reset all category counters

  # print(f"Selected Categories {selected_categories} \n category_counters {category_counters}")

  for image_path, label in sample_list:
      if label in selected_categories and category_counters[label] < 20:
          if category_counters[label] < 5:
              # Add to training set
              training_images.append(image_path)
              training_labels.append(label)
          else:
              # Add to testing set
              testing_images.append(image_path)
              testing_labels.append(label)
          category_counters[label] += 1

  # Define transforms (you can customize these)
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
  ])
  # Define a custom dataset class
  class CustomDataset(Dataset):
      def __init__(self, images, labels, transform=None):
          self.images = images
          self.labels = labels
          self.transform = transform

      def __len__(self):
          return len(self.images)

      def __getitem__(self, idx):
          image_path, label = self.images[idx], self.labels[idx]
          image = Image.open(image_path).convert("RGB")

          if self.transform:
              image = self.transform(image)

          return image, label

  # Create datasets and data loaders for training and testing
  train_dataset = CustomDataset(training_images, training_labels, transform=transform)
  test_dataset = CustomDataset(testing_images, testing_labels, transform=transform)

  # Define batch size
  batch_size = 32

  # Create DataLoader instances
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader



