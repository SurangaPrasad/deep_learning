import os
import gdown
from pathlib import Path
import requests
import tarfile
import zipfil

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
