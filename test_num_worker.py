import os
import requests
import zipfile
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

def download_tiny_imagenet(destination):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    if not os.path.exists(destination):
        os.makedirs(destination)

    zip_path = os.path.join(destination, 'tiny-imagenet-200.zip')
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        os.remove(zip_path)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet already downloaded.")

def test_num_workers(batch_size, num_workers, data_path):
    try:
        # Transformations for the dataset (adjust as needed)
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor()
        ])

        # Loading Tiny ImageNet
        dataset = datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200', 'train'), transform=transform)

        # DataLoader with specified num_workers and batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        # Start time
        start_time = time.time()

        # Iterate through the dataset
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx == 10:  # Limiting to 10 batches for time efficiency
                break

        # End time
        end_time = time.time()

        # Calculate and print time taken
        print(f"Batch Size: {batch_size}, num_workers: {num_workers}, Time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred with batch size {batch_size}, num_workers={num_workers}: {e}")

if __name__ == "__main__":
    data_path = './data'
    download_tiny_imagenet(data_path)
    for batch_size in [32, 64, 128]:  # Test with different batch sizes
        for num in range(0, 8):  # Testing with 0 to 7 workers
            test_num_workers(batch_size, num, data_path)
