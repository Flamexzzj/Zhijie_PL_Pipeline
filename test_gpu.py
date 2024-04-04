import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Print CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")

        # Print properties of each CUDA device
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            print(f"Device {i} Properties: {device_properties}")
    else:
        print("CUDA is not available. Please ensure that PyTorch is installed with CUDA support and that your CUDA drivers are up to date.")

def test_num_workers(num_workers):
    try:
        # Transformations for the dataset
        transform = transforms.Compose([transforms.ToTensor()])

        # Using CIFAR10 as an example dataset
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # DataLoader with specified num_workers
        data_loader = DataLoader(dataset, batch_size=64, num_workers=num_workers)

        # Start time
        start_time = time.time()

        # Iterate through the dataset
        for batch_idx, (data, targets) in enumerate(data_loader):
            pass

        # End time
        end_time = time.time()

        # Calculate and print time taken
        print(f"Time taken with num_workers={num_workers}: {end_time - start_time} seconds")

    except Exception as e:
        print(f"An error occurred with num_workers={num_workers}: {e}")

if __name__ == "__main__":
    check_cuda()
    print("GPU check done!")
    print("*"*50)
    print("Testing with 0 to 4 workers")
    print("*"*50)
    for num in range(0, 5):  # Testing with 0 to 4 workers
        test_num_workers(num)
