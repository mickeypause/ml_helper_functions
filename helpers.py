import torch
import platform


def accuracy_fn(y_true, y_pred):
    # torch.eq() calculates where two tensors are equal
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
 
def setup_device():
    operating_system = platform.system()

    if operating_system == "Darwin":
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    else: 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return device 



def download_data(url: str, data_path: str, image_path: str, file_name: str ):
    """
    Function witch downloads data from link

    Args: 
        url: direct url to file 
        data_path: dir in which should data be placed
        image_path: complete path to image 
        file_name: name of download file 
    """

    import os
    import zipfile
    from pathlib import Path
    import requests

    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        zip_path = data_path /file_name
        
        with open(zip_path, "wb") as f:
            request = requests.get(url)
            print("Downloading data...")
            f.write(request.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            print("Unzipping data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        os.remove(zip_path)
    
    print('Done.')
