import os
import tarfile
import requests
from tqdm import tqdm
import shutil

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def prepare_lfw_dataset():
    """Download and prepare the LFW dataset."""
    # Create directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'datasets')
    lfw_dir = os.path.join(datasets_dir, 'lfw')
    download_dir = os.path.join(datasets_dir, 'downloads')
    
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    # Download LFW dataset
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    lfw_file = os.path.join(download_dir, "lfw.tgz")
    
    print("Downloading LFW dataset...")
    if not os.path.exists(lfw_file):
        download_file(lfw_url, lfw_file)
    
    # Extract dataset
    print("\nExtracting LFW dataset...")
    if os.path.exists(lfw_dir):
        shutil.rmtree(lfw_dir)
    
    with tarfile.open(lfw_file, 'r:gz') as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            
            tar.extractall(path, members, numeric_owner=numeric_owner)
            
        safe_extract(tar, datasets_dir)
    
    # Reorganize dataset to keep only classes with sufficient samples
    min_samples = 10  # Minimum number of samples per class
    processed_dir = os.path.join(datasets_dir, 'lfw_processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    print("\nProcessing dataset...")
    class_counts = {}
    
    # Count samples per class
    for person in os.listdir(lfw_dir):
        person_dir = os.path.join(lfw_dir, person)
        if os.path.isdir(person_dir):
            n_samples = len(os.listdir(person_dir))
            if n_samples >= min_samples:
                class_counts[person] = n_samples
    
    # Copy selected classes
    with tqdm(total=len(class_counts), desc="Copying selected classes") as pbar:
        for person in class_counts:
            src_dir = os.path.join(lfw_dir, person)
            dst_dir = os.path.join(processed_dir, person)
            shutil.copytree(src_dir, dst_dir)
            pbar.update(1)
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"Total number of classes: {len(class_counts)}")
    print(f"Total number of images: {sum(class_counts.values())}")
    print(f"Average images per class: {sum(class_counts.values()) / len(class_counts):.2f}")
    
    # Update example script path
    example_script = os.path.join(base_dir, 'examples', 'train_siamese.py')
    if os.path.exists(example_script):
        with open(example_script, 'r') as f:
            content = f.read()
        
        content = content.replace(
            'data_dir = "path/to/your/dataset"',
            'data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "lfw_processed")'
        )
        
        with open(example_script, 'w') as f:
            f.write(content)
    
    print("\nDataset preparation completed!")
    print(f"Processed dataset is available at: {processed_dir}")

if __name__ == "__main__":
    prepare_lfw_dataset()
