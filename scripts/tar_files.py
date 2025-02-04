import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

def tar_single_directory(subdir_path, output_tar_name):
    with tarfile.open(output_tar_name, "w:gz") as tar:
        tar.add(subdir_path, arcname=os.path.basename(subdir_path))
    print(f"Created {output_tar_name}")

def tar_directories_parallel(parent_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    with ThreadPoolExecutor() as executor:
        for subdir in subdirs:
            output_tar_name = os.path.join(output_dir, os.path.basename(subdir) + ".tar.gz")
            executor.submit(tar_single_directory, subdir, output_tar_name)

# Example usage
parent_directory = "/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_Bi/1bar"
output_directory = "/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_Bitars"
tar_directories_parallel(parent_directory, output_directory)
