import os
import shutil

def copy_files(source_dir, goal_dir, file_extensions):
    """
    Aggregates files with specified extensions from source_dir (including subfolders)
    and moves them to goal_dir. It does not copy files that already exist in goal_dir.

    :param source_dir: Directory to search for files.
    :param goal_dir: Directory to move the files to.
    :param file_extensions: List of file extensions to look for.
    """
    
    os.makedirs(goal_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(tuple(file_extensions)):
                source_file_path = os.path.join(root, file)
                goal_file_path = os.path.join(goal_dir, file)

                if not os.path.exists(goal_file_path):
                    shutil.copy2(source_file_path, goal_file_path)

# Usage example
source_directory = "/home/dongjun/data/physionet_dsail/"
goal_directory = "/home/dongjun/data/physionet_aggr/"
extensions = [".hea", ".mat"]

copy_files(source_directory, goal_directory, extensions)
print("Data Aggregation Complete!")