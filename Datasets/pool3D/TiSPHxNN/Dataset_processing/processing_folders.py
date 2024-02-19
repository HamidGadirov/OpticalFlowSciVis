import os
import re
import shutil
from typing import List, Dict, Tuple

def analyze_filenames_in_folder(folder_path: str, file_type: str) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[int, int]]]:
    """
    Analyzes filenames in the specified folder, extracting attribute names and number ranges.

    :param folder_path: Path to the folder containing the files.
    :return: A tuple containing two dictionaries:
             - The first dictionary has attributes as keys and lists of numbers as values.
             - The second dictionary has attributes as keys and tuples representing the number ranges as values.
    """
    # Get filenames from the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Extract 'xxx' part and numbers
    attribute_dict = {}
    for filename in filenames:
        match = re.match(rf"([a-z,A-Z,0-9,_]+)_([0-9]+)\.{file_type}", filename)
        if match:
            attribute, number = match.groups()
            number = int(number)
            if attribute in attribute_dict:
                attribute_dict[attribute].append(number)
            else:
                attribute_dict[attribute] = [number]

    # Determine the range for each attribute
    range_dict = {attr: (min(numbers), max(numbers)) for attr, numbers in attribute_dict.items()}
    # Print the name of this function and the result
    print(f"{analyze_filenames_in_folder.__name__}({folder_path}): attributes: {attribute_dict.keys()}, ranges: {range_dict}")
    return attribute_dict, range_dict

def clear_folder(folder_path):
    """
    Clears all files and subdirectories in the specified folder.

    Parameters:
    folder_path (str): The path to the folder to clear.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def concatDataset(input_folder_path, folder_list, attr_name_list, output_folder_path):
    """
    Concatenates datasets from multiple folders, renaming the files to maintain a continuous numbering
    across folders for each attribute specified in attr_name_list.

    This function first counts the number of files for each attribute in each folder, using these counts 
    to determine the starting index for file renaming in each subsequent folder. It ensures that file 
    numbering is continuous and consistent across all folders.

    Args:
    input_folder_path (str): The base directory containing the folders with the datasets.
    folder_list (list of str): A list of folder names, in the order in which they should be processed.
    attr_name_list (list of str): A list of attribute names to be used for filtering and renaming files.
    output_folder_path (str): The directory where the concatenated and renamed files will be saved.

    The function copies files from each folder in folder_list, renames them based on their attribute name
    and a continuous numbering scheme, and then saves these files in output_folder_path.

    Note:
    - Assumes file names are in the format '{attribute_name}_{number}.npy'.
    - The directories specified in input_folder_path and output_folder_path should exist.
    - The function does not handle cases where the output folder already contains files with the same names
      as the renamed files, which may lead to overwriting.
    """
    total_number_of_frames = 0
    file_counters = {folder: {attr_name: 0 for attr_name in attr_name_list} for folder in folder_list}

    for folder_idx in range(len(folder_list)):
        for attr_idx in range(len(attr_name_list)):
            folder_name = folder_list[folder_idx]
            attr_name = attr_name_list[attr_idx]
            folder_path = os.path.join(input_folder_path, folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.npy'):
                    parts = file_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[0] == attr_name:
                        file_counters[folder_name][attr_name] += 1

    total_number_of_frames = int(sum(file_counters[i][attr_name_list[0]] for i in folder_list))
    print("concatDataset(): Files have been concatenated and copied to:", output_folder_path)

    for folder_idx in range(len(folder_list)-1):
        folder_name = folder_list[folder_idx]
        next_folder_name = folder_list[folder_idx+1]
        for attr_idx in range(len(attr_name_list)):
            attr_name = attr_name_list[attr_idx]
            file_counters[next_folder_name][attr_name] += file_counters[folder_name][attr_name]
    
    print("concatDataset(): File counters:", file_counters)

    
    for folder_idx in range(len(folder_list)-1, 0, -1):
        folder_name = folder_list[folder_idx]
        previoud_folder_name = folder_list[folder_idx-1]
        for attr_idx in range(len(attr_name_list)):
            attr_name = attr_name_list[attr_idx]
            file_counters[folder_name][attr_name] = file_counters[previoud_folder_name][attr_name]
    first_folder_name = folder_list[0]
    for attr_idx in range(len(attr_name_list)):
        attr_name = attr_name_list[attr_idx]
        file_counters[first_folder_name][attr_name] = 0


    print("concatDataset(): File counters:", file_counters)

    for folder_name in folder_list:
        folder_path = os.path.join(input_folder_path, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.npy'):
                parts = file_name.rsplit('_', 1)
                if len(parts) == 2 and parts[0] in attr_name_list:
                    attribute_name = parts[0]
                    old_file_number = int(parts[1].split('.')[0])
                    new_file_number = old_file_number + file_counters[folder_name][attribute_name]
                    new_file_name = f'{attribute_name}_{new_file_number}.npy'
                    new_file_path = os.path.join(output_folder_path, new_file_name)
                    shutil.copy(file_path, new_file_path)

    
    print("concatDataset(): Total number of frames:", total_number_of_frames)
    return total_number_of_frames