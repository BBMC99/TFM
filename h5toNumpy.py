import h5py
import numpy as np
from pprint import pprint

def read_hdf5_to_numpy_arrays(file_path):
    numpy_arrays = {}
    
    # Abrir el archivo en modo lectura
    with h5py.File(file_path, 'r') as hdf5_file:
        # Leer los datasets en el archivo
        def read_group(group, path=''):
            for key in group.keys():
                item = group[key]
                full_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Dataset):
                    numpy_arrays[full_path] = item[:]
                elif isinstance(item, h5py.Group):
                    read_group(item, full_path)

        read_group(hdf5_file)
    
    return numpy_arrays

def read_hdf5_to_dict(file_path):

    data_dict = {}

    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as hdf5_file:
        # Recursively read datasets in the file
        def read_group(group, path=''):
            for key in group.keys():
                item = group[key]
                full_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Dataset):
                    # Add dataset to the dictionary
                    data_dict[full_path] = item[:]
                elif isinstance(item, h5py.Group):
                    # Recursively read sub-groups
                    read_group(item, full_path)

        read_group(hdf5_file)

    return data_dict

# Example usage
if __name__ == "__main__":
    hdf5_file_path = "/home/bruno/Downloads/MHSL1_20240521T021851Z_20240521T040051Z_epct_fa2babab_F.h5"  # Replace with the path to your HDF5 file
    data = read_hdf5_to_dict(hdf5_file_path)

    # Print dataset names and their corresponding shapes
    for name, array in data.items():
        print(f"Dataset: {name}, Shape: {array.shape}, Dtype: {array.dtype}")
    example_key = next(iter(data))  # Get the first key (dataset name)
    print(f"\n--- Example Dataset Access ---")
    print(f"Name: {example_key}")
    print(f"Data (First 5 entries): {data[example_key].flat[:5]}")



