import netCDF4
import numpy as np
from pprint import pprint

def read_netcdf_to_dict(file_path):
    data_dict = {}

    # Open the NetCDF file in read mode
    with netCDF4.Dataset(file_path, 'r') as nc_file:
        def read_group(group, path=''):
            for key in group.variables.keys():
                full_path = f"{path}/{key}" if path else key
                print(f"Reading variable: {full_path}")
                data_dict[full_path] = group.variables[key][:]  # Read variable as a NumPy array
            for sub_group_name in group.groups.keys():
                sub_group = group.groups[sub_group_name]
                sub_path = f"{path}/{sub_group_name}" if path else sub_group_name
                read_group(sub_group, sub_path)

        # Start reading from the root group
        read_group(nc_file)

    return data_dict