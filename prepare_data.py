import numpy as np
import grib2np
import h5toNumpy
from pprint import pprint

def prepare_ECMWF_data(ECMWF_data):
    ECMWF_dict=grib2np.grib2np(ECMWF_data)
    return ECMWF_dict

def prepare_MHS_data(MHS_data):
    MHS_dict=h5toNumpy.read_hdf5_to_numpy_arrays(MHS_data)
    return MHS_dict
