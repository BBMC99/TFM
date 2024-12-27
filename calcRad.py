import pyrttov
import datetime
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import sonde
import distBrng
import radiance
import pressLevels
import humidity
from multiprocessing import Pool
import sys

# Example of using the Rttov class to call RTTOV for MHS only
# Simplified to calculate a single profile and avoid second derivatives

stateType = 'ECMWF'
NLEVELS = 101  # European models use 101 levels

rttov_installdir = '/home/xcalbet/RTTOV13/'

rcnw = 1.60771704E6  # kg/kg -> ppmv water vapour
rcnv = 6.03504E5     # kg/kg -> ppmv ozone
rcnco2 = 6.58183E5   # kg/kg -> ppmv co2
Md = 28.966          # Molecular mass of dry air
Mw = 18.016          # Molecular mass of water

# Simplified function to process a single profile
def proc_rttov():
    """Processes a single profile for MHS."""
    # Single profile setup
    nprofiles = 1
    nlevels = NLEVELS
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    # Example values for one profile (replace with actual data inputs)
    pressure = np.logspace(3, 2, nlevels)  # Logarithmic pressure levels
    temperature = np.linspace(300, 200, nlevels)  # Linearly decreasing temperature
    q = np.full(nlevels, 0.002)  # Example specific humidity

    myProfiles.P = np.expand_dims(pressure, axis=0)
    myProfiles.T = np.expand_dims(temperature, axis=0)
    myProfiles.Q = np.expand_dims(q, axis=0)

    # Example setup for surface parameters
    myProfiles.Skin = np.array([[290.0, 0.0, 0.0, 0.0, 2.3, 1.9, 21.8, 0.0, 0.5]])
    myProfiles.SurfGeom = np.array([[45.0, 10.0, 0.0]])
    myProfiles.Angles = np.array([[53.0, 0.0, 40.0, 0.0]])
    myProfiles.DateTimes = np.array([[2023, 1, 1, 0, 0, 0]])

    # Setting up MHS RTTOV
    mhsRttov = pyrttov.Rttov()
    mhsRttov.FileCoef = f"{rttov_installdir}/rtcoef_rttov13/rttov13pred101L/rtcoef_metop_2_mhs.dat"
    mhsRttov.Options.AddInterp = True
    mhsRttov.Options.StoreTrans = True
    mhsRttov.Options.VerboseWrapper = True

    try:
        mhsRttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write(f"Error loading MHS instrument: {e}\n")
        return

    # Assign profiles
    mhsRttov.Profiles = myProfiles

    # Run RTTOV
    try:
        mhsRttov.runDirect()
        print("MHS radiances:", mhsRttov.Rads)
    except pyrttov.RttovError as e:
        sys.stderr.write(f"Error running RTTOV: {e}\n")

    del mhsRttov

if __name__ == "__main__":
    proc_rttov()
