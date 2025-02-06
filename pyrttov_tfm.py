# Example of using the Rttov class to call RTTOV for multiple instruments
# with the emissivity and BRDF atlases

# Three Rttov instances are created representing three instruments

import pyrttov
import example_data as ex
import netCDF4toNumpy
import grib2np
import prepare_data
import numpy as np
import os
import sys
from pprint import pprint
import create_dataset

rttov_installdir = '/home/ecca/Desktop/RTTOV/'

mhs_data_path='/home/ecca/Downloads/data/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+MHS_C_EUMP_20240521021851_60570_eps_o_l1.nc'
ecmwf_data_path='/home/ecca/Downloads/data/ECMWF/ECMWF_20240521_00+02.grib'

mhs_data, ecmwf_data=prepare_data.prepare_all_data(ecmwf_data_path, mhs_data_path)

if __name__ == '__main__':

    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------

    # Declare an instance of Profiles
    nlevels = 54
    nprofiles = 10
    # len(ecmwf_data['t_filtered'])
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    # valiable to store the results
    Result = []
    def expand2nprofiles(n, nprof):
        """Transform 1D array to a [nprof, nlevels] array"""
        outp = np.empty((nprof, len(n)), dtype=n.dtype)
        for i in range(nprof):
            outp[i, :] = n[:]
        return outp

    
    ## PRINTS PARA CHEQUEAR
    #print('MHS lat shape', mhs_data['lat'].shape)
    #print('ecmwf_data[\'lat\'] shape', ecmwf_data['lat'].shape)
    #print(ecmwf_data['lat'])
    #print(ecmwf_data['lon'])
    #print(mhs_data['lat'])

    # TODO: preguntar que es la Z y si se puede usar el terrain_elevation de MHS en vez de la Z de ECMWF
    #print('Z: ', ecmwf_data['z'])
    #print('Terrain elevation: ', mhs_data['terrain_elevation'])

    # ECMWF data
    myProfiles.P = ecmwf_data['p_filtered'][:nprofiles, :nlevels]
    myProfiles.T = ecmwf_data['t_filtered'][:nprofiles,:nlevels]
    myProfiles.Q = ecmwf_data['q_filtered'][:nprofiles, :nlevels]
   
    # MHS data
    myProfiles.Angles = mhs_data['angles'][:nprofiles]
    myProfiles.SurfType = mhs_data['surftype_filtered'][:nprofiles, :nlevels]
    myProfiles.SurfGeom = mhs_data['surfgeom'][:nprofiles, :nlevels]
    myProfiles.Skin = ecmwf_data['skin_filtered'][:nprofiles, :nlevels]

    # TODO: check S2m en calcRad    
    myProfiles.S2m = expand2nprofiles(ex.s2m[0], nprofiles)

    # Others
    myProfiles.GasUnits = ex.gas_units
    myProfiles.CO2 = expand2nprofiles(ex.co2_ex[:nlevels], nprofiles)
    myProfiles.DateTimes = mhs_data['datetimes'][:nprofiles]

    # --------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------
    # Set up Rttov instances for each instrument
    # ------------------------------------------------------------------------

    # Create Rttov objects for MHS instrument
    mhsRttov = pyrttov.Rttov()

    # For MHS we will read all channels
    nchan_mhs = 5
    
    ## fichero de coeficientes rtcoef_metop_2_mhs.dat
    mhsRttov.FileCoef = '{}/{}'.format(rttov_installdir,
                                       "rtcoef_rttov13/rttov13pred54L/rtcoef_metop_2_mhs.dat")
    mhsRttov.Options.AddInterp = True
    mhsRttov.Options.StoreTrans = True
    mhsRttov.Options.VerboseWrapper = True

    # Load the instruments: for HIRS and MHS do not supply a channel list and
    # so read all channels
    try:
        mhsRttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        sys.exit(1)

    # Associate the profile with Rttov instance
    mhsRttov.Profiles = myProfiles

    # ------------------------------------------------------------------------
    # Load the emissivity and BRDF atlases
    # ------------------------------------------------------------------------

    # Load the emissivity and BRDF atlases:
    # - load data for the month in the profile data
    # - load the IR emissivity atlas data for multiple instruments so it can be used for SEVIRI and HIRS
    # - SEVIRI is the only VIS/NIR instrument we can use the single-instrument initialisation for the BRDF atlas

    # TELSEM2 atlas does not require an Rttov object to initialise
    mwAtlas = pyrttov.Atlas()
    mwAtlas.AtlasPath = '{}/{}'.format(rttov_installdir, "emis_data")
    mwAtlas.loadMwEmisAtlas(ex.datetimes[0][1])

    # Set up the surface emissivity/reflectance arrays and associate with the Rttov object
    surfemisrefl_mhs = np.zeros((5,nprofiles,nchan_mhs), dtype=np.float64)

    mhsRttov.SurfEmisRefl = surfemisrefl_mhs

    # ------------------------------------------------------------------------
    # Call RTTOV
    # ------------------------------------------------------------------------

    # Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
    # Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
    # calcemis/calcrefl TRUE - see RTTOV user guide)

    surfemisrefl_mhs[:,:,:]    = -1.

    # Call emissivity and BRDF atlases
    try:
        surfemisrefl_mhs[0,:,:] = mwAtlas.getEmisBrdf(mhsRttov)

    except pyrttov.RttovError as e:
        sys.stderr.write("Error calling atlas: {!s}".format(e))

    # Call the RTTOV direct model for MHS instrument:
    # no arguments are supplied to runDirect so all loaded channels are
    # simulated
    try:
        mhsRttov.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Print out some of the output
    # ------------------------------------------------------------------------

    print
    print("SELECTED OUTPUT")
    print

    # We can access the RTTOV transmission structure because the store_trans
    # option was set above for mhsRttov

    # El resultado se da en 5 canales por cada perfil. 

    print("MHS total transmittance")
    for p in range(nprofiles):
        print("Profile {:d}:".format(p))
        channels = []
        for c in range(nchan_mhs):
            print("  Ch #{:02d} tau={:f}".format(c + 1,
                                                 mhsRttov.TauTotal[p, c]))
            channels.append(mhsRttov.TauTotal[p][c])
        Result.append(channels)    
        print

    create_dataset.create_dataset(myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.Angles, myProfiles.SurfType, myProfiles.SurfGeom, 
                                  myProfiles.Skin, myProfiles.S2m, myProfiles.CO2, myProfiles.DateTimes, np.array(Result))


    # ------------------------------------------------------------------------
    # Save the information in a csv file for latter use
    # ------------------------------------------------------------------------
"""
    print('P shape ', myProfiles.P.shape)
    print('P ', myProfiles.P)
    print('T shape ', myProfiles.T.shape)
    print('T ', myProfiles.T)
    print('Q shape ', myProfiles.Q.shape)
    print('Q ', myProfiles.Q)
    print('Angles shape ', myProfiles.Angles.shape)
    print('Angles ', myProfiles.Angles)
    print('SurfType shape ', myProfiles.SurfType.shape)
    print('SurfType ', myProfiles.SurfType)
    print('SurfGeom shape ', myProfiles.SurfGeom.shape)
    print('SurfGeom ', myProfiles.SurfGeom)
    print('Skin shape ', myProfiles.Skin.shape)
    print('Skin ', myProfiles.Skin)
    print('S2m shape ', myProfiles.S2m.shape)
    print('S2m ', myProfiles.S2m)
    print('CO2 shape ', myProfiles.CO2.shape)
    print('CO2 ', myProfiles.CO2)
    print('DateTimes shape ', myProfiles.DateTimes.shape)
    print('DateTimes ', myProfiles.DateTimes)
    print('Result ', Result)
"""
    
    # ------------------------------------------------------------------------
    # Deallocate memory
    # ------------------------------------------------------------------------

    # Because of Python's garbage collector, there should be no need to
    # explicitly deallocate memory
