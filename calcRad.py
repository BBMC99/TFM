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

# Example of using the Rttov class to call RTTOV for multiple instruments
# with the emissivity and BRDF atlases
# export PYTHONPATH=/home/xcalbet/RTTOV13/wrapper
# Three Rttov instances are created representing three instruments

#stateType='ECMWF'
stateType='Extrapolate'

NLEVELS=50
#NLEVELS=90

rttov_installdir = '/home/xcalbet/RTTOV13/'

nVars=3

rcnw = 1.60771704E6       # kg/kg--> ppmv water vapour
rcnv = 6.03504E5          # kg/kg--> ppmv ozone
rcnco2 = 6.58183E5          # kg/kg--> ppmv co2
Md=28.966                 # Molecular mass of dry air
Mw=18.016                 # Molecular mass of water

def expand2nprofiles(n, nprof):
    # Transform 1D array to a [nprof, nlevels] array
    outp = np.empty((nprof, len(n)), dtype=n.dtype)
    for i in range(nprof):
        outp[i, :] = n[:]
    return outp

def proc_rttov(idx):

    sidx="%06d" % idx
    #print("idx=",idx)
    if (idx+nbufGran) >= nrecs:
        nbuf=nrecs-idx
    else:
        nbuf=nbufGran

    ## We test if this record has been written already or not
    #test_bt_iasi=rFile.variables['BT_IASI'][idx:idx+nbuf,:]
    #print("test_bt_iasi=",test_bt_iasi[0,0])
    #if np.isfinite(test_bt_iasi[0,0]):
    #    print("Record number ",idx," seems to be already processed. Skipping.")
    #    return

    #nprofiles = nVars*ndTh*ndTh*nlevels*nlevels*nbuf
    nprofiles = nVars*ndTh*ndTh*(nlevels*nlevels//2+nlevels//2)*nbuf
    #nprofiles = nVars*ndTh*nlevels*nbuf
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)

    #newDims=(nVars,ndTh,nlevels,nbuf,nlevels)
    newDims3D=(nVars,ndTh,ndTh,(nlevels*nlevels//2+nlevels//2),nbuf,nlevels)
    newDimsSurf=(nVars,ndTh,ndTh,(nlevels*nlevels//2+nlevels//2),nbuf)
    
    # Set up the surface emissivity/reflectance arrays and associate with the Rttov objects
    surfemisrefl_iasi = np.zeros((4,nprofiles,nchan_iasi), dtype=np.float64)
    surfemisrefl_mhs = np.zeros((4,nprofiles,nchan_mhs), dtype=np.float64)
    iasiRttov.SurfEmisRefl = surfemisrefl_iasi
    mhsRttov.SurfEmisRefl = surfemisrefl_mhs


    #print("idx, idx+nbuf ",idx,idx+nbuf)
    #print("nbuf=",nbuf)
    #print("nbufGran=",nbufGran)
    #print("skt=",skt[idx:idx+nbuf])
    #print("idx lon lat IASI point ",idx,nbuf, lon[idx],lat[idx],"------------------------------->")
    


    # This example program simulates two profiles for each of three instruments
    # The example profile data are defined in example_data

    # ------------------------------------------------------------------------
    # Set up the profile data
    # ------------------------------------------------------------------------

    # Associate the profiles and other data from example_data.h with myProfiles
    # Note that the simplecloud, clwscheme, icecloud and zeeman data are not mandatory and
    # are omitted here

    myProfiles.GasUnits = 1
    pArr=np.tile(p,(nprofiles,1))
    #print("pArr=",pArr.shape)
    #myProfiles.P = p[np.newaxis,:]
    myProfiles.P = pArr
    #print("P=",myProfiles.P.shape)

    tB=np.copy(np.broadcast_to(t[idx:idx+nbuf,:],(nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nlevels)))
    #print("tB=",tB.shape)
    qB=np.copy(np.broadcast_to(q[idx:idx+nbuf,:],(nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nlevels)))
    #print("qB=",qB.shape)

    # Calculations with T and e
    if nVars > 1:
        count=0
        for ilev1 in range(nlevels):
            for ilev2 in range(ilev1+1):
                for idt1 in range(dT.size):
                    for idt2 in range(dT.size):
                        vmr1=qB[2,idt1,idt2,count,:,ilev1]/(1.0-qB[2,idt1,idt2,count,:,ilev1]+Md/Mw*qB[2,idt1,idt2,count,:,ilev1])*Md/Mw
                        vmr1=vmr1*(1.0+dh[idt1])
                        qB[2,idt1,idt2,count,:,ilev1]=Mw*vmr1/Md/(1.0+Mw*vmr1/Md-vmr1)
                        tB[2,idt1,idt2,count,:,ilev2]+=dT[idt2]
                count+=1


    # Calculations with T
    if nVars > 1:
        count=0
        for ilev1 in range(nlevels):
            for ilev2 in range(ilev1+1):
                for idt1 in range(dT.size):
                    for idt2 in range(dT.size):
                        tB[1,idt1,idt2,count,:,ilev1]+=dT[idt1]
                        tB[1,idt1,idt2,count,:,ilev2]+=dT[idt2]
                count+=1

    # Calculations with e or vmr
    # vmr as volume mixing ratio versus DRY air
    count=0
    for ilev1 in range(nlevels):
        for ilev2 in range(ilev1+1):
            for idt1 in range(dT.size):
                for idt2 in range(dT.size):
                    # Mixing ratio respect to Dry air
                    #vmr1=qB[0,idt1,idt2,count,:,ilev1]/(1.0-qB[0,idt1,idt2,count,:,ilev1])*Md/Mw
                    #vmr1=vmr1*(1.0+dh[idt1])
                    #qB[0,idt1,idt2,count,:,ilev1]=Mw*vmr1/Md/(1.0+Mw*vmr1/Md)
                    # Mixing ratio respect to Wet air
                    vmr1=qB[0,idt1,idt2,count,:,ilev1]/(1.0-qB[0,idt1,idt2,count,:,ilev1]+Md/Mw*qB[0,idt1,idt2,count,:,ilev1])*Md/Mw
                    vmr1=vmr1*(1.0+dh[idt1])
                    qB[0,idt1,idt2,count,:,ilev1]=Mw*vmr1/Md/(1.0+Mw*vmr1/Md-vmr1)
                    # Mixing ratio respect to Dry air
                    #vmr2=qB[0,idt1,idt2,count,:,ilev2]/(1.0-qB[0,idt1,idt2,count,:,ilev2])*Md/Mw
                    #vmr2=vmr2*(1.0+dh[idt2])
                    #qB[0,idt1,idt2,count,:,ilev2]=Mw*vmr2/Md/(1.0+Mw*vmr2/Md)
                    # Mixing ratio respect to Wet air
                    vmr2=qB[0,idt1,idt2,count,:,ilev2]/(1.0-qB[0,idt1,idt2,count,:,ilev2]+Md/Mw*qB[0,idt1,idt2,count,:,ilev2])*Md/Mw
                    vmr2=vmr2*(1.0+dh[idt2])
                    qB[0,idt1,idt2,count,:,ilev2]=Mw*vmr2/Md/(1.0+Mw*vmr2/Md-vmr2)
            count+=1

    myProfiles.T = tB.reshape((nprofiles,nlevels))
    #print("myProfiles.T ",myProfiles.T.shape)
    del tB

    myProfiles.Q = qB.reshape((nprofiles,nlevels))
    #print("myProfiles.Q ",myProfiles.Q.shape)
    #del RB
    del qB
    #del pB
    
    #myProfiles.CO2 = expand2nprofiles(ex.co2_ex, nprofiles)
    #print(myProfiles.CO2)
    o3B=np.copy(np.broadcast_to(o3[idx:idx+nbuf,:],newDims3D))
    #o3B=np.copy(np.broadcast_to(o3[idx:idx+nbuf,:],(nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,o3[idx:idx+nbuf,:].shape[1])))
    #myProfiles.O3 = o3[idx:idx+nbuf,:]
    myProfiles.O3 = o3B.reshape((nprofiles,o3[idx:idx+nbuf,:].shape[1]))
    #print("myProfiles.O3 ",myProfiles.O3.shape)
    del o3B
    
    co2B=np.copy(np.broadcast_to(co2[idx:idx+nbuf,:],newDims3D))
    myProfiles.CO2 = co2B.reshape((nprofiles,co2[idx:idx+nbuf,:].shape[1]))
    #print("myProfiles.CO2 ",myProfiles.CO2.shape)
    del co2B
   
    #angles=np.array([[satzen[idx],satazi[idx],sunzen[idx],sunazi[idx]]])
    satzenB=np.copy(np.broadcast_to(satzen[idx:idx+nbuf],newDimsSurf))
    satzenB=satzenB.reshape((nprofiles))
    sataziB=np.copy(np.broadcast_to(satazi[idx:idx+nbuf],newDimsSurf))
    sataziB=sataziB.reshape((nprofiles))
    sunzenB=np.copy(np.broadcast_to(sunzen[idx:idx+nbuf],newDimsSurf))
    sunzenB=sunzenB.reshape((nprofiles))
    sunaziB=np.copy(np.broadcast_to(sunazi[idx:idx+nbuf],newDimsSurf))
    sunaziB=sunaziB.reshape((nprofiles))
    angles=np.vstack((satzenB,sataziB,sunzenB,sunaziB)).transpose()
    myProfiles.Angles = angles
    #print("angles=",angles.shape)

    #s2m=np.array([[sp[idx],sat[idx],sq[idx],0.,0.,0.]])
    #cero=np.repeat(0.,nbuf)
    cero=np.repeat(0.,nprofiles)
    #print("cero=",cero.shape)
    # Elevation of sodankyla
    #elevation=cero+0.1793
    # Elevation of lindenberg
    elevation=cero+0.1038

    #print("sp=",sp[idx:idx+nbuf].shape)
    #print("sat=",np.squeeze(sat[idx:idx+nbuf]).shape)
    
    spB=np.copy(np.broadcast_to(sp[idx:idx+nbuf],newDimsSurf))
    spB=spB.reshape((nprofiles))
    satB=np.copy(np.broadcast_to(np.squeeze(sat[idx:idx+nbuf]),newDimsSurf))
    satB=satB.reshape((nprofiles))
    sqB=np.copy(np.broadcast_to(np.squeeze(sq[idx:idx+nbuf]),newDimsSurf))
    sqB=sqB.reshape((nprofiles))
    s2m=np.vstack((spB,satB,sqB,cero,cero,cero)).transpose()
    #print("s2m=",s2m.shape)
    myProfiles.S2m = s2m

    sktB=np.copy(np.broadcast_to(skt[idx:idx+nbuf],newDimsSurf))
    sktB=sktB.reshape((nprofiles))
    # No salinity, no snow , bare soil fastem
    #skin=np.array([[skt[idx],0.,0.,0.,2.3, 1.9, 21.8, 0.0, 0.5]])
    #skin=np.vstack((skt[idx:idx+nbuf],cero,cero,cero,np.repeat(2.3,nbuf), np.repeat(1.9,nbuf), np.repeat(21.8,nbuf), cero, np.repeat(0.5,nbuf))).transpose()
    skin=np.vstack((sktB,cero,cero,cero,np.repeat(2.3,nprofiles), np.repeat(1.9,nprofiles), np.repeat(21.8,nprofiles), cero, np.repeat(0.5,nprofiles))).transpose()
    myProfiles.Skin = skin

    # Land
    #surftype=np.array([[0,0]])
    surftype=np.vstack((cero,cero)).transpose()
    myProfiles.SurfType = surftype
    # Elevation of Dinajpur
    #surfgeom=np.array([[lat[idx],lon[idx],0.042]])
    # We set elevation to zero
    latB=np.copy(np.broadcast_to(lat[idx:idx+nbuf],newDimsSurf))
    latB=latB.reshape((nprofiles))
    lonB=np.copy(np.broadcast_to(lon[idx:idx+nbuf],newDimsSurf))
    lonB=lonB.reshape((nprofiles))
    #surfgeom=np.vstack((lat[idx:idx+nbuf],lon[idx:idx+nbuf],cero)).transpose()
    surfgeom=np.vstack((latB,lonB,elevation)).transpose()
    myProfiles.SurfGeom = surfgeom
    #datetimes=np.array([[year[idx],month[idx],day[idx],hour[idx],minu[idx],secs[idx]]])

    yearB=np.copy(np.broadcast_to(year[idx:idx+nbuf],newDimsSurf))
    yearB=yearB.reshape((nprofiles))
    monthB=np.copy(np.broadcast_to(month[idx:idx+nbuf],newDimsSurf))
    monthB=monthB.reshape((nprofiles))
    dayB=np.copy(np.broadcast_to(day[idx:idx+nbuf],newDimsSurf))
    dayB=dayB.reshape((nprofiles))
    hourB=np.copy(np.broadcast_to(hour[idx:idx+nbuf],newDimsSurf))
    hourB=hourB.reshape((nprofiles))
    minuB=np.copy(np.broadcast_to(minu[idx:idx+nbuf],newDimsSurf))
    minuB=minuB.reshape((nprofiles))
    secsB=np.copy(np.broadcast_to(secs[idx:idx+nbuf],newDimsSurf))
    secsB=secsB.reshape((nprofiles))
    datetimes=np.vstack((yearB,monthB,dayB,hourB,minuB,secsB)).transpose()
    myProfiles.DateTimes = datetimes


    # Associate the profiles with each Rttov instance
    mhsRttov.Profiles = myProfiles
    iasiRttov.Profiles = myProfiles

    if monthOld != month[idx]:
        irAtlas.loadIrEmisAtlas(month[idx], ang_corr=True) # Include angular correction, but do not initialise for single-instrument

        mwAtlas.loadMwEmisAtlas(month[idx])

    # ------------------------------------------------------------------------
    # Call RTTOV
    # ------------------------------------------------------------------------

    # Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
    # Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
    # calcemis/calcrefl TRUE - see RTTOV user guide)

    surfemisrefl_mhs[:,:,:]    = -1.
    surfemisrefl_iasi[:,:,:]    = -1.

    # Call emissivity and BRDF atlases
    try:
        # Do not supply a channel list for SEVIRI: this returns emissivity/BRDF values for all
        # *loaded* channels which is what is required
        print("loading emis ")
        #surfemisrefl_iasi[0,:,:] = irAtlas.getEmisBrdf(iasiRttov)
        surfemisrefl_iasi[0,:,:] = emsPine
        surfemisrefl_mhs[0,:,:] = mwAtlas.getEmisBrdf(mhsRttov)
        print("end loading emis")
        
    except pyrttov.RttovError as e:
        # If there was an error the emissivities/BRDFs will not have been modified so it
        # is OK to continue and call RTTOV with calcemis/calcrefl set to TRUE everywhere
        sys.stderr.write("Error calling atlas: {!s}".format(e))

    # Call the RTTOV direct model for each instrument:
    # no arguments are supplied to runDirect so all loaded channels are
    # simulated
    try:
        mhsRttov.runDirect()
        #iasiRttov.runDirect()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # Print out some of the output
    # ------------------------------------------------------------------------
    
    #print
    #print("SELECTED OUTPUT")
    #print
    
    #print("IASI radiances")
    #for p in range(nprofiles):
    #    print("Profile {:d}:".format(p))
    #    for c in range(nchan_iasi):
    #        print("  Ch #{:02d} rad={:f}".format(c + 1, iasiRttov.Rads[p, c]))
    #    print
    
    OutIASI=iasiRttov.Rads
    OutMHS=mhsRttov.Rads

    # For testing
    #OutIASI=np.zeros((nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nchanIASI.size))
    #OutMHS=np.zeros((nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nchanMHS.size))
    if OutIASI is not None:
        print("OutIASI=",OutIASI)
        print("OutIASI=",OutIASI.shape)
    #print("OutMHS=",OutMHS.shape)
    if OutIASI is not None:
        OutIASI=OutIASI.reshape((nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nchanIASI.size))
    OutMHS=OutMHS.reshape((nVars,ndTh,ndTh,nlevels**2//2+nlevels//2,nbuf,nchanMHS.size))
    #OutIASI=OutIASI.reshape((nVars,ndTh,nlevels,nbuf,nchanIASI.size))
    #OutMHS=OutMHS.reshape((nVars,ndTh,nlevels,nbuf,nchanMHS.size))

    if OutIASI is not None:
        print("OutIASI=",OutIASI.shape)
    #print("OutMHS=",OutMHS.shape)
    #print("OutMHS 45=",OutMHS[0,0,45,0,0])
    #print("OutMHS 45=",OutMHS[0,1,45,0,0])
    #print("OutMHS 45=",OutMHS[0,2,45,0,0])
    #print("OutMHS 46=",OutMHS[0,0,46,0,0])
    #print("OutMHS 46=",OutMHS[0,1,46,0,0])
    #print("OutMHS 46=",OutMHS[0,2,46,0,0])

    # Convert to BT
    if OutIASI is not None:
        print("min max OutIASI ",np.amin(OutIASI),np.amax(OutIASI))
        #OutIASI=(OutIASI>0)*bcon2/np.log(1.0+bcon1/(OutIASI*(OutIASI>0)+(OutIASI+0.1)*(OutIASI<=0)))+0.0*(OutIASI<=0)
        print("min max OutIASI ",np.amin(OutIASI),np.amax(OutIASI))
    #print("min max OutMHS  ",np.amin(OutMHS),np.amax(OutMHS))
    #OutMHS=bcon2Mhs/np.log(1.0+bcon1Mhs/OutMHS)
    #print("min max OutMHS  ",np.amin(OutMHS),np.amax(OutMHS))
    #print("OutMHS 45=",OutMHS[0,0,45,0,0])
    #print("OutMHS 45=",OutMHS[0,1,45,0,0])
    #print("OutMHS 45=",OutMHS[0,2,45,0,0])
    #print("OutMHS 46=",OutMHS[0,0,46,0,0])
    #print("OutMHS 46=",OutMHS[0,1,46,0,0])
    #print("OutMHS 46=",OutMHS[0,2,46,0,0])
    if OutIASI is not None:
        BtIASI=OutIASI[0,1,1,0,:,:]
    BtMHS=OutMHS[0,1,1,0,:,:]

    # Index 1 is for temperature
    # Calculation of derivaties of BT
    # JT ->  First derivatives (Jacobians) of temperature
    # J2T -> Second derivatives of temperature
    if OutIASI is not None:
        JT_IASI=np.zeros((nlevels,nbuf,nchanIASI.size))
        J2T_IASI=np.zeros((nlevels,nlevels,nbuf,nchanIASI.size))
    #J2RT_IASI=np.zeros((nlevels,nbuf,nchanIASI.size))
    JT_MHS=np.zeros((nlevels,nbuf,nchanMHS.size))
    J2T_MHS=np.zeros((nlevels,nlevels,nbuf,nchanMHS.size))
    #J2RD_MHS=np.zeros((nlevels,nbuf,nchanMHS.size))

    # Index 0 is for humidity
    # Calculation of derivaties of BT
    # JR ->  First derivatives (Jacobians) of humidity
    # J2R -> Second derivatives of humidity
    if OutIASI is not None:
        JR_IASI=np.zeros((nlevels,nbuf,nchanIASI.size))
        J2R_IASI=np.zeros((nlevels,nlevels,nbuf,nchanIASI.size))
        J2RT_IASI=np.zeros((nlevels,nlevels,nbuf,nchanIASI.size))
        #J2RD_IASI=np.zeros((nlevels,nbuf,nchanIASI.size))
    JR_MHS=np.zeros((nlevels,nbuf,nchanMHS.size))
    J2R_MHS=np.zeros((nlevels,nlevels,nbuf,nchanMHS.size))
    J2RT_MHS=np.zeros((nlevels,nlevels,nbuf,nchanMHS.size))
    #J2RD_MHS=np.zeros((nlevels,nbuf,nchanMHS.size))
    count=0
    for ilev1 in range(nlevels):
        for ilev2 in range(ilev1+1):
            if ilev1 == 60 and ilev2 == 60:
                print("Cross 2nd derivative MHS ",(OutMHS[0,2,2,count,0,2]-OutMHS[0,2,0,count,0,2]-OutMHS[0,0,2,count,0,2]+OutMHS[0,0,0,count,0,2])/(4.0*Dh*Dh))
                print("Direct 2nd derivative MHS ",(OutMHS[0,2,1,count,0,2]-2.0*OutMHS[0,1,1,count,0,2]+OutMHS[0,0,1,count,0,2])/Dh**2)
                print("for level p=",p[60])


            if ilev1 == ilev2:
                if OutIASI is not None:

                    JR_IASI[ilev1,:,:]=(OutIASI[0,2,1,count,:,:]-OutIASI[0,0,1,count,:,:])/(2.0*Dh)
                    J2R_IASI[ilev1,ilev2,:,:]=(OutIASI[0,2,1,count,:,:]-2.0*OutIASI[0,1,1,count,:,:]+OutIASI[0,0,1,count,:,:])/Dh**2
                JR_MHS[ilev1,:,:]=(OutMHS[0,2,1,count,:,:]-OutMHS[0,0,1,count,:,:])/(2.0*Dh)
                J2R_MHS[ilev1,ilev2,:,:]=(OutMHS[0,2,1,count,:,:]-2.0*OutMHS[0,1,1,count,:,:]+OutMHS[0,0,1,count,:,:])/Dh**2
                if nVars > 1:
                    if OutIASI is not None:

                        JT_IASI[ilev1,:,:]=(OutIASI[1,2,1,count,:,:]-OutIASI[1,0,1,count,:,:])/(2.0*DT)
                        J2T_IASI[ilev1,ilev2,:,:]=(OutIASI[1,2,1,count,:,:]-2.0*OutIASI[1,1,1,count,:,:]+OutIASI[1,0,1,count,:,:])/DT**2
                    JT_MHS[ilev1,:,:]=(OutMHS[1,2,1,count,:,:]-OutMHS[1,0,1,count,:,:])/(2.0*DT)
                    J2T_MHS[ilev1,ilev2,:,:]=(OutMHS[1,2,1,count,:,:]-2.0*OutMHS[1,1,1,count,:,:]+OutMHS[1,0,1,count,:,:])/DT**2
                    if OutIASI is not None:
                        J2RT_IASI[ilev1,ilev2,:,:]=(OutIASI[2,2,2,count,:,:]-OutIASI[2,2,0,count,:,:]-OutIASI[2,0,2,count,:,:]+OutIASI[2,0,0,count,:,:])/(4.0*Dh*DT)
                        J2RT_IASI[ilev2,ilev1,:,:]=(OutIASI[2,2,2,count,:,:]-OutIASI[2,2,0,count,:,:]-OutIASI[2,0,2,count,:,:]+OutIASI[2,0,0,count,:,:])/(4.0*Dh*DT)
                    J2RT_MHS[ilev1,ilev2,:,:]=(OutMHS[2,2,2,count,:,:]-OutMHS[2,2,0,count,:,:]-OutMHS[2,0,2,count,:,:]+OutMHS[2,0,0,count,:,:])/(4.0*Dh*DT)
                    J2RT_MHS[ilev2,ilev1,:,:]=(OutMHS[2,2,2,count,:,:]-OutMHS[2,2,0,count,:,:]-OutMHS[2,0,2,count,:,:]+OutMHS[2,0,0,count,:,:])/(4.0*Dh*DT)
                #if ilev1 == 63:
                #    print("p 63 ",p[63])
                #    print("OutIASI[0,2,63,0,0]=",OutIASI[1,2,1,count,0,0])
                #    print("OutIASI[0,1,63,0,0]=",OutIASI[1,1,1,count,0,0])
                #    print("OutIASI[0,0,63,0,0]=",OutIASI[1,0,1,count,0,0])
                #    print("JT CALC 1",OutIASI[1,2,1,count,0,0]-OutIASI[1,0,1,count,0,0])
                #    print("JT CALC 2",(OutIASI[1,2,1,count,0,0]-OutIASI[1,0,1,count,0,0])/(2.0*DT))
                #    print("JT level 63 =",JT_IASI[63,0,0])

            else:
                if OutIASI is not None:
                    J2R_IASI[ilev1,ilev2,:,:]=(OutIASI[0,2,2,count,:,:]-OutIASI[0,2,0,count,:,:]-OutIASI[0,0,2,count,:,:]+OutIASI[0,0,0,count,:,:])/(4.0*Dh*Dh)
                    J2R_IASI[ilev2,ilev1,:,:]=(OutIASI[0,2,2,count,:,:]-OutIASI[0,2,0,count,:,:]-OutIASI[0,0,2,count,:,:]+OutIASI[0,0,0,count,:,:])/(4.0*Dh*Dh)
                J2R_MHS[ilev1,ilev2,:,:]=(OutMHS[0,2,2,count,:,:]-OutMHS[0,2,0,count,:,:]-OutMHS[0,0,2,count,:,:]+OutMHS[0,0,0,count,:,:])/(4.0*Dh*Dh)
                J2R_MHS[ilev2,ilev1,:,:]=(OutMHS[0,2,2,count,:,:]-OutMHS[0,2,0,count,:,:]-OutMHS[0,0,2,count,:,:]+OutMHS[0,0,0,count,:,:])/(4.0*Dh*Dh)
                if nVars > 1:
                    if OutIASI is not None:
                        J2T_IASI[ilev1,ilev2,:,:]=(OutIASI[1,2,2,count,:,:]-OutIASI[1,2,0,count,:,:]-OutIASI[1,0,2,count,:,:]+OutIASI[1,0,0,count,:,:])/(4.0*DT*DT)
                        J2T_IASI[ilev2,ilev1,:,:]=(OutIASI[1,2,2,count,:,:]-OutIASI[1,2,0,count,:,:]-OutIASI[1,0,2,count,:,:]+OutIASI[1,0,0,count,:,:])/(4.0*DT*DT)
                    J2T_MHS[ilev1,ilev2,:,:]=(OutMHS[1,2,2,count,:,:]-OutMHS[1,2,0,count,:,:]-OutMHS[1,0,2,count,:,:]+OutMHS[1,0,0,count,:,:])/(4.0*DT*DT)
                    J2T_MHS[ilev2,ilev1,:,:]=(OutMHS[1,2,2,count,:,:]-OutMHS[1,2,0,count,:,:]-OutMHS[1,0,2,count,:,:]+OutMHS[1,0,0,count,:,:])/(4.0*DT*DT)
                    if OutIASI is not None:
                        J2RT_IASI[ilev1,ilev2,:,:]=(OutIASI[2,2,2,count,:,:]-OutIASI[2,2,0,count,:,:]-OutIASI[2,0,2,count,:,:]+OutIASI[2,0,0,count,:,:])/(4.0*Dh*DT)
                        J2RT_IASI[ilev2,ilev1,:,:]=(OutIASI[2,2,2,count,:,:]-OutIASI[2,2,0,count,:,:]-OutIASI[2,0,2,count,:,:]+OutIASI[2,0,0,count,:,:])/(4.0*Dh*DT)
                    J2RT_MHS[ilev1,ilev2,:,:]=(OutMHS[2,2,2,count,:,:]-OutMHS[2,2,0,count,:,:]-OutMHS[2,0,2,count,:,:]+OutMHS[2,0,0,count,:,:])/(4.0*Dh*DT)
                    J2RT_MHS[ilev2,ilev1,:,:]=(OutMHS[2,2,2,count,:,:]-OutMHS[2,2,0,count,:,:]-OutMHS[2,0,2,count,:,:]+OutMHS[2,0,0,count,:,:])/(4.0*Dh*DT)
                
            count+=1
    
    #JR_IASI=(OutIASI[0,2,1,0,:,:]-OutIASI[0,0,1,0,:,:])/(2.0*Dh)
    #J2R_IASI=(OutIASI[0,2,0,0,:]-2.0*OutIASI[0,1,:,:,:]+OutIASI[1,0,:,:,:])/Dh**2
    #JR_MHS=(OutMHS[0,2,1,0,:,:]-OutMHS[0,0,1,0,:,:])/(2.0*Dh)
    #J2R_MHS=(OutMHS[1,2,:,:,:]-2.0*OutMHS[1,1,:,:,:]+OutMHS[1,0,:,:,:])/Dh**2
    
    #print("avg J2R IASI ",np.average(J2R_IASI))
    #print("max min J2R IASI ",np.amax(J2R_IASI),np.amin(J2R_IASI))
    if OutIASI is not None:
        print("max min JR IASI ",np.amax(JR_IASI),np.amin(JR_IASI))

    # Preparing output file
    if stateType == 'ECMWF':
        out_filename='../data/RTTOVX2/RTTOVX2_ECMWF_'+radCols[0]+'_'+radCols[1]+'_'+radCols[2]+'.nc'
    else:
        out_filename='../data/RTTOVX2/RTTOVX2_Extrp_'+radCols[0]+'_'+radCols[1]+'_'+radCols[2]+'.nc'
        
    #out_filename='OBS_SYN_REG_X2_'+file
    print("out Filename ",out_filename)
    if os.path.isfile(out_filename):
        radF=netCDF4.Dataset(out_filename,'r+',format='NETCDF4')
    else:
        radF=netCDF4.Dataset(out_filename,'w',format='NETCDF4')
        
        #radF.createDimension('nrec',nrecs)
        radF.createDimension('nrec',1)
        nrecDim = radF.createVariable('nrec', recs.dtype,('nrec'))
        #recDim = radF.createVariable('rec', np.dtype('int64'),('rec'))
        #radF.variables['nrec'][:]=recs
        radF.variables['nrec'][:]=np.arange(1)
        
        radF.createDimension('nlevel',nlevels)
        nlevelDim = radF.createVariable('nlevel', recs.dtype,('nlevel'))
        #recDim = radF.createVariable('rec', np.dtype('int64'),('rec'))
        radF.variables['nlevel'][:]=np.arange(nlevels)
        
        radF.createDimension('ndTh',ndTh)
        ndThDim = radF.createVariable('ndTh', recs.dtype,('ndTh'))
        radF.variables['ndTh'][:]=np.arange(ndTh)

        radF.createDimension('nTq',nVars)
        nTqDim = radF.createVariable('nTq', recs.dtype,('nTq'))
        radF.variables['nTq'][:]=np.arange(nVars)
        
        if OutIASI is not None:
            radF.createDimension('nchanIASI',nchan_iasi)
            nchanIASIDim = radF.createVariable('nchanIASI', nchanIASI.dtype,('nchanIASI'))
            radF.variables['nchanIASI'][:]=nchanIASI
        
        radF.createDimension('nchanMHS',nchan_mhs)
        nchanMHSDim = radF.createVariable('nchanMHS', nchanMHS.dtype,('nchanMHS'))
        radF.variables['nchanMHS'][:]=nchanMHS
        
        if OutIASI is not None:
            radVarIASI = radF.createVariable('RAD_IASI', np.dtype('float64'),('nrec','nchanIASI'),zlib=True)
        radVarMHS = radF.createVariable('RAD_MHS', np.dtype('float64'),('nrec','nchanMHS'),zlib=True)

        if nVars > 1:
            if OutIASI is not None:
                radJTVarIASI = radF.createVariable('RAD_JT_IASI', np.dtype('float64'),('nlevel','nrec','nchanIASI'),zlib=True)
            radJTVarMHS  = radF.createVariable('RAD_JT_MHS', np.dtype('float64'),('nlevel','nrec','nchanMHS'),zlib=True)
            if OutIASI is not None:
                radJ2TVarIASI = radF.createVariable('RAD_J2T_IASI', np.dtype('float64'),('nlevel','nlevel','nrec','nchanIASI'),zlib=True)
            radJ2TVarMHS  = radF.createVariable('RAD_J2T_MHS', np.dtype('float64'),('nlevel','nlevel','nrec','nchanMHS'),zlib=True)
            if OutIASI is not None:
                radJ2RTVarIASI = radF.createVariable('RAD_J2RT_IASI', np.dtype('float64'),('nlevel','nlevel','nrec','nchanIASI'),zlib=True)
            radJ2RTVarMHS  = radF.createVariable('RAD_J2RT_MHS', np.dtype('float64'),('nlevel','nlevel','nrec','nchanMHS'),zlib=True)

        if OutIASI is not None:
            radJRVarIASI = radF.createVariable('RAD_JR_IASI', np.dtype('float64'),('nlevel','nrec','nchanIASI'),zlib=True)
        radJRVarMHS  = radF.createVariable('RAD_JR_MHS', np.dtype('float64'),('nlevel','nrec','nchanMHS'),zlib=True)
        
        if OutIASI is not None:
            radJ2RVarIASI = radF.createVariable('RAD_J2R_IASI', np.dtype('float64'),('nlevel','nlevel','nrec','nchanIASI'),zlib=True)
        #radJ2RDVarIASI = radF.createVariable('RAD_J2RD_IASI', np.dtype('float64'),('nlevel','nrec','nchanIASI'))
        radJ2RVarMHS  = radF.createVariable('RAD_J2R_MHS', np.dtype('float64'),('nlevel','nlevel','nrec','nchanMHS'),zlib=True)
        #radJ2RDVarMHS = radF.createVariable('RAD_J2RD_MHS', np.dtype('float64'),('nlevel','nrec','nchanMHS'))

        #radF.variables['rads'][:]=0

    #radF.variables['RAD_IASI'][idx:idx+nbuf,:]=BtIASI
    #radF.variables['RAD_MHS'][idx:idx+nbuf,:]=BtMHS
    if OutIASI is not None:
        radF.variables['RAD_IASI'][0,:]=BtIASI
    radF.variables['RAD_MHS'][0,:]=BtMHS

    zero_dimension=1
    radF.createDimension('zero_dimension',1)
    dsetZ=radF.createVariable('zero_dimension',type(zero_dimension),('zero_dimension'))
    dsetZ.units='1'
    radF.variables['zero_dimension'][:]=zero_dimension
    
    nMin_dimension=np.arange(nMin,dtype=np.int64)
    radF.createDimension('nMin_dimension',nMin)
    dsetNMin=radF.createVariable('nMin_dimension',nMin_dimension.dtype,('nMin_dimension'))
    dsetNMin.units='1'
    radF.variables['nMin_dimension'][:]=nMin_dimension

    radVarMHSObs = radF.createVariable('RAD_MHS_OBS', minRadMhs.dtype,('nrec','nchanMHS','nMin_dimension'),zlib=True)
    radF.variables['RAD_MHS_OBS'][0,:,:]=minRadMhs
    varMHSObsLon = radF.createVariable('MHS_LON', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_LON'][0,:]=mhsObs.lon
    varMHSObsLat = radF.createVariable('MHS_LAT', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_LAT'][0,:]=mhsObs.lat
    varMHSObsTime = radF.createVariable('MHS_COLOC_TIMEDIFF', np.dtype('float64'),('nrec','nMin_dimension'),zlib=True)
    radF.variables['MHS_COLOC_TIMEDIFF'][0,:]=timeMhs
    varMHSObsTime = radF.createVariable('MHS_COLOC_DIST', np.dtype('float64'),('nrec','nMin_dimension'),zlib=True)
    radF.variables['MHS_COLOC_DIST'][0,:]=distMhs
    varMHSObsSunazi = radF.createVariable('MHS_SUNAZI', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_SUNAZI'][0,:]=mhsObs.sunazi
    varMHSObsSunzen = radF.createVariable('MHS_SUNZEN', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_SUNZEN'][0,:]=mhsObs.sunzen
    varMHSObsSatazi = radF.createVariable('MHS_SATAZI', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_SATAZI'][0,:]=mhsObs.satazi
    varMHSObsSatzen = radF.createVariable('MHS_SATZEN', np.dtype('float64'),('nrec','zero_dimension'),zlib=True)
    radF.variables['MHS_SATZEN'][0,:]=mhsObs.satzen
    
    
    #print("rad satzen ",mhsObs.rad.shape,mhsObs.satzen)
    #print("BtMHS ",BtMHS.shape)
    ## Checking with previous work
    #JE=np.load('/home/xcalbet/gradient/RGFandRTM/2DOpt/JE.npy')
    #print("JE=",JE.shape)
    #
    ## Base
    #f0,tx0,Tb0,tau0=np.loadtxt('/home/xcalbet/gradient/RGFandRTM/2DOpt/dataMan/ManusE46_1.out',unpack=True)
    #plt.plot(freqMhs[2:5],OutMHS[1,1,46,:,2:5].T,'o',color='red')
    #plt.plot(f0,Tb0,color='red')
    ## Negative perturbation in E
    #fn,txn,Tbn,taun=np.loadtxt('/home/xcalbet/gradient/RGFandRTM/2DOpt/dataMan/ManusE46_0.out',unpack=True)
    #plt.plot(fn,Tbn,color='blue')
    #plt.plot(freqMhs[2:5],OutMHS[1,0,46,:,2:5].T,'o',color='blue')
    ## Positive perturbation in E
    #fp,txp,Tbp,taup=np.loadtxt('/home/xcalbet/gradient/RGFandRTM/2DOpt/dataMan/ManusE46_2.out',unpack=True)
    #plt.plot(fp,Tbp,color='green')
    #plt.plot(freqMhs[2:5],OutMHS[1,2,46,:,2:5].T,'o',color='green')
    #plt.show()
    #
    ## Both perturbations in E
    #plt.plot(fn,(Tbp-Tbn)/(2.0*Dh),color='red')
    #plt.plot(f0,JE[:,46],color='orange')
    #plt.plot(freqMhs[2:5],(OutMHS[1,2,46,:,2:5].T-OutMHS[1,0,46,:,2:5].T)/(2.0*Dh),'o',color='red')
    ## Negative perturbation in E
    #plt.plot(fn,(Tb0-Tbn)/Dh,color='blue')
    #plt.plot(freqMhs[2:5],(OutMHS[1,1,46,:,2:5].T-OutMHS[1,0,46,:,2:5].T)/Dh,'o',color='blue')
    ## Positive perturbation in E
    #plt.plot(fp,(Tbp-Tb0)/Dh,color='green')
    #plt.plot(freqMhs[2:5],(OutMHS[1,2,46,:,2:5].T-OutMHS[1,1,46,:,2:5].T)/Dh,'o',color='green')
    #plt.show()
    #
    ## Second derivative
    #plt.plot(freqMhs[2:5],(OutMHS[1,2,46,:,2:5].T-2.0*OutMHS[1,1,46,:,2:5].T+OutMHS[1,0,46,:,2:5].T)/(Dh**2),'o',color='red')
    #plt.plot(fn,(Tbp-2.0*Tb0+Tbn)/(Dh**2),color='blue')
    #plt.show()
    #
    #print("Tbn=",Tbn)
    #print("Tb0=",Tb0)
    #print("Tbp=",Tbp)
    #quit()

    
    #radF.variables['RAD_JR_IASI'][:,idx:idx+nbuf,:]=JR_IASI
    #radF.variables['RAD_JR_MHS'][:,idx:idx+nbuf,:]=JR_MHS
    #
    #radF.variables['RAD_J2R_IASI'][:,:,idx:idx+nbuf,:]=J2R_IASI
    ##radF.variables['RAD_J2RD_IASI'][:,idx:idx+nbuf,:]=J2RD_IASI
    #radF.variables['RAD_J2R_MHS'][:,:,idx:idx+nbuf,:]=J2R_MHS
    ##radF.variables['RAD_J2RD_MHS'][:,idx:idx+nbuf,:]=J2RD_MHS
    #if nVars > 1:
    #    radF.variables['RAD_JT_IASI'][:,idx:idx+nbuf,:]=JT_IASI
    #    radF.variables['RAD_J2T_IASI'][:,:,idx:idx+nbuf,:]=J2T_IASI
    #    radF.variables['RAD_JT_MHS'][:,idx:idx+nbuf,:]=JT_MHS
    #    radF.variables['RAD_J2T_MHS'][:,:,idx:idx+nbuf,:]=J2T_MHS

    if OutIASI is not None:
        radF.variables['RAD_JR_IASI'][:,0,:]=JR_IASI
    radF.variables['RAD_JR_MHS'][:,0,:]=JR_MHS

    if OutIASI is not None:
        radF.variables['RAD_J2R_IASI'][:,:,0,:]=J2R_IASI
    #radF.variables['RAD_J2RD_IASI'][:,0,:]=J2RD_IASI
    radF.variables['RAD_J2R_MHS'][:,:,0,:]=J2R_MHS
    #radF.variables['RAD_J2RD_MHS'][:,0,:]=J2RD_MHS
    if nVars > 1:
        if OutIASI is not None:
            radF.variables['RAD_JT_IASI'][:,0,:]=JT_IASI
            radF.variables['RAD_J2T_IASI'][:,:,0,:]=J2T_IASI
            radF.variables['RAD_J2RT_IASI'][:,:,0,:]=J2RT_IASI
        radF.variables['RAD_JT_MHS'][:,0,:]=JT_MHS
        radF.variables['RAD_J2T_MHS'][:,:,0,:]=J2T_MHS
        radF.variables['RAD_J2RT_MHS'][:,:,0,:]=J2RT_MHS


        
    print("Closing Output file")
    radF.close()
    fLog.write("out "+out_filename+"\n")
    fLog.close()

    return


if __name__ == "__main__":

    emsPine=np.fromfile("pineold.bin",dtype='>f')
    radListFil=open('overpass_purged_gruan_radiosonde_sequential_list.txt','r')
    #radListFil=open('test.txt','r')
    pLevs=pressLevels.L50().levels

    countLine=0
    for radLine in radListFil:
        countLine+=1
        if countLine == 1:
            fLog=open('calcRad.log','w')
        else:
            fLog=open('calcRad.log','a')

        fLog.write("countLine ="+str(countLine)+"**********************************************************************************************\n")
        print("countLine =",countLine,"**********************************************************************************************")
        # For some reason it collapses and has to be done in chunks
        # Reason found: Pool has a memory leak and after 40 iterations with MHS it fills up RAM and Swap
        if countLine < -1:
        #if countLine < 41:
        #if countLine < 69:
            continue
        radLine=radLine.strip()
        #radLine='20070716201156 LIN-RS-01_2_RS92-GDP_002_20070716T191100_1-000-001.nc LIN-RS-01_2_RS92-GDP_002_20070716T200600_1-000-001.nc'  # For testing purposes
        radCols=radLine.split()

        fLog.write(radCols[0]+"  "+radCols[1]+"  "+radCols[2]+"\n")
        print(radCols[0],"  ",radCols[1],"  ",radCols[2])
        oTimeStr=radCols[0]
        sondFil=[None]*2
        sondFil[0]='../data/Lindenberg_GRUAN_2007/'+radCols[1]
        sondFil[1]='../data/Lindenberg_GRUAN_2007/'+radCols[2]
        
        sond=sonde.sonde()
        sond.readGRUANSonde(sondFil[1])

        # Overpass Time in datetime format
        oTime=datetime.datetime(int(oTimeStr[0:4]),int(oTimeStr[4:6]),int(oTimeStr[6:8]),hour=int(oTimeStr[8:10]),minute=int(oTimeStr[10:12]),second=int(oTimeStr[12:14]))
        #print("oTime ",oTime)
        mhsObs=radiance.mhs()
        nMin=4
        distMhs,timeMhs,minRadMhs=mhsObs.closestMHS(sond.siteLon,sond.siteLat,oTime,'../data/MHS',nMin)
        if distMhs is None:
            print("Error: NO MHS Obs data found\n")
            fLog.write("Error: NO MHS Obs data found\n")
            continue
        fLog.write("distMhs timeMhs "+str(distMhs)+"  "+str(timeMhs)+"\n")
        print("distMhs timeMhs minRadMhs ",distMhs,timeMhs,minRadMhs)
        #print("rad satzen ",mhsObs.rad,mhsObs.satzen)

        # Read WVVar file
        print('../data/WVVar/WVVar_'+radCols[0]+'_'+radCols[1]+'_'+radCols[2]+'.nc')
        ncWVVar=netCDF4.Dataset('../data/WVVar/WVVar_'+radCols[0]+'_'+radCols[1]+'_'+radCols[2]+'.nc','r')
        if stateType == 'ECMWF':
            print("using ECMWF state")
            statEcmwf=sonde.state()
            statEcmwf.readNc('ECMWF State',ncWVVar)
            state=statEcmwf
        else:
            print("using Extrp state")
            statExtrp=sonde.state()
            statExtrp.readNc('Extrapolated State',ncWVVar)
            state=statExtrp

        #print("stat ecmwf pres ",state.press)
        #print("pLevs ",pLevs)

        # Preparing variables for RTTOV
        satzen=np.array([mhsObs.satzen])
        
        lon=expand2nprofiles(np.array([mhsObs.lon]),1)
        lat=expand2nprofiles(np.array([mhsObs.lat]),1)
        
        cfr=expand2nprofiles(np.array([0.0]),1)
        lfr=expand2nprofiles(np.array([1.0]),1)
        #NON OK
        pO=state.press
        #print("pO=",pO.shape)
        
        sp=expand2nprofiles(np.array([state.surfPress]),1)
        #print("New sp=",sp)
            
        tO=expand2nprofiles(state.T,1)
        #print("tO=",tO.shape)
        qO=expand2nprofiles(state.q,1)
        qO=qO*(qO>0.0)+0.0*(qO<=0.0)
            
        o3O=expand2nprofiles(state.O3,1)   # kg/kg -> ppmv
        co2O=expand2nprofiles((np.zeros(state.T.size)+384.02)/rcnco2,1)
        satazi=expand2nprofiles(np.array([mhsObs.satazi]),1)
        sunzen=expand2nprofiles(np.array([mhsObs.sunzen]),1)
        sunazi=expand2nprofiles(np.array([mhsObs.sunazi]),1)
        sat=expand2nprofiles(np.array([state.surfT+273.15]),1)
        skt=expand2nprofiles(np.array([state.skt]),1)
        sq=expand2nprofiles(np.array([humidity.RH2q(state.surfRH/100.0,state.surfPress,state.surfT+273.15)]),1)
        
        year=expand2nprofiles(np.array([state.startTime.year]),1)
        month=expand2nprofiles(np.array([state.startTime.month]),1)
        day=expand2nprofiles(np.array([state.startTime.day]),1)
                  
        hour=expand2nprofiles(np.array([state.startTime.hour]),1)
        minu=expand2nprofiles(np.array([state.startTime.minute]),1)
        secs=expand2nprofiles(np.array([state.startTime.second]),1)
        # Finished reading input file
        
            
        #print("pO=",pO.shape)
        #print("tO=",tO.shape)
        #print("qO=",qO.shape)
        #print("tO=",tO.shape)
        #print("pO=",pO.shape)
        #print("tO=",tO.shape)
        #print("qO=",qO.shape)
        #print("tO=",tO.shape)
        
        if NLEVELS == 50:# 50 Levels
            t=np.zeros((tO.shape[0],pLevs.size))
            q=np.zeros((qO.shape[0],pLevs.size))
            o3=np.zeros((o3O.shape[0],pLevs.size))
            co2=np.zeros((co2O.shape[0],pLevs.size))
            for irec in range(tO.shape[0]):
                t[irec,:]=np.interp(np.log(pLevs),np.log(pO),tO[irec,:])
                q[irec,:]=np.interp(np.log(pLevs),np.log(pO),qO[irec,:])
                o3[irec,:]=np.interp(np.log(pLevs),np.log(pO),o3O[irec,:])
                co2[irec,:]=np.interp(np.log(pLevs),np.log(pO),co2O[irec,:])
            #print("t=",t.shape)
            p=pLevs
        elif NLEVELS == 90:
            # 90 Levels
            t=tO
            q=qO
            o3=o3O
            co2=co2O
            p=pO
            
        #print("t=",t)
        #print("q=",q.shape)
        #print("q=",q)
        #print("o3=",o3)
        #print("co2=",co2)
        #quit()
        #e=humidity.q2e(q,p)
        #dwptE=humidity.e2Tdew(e)
        #print("dwptE=",dwptE.shape)
        #eO=humidity.q2e(qO,pO)
        #dwptO=humidity.e2Tdew(eO)
        
        # Plot profile
        #plt.ylim(1100,10)
        #plt.semilogy(t.T,p)
        #plt.semilogy(tO.T,pO)
        #plt.semilogy(dwptE.T,p,color='red')
        #plt.semilogy(dwptO.T,pO)
        #plt.show()
        #quit()
        
        #plt.plot(o3[30,:],-np.log10(p))
        #plt.plot(o3O[30,:],-np.log10(pO),'r')
        #plt.show()
        #quit()
        
        nlevels = len(p)
        #print("nlevels=",nlevels)
        # Number of positions to calculate derivatie
        ndTh=3
        DT=0.5
        dT=np.linspace(-DT,DT,num=ndTh)
        #Dh=0.2
        Dh=0.15
        dh=np.linspace(-Dh,Dh,num=ndTh)
        
        nrecs=sp.size
        #print("New nrecs=",nrecs)
        recs=np.arange(nrecs)
        #print(recs.dtype)

        mhsCalc=radiance.mhs()
        iasiCalc=radiance.iasi()
        # For HIRS and MHS we will read all channels, but we will read a subset
        # for SEVIRI
        nchan_iasi=8461
        nchan_mhs = 5
        
        #nchanIASI=np.arange(nchan_iasi)
        nchan_iasi=iasiCalc.idxWn.size
        nchanIASI=iasiCalc.idxWn
        nchanIASI_tuple=tuple(iasiCalc.idxWn.tolist())
        #print("nchanIASI=",nchanIASI_tuple)
        nchanMHS=np.arange(nchan_mhs)
        
        
        nbufGran=1
        
        # Declare an instance of Profiles
        #nbuf=nbufGran
        #nlevels = len(p)
        #nprofiles = nbuf
        #nprofiles=2*ndTh*nlevels*nbuf
        #myProfiles = pyrttov.Profiles(nprofiles, nlevels)
        
        # ------------------------------------------------------------------------
        # Set up Rttov instances for each instrument
        # ------------------------------------------------------------------------
        # Create three Rttov objects for three instruments
        mhsRttov = pyrttov.Rttov()
        iasiRttov = pyrttov.Rttov()
        
        # Set the options for each Rttov instance:
        # - the path to the coefficient file must always be specified
        # - turn RTTOV interpolation on (because input pressure levels differ from
        #   coefficient file levels)
        # - set the verbose_wrapper flag to true so the wrapper provides more
        #   information
        # - enable solar simulations for SEVIRI
        # - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
        #   the SEVIRI and MHS simulations)
        # - enable the store_trans wrapper option for MHS to provide access to
        #   RTTOV transmission structure
        
        iasiRttov.FileCoef = '{}/{}'.format(rttov_installdir,
                                            #"rtcoef_rttov13/rttov7pred54L/rtcoef_metop_2_iasi.H5")
                                           #"rtcoef_rttov13/rttov7pred101L/rtcoef_metop_2_iasi.H5")
                                            "rtcoef_rttov13/rttov13pred101L/rtcoef_metop_2_iasi_o3co2.H5")
        iasiRttov.Options.AddInterp = True
        iasiRttov.Options.VerboseWrapper = True
        #iasiRttov.Options.CO2Data = True
        iasiRttov.Options.OzoneData = True
        
        mhsRttov.FileCoef = '{}/{}'.format(rttov_installdir,
                                           "rtcoef_rttov13/rttov13pred54L/rtcoef_metop_2_mhs.dat")
        mhsRttov.Options.AddInterp = True
        mhsRttov.Options.StoreTrans = True
        mhsRttov.Options.VerboseWrapper = True
        #mhsRttov.Options.CO2Data = True
        #mhsRttov.Options.OzoneData = True
        
        # Load the instruments: for HIRS and MHS do not supply a channel list and
        # so read all channels
        try:
            mhsRttov.loadInst()
            iasiRttov.loadInst(nchanIASI_tuple)
        except pyrttov.RttovError as e:
            sys.stderr.write("Error loading instrument(s): {!s}".format(e))
            sys.exit(1)
        
        
        # ------------------------------------------------------------------------
        # Load the emissivity and BRDF atlases
        # ------------------------------------------------------------------------
        
        # Load the emissivity and BRDF atlases:
        # - load data for the month in the profile data
        # - load the IR emissivity atlas data for multiple instruments so it can be used for SEVIRI and HIRS
        # - SEVIRI is the only VIS/NIR instrument we can use the single-instrument initialisation for the BRDF atlas
        
        idx=0
        monthOld=month[idx]
        irAtlas = pyrttov.Atlas()
        irAtlas.AtlasPath = '{}/{}'.format(rttov_installdir, "emis_data")
        irAtlas.loadIrEmisAtlas(month[idx], ang_corr=True) # Include angular correction, but do not initialise for single-instrument
        
        # TELSEM2 atlas does not require an Rttov object to initialise
        mwAtlas = pyrttov.Atlas()
        mwAtlas.AtlasPath = '{}/{}'.format(rttov_installdir, "emis_data")
        mwAtlas.loadMwEmisAtlas(month[idx])
        
        #for idx in range(0,nrecs,nbufGran):
        #for idx in range(iProc,iProc+1):
        #    proc_rttov(idx)
        #pool=Pool(processes=1)
        #pool=Pool(processes=4)
        #pool=Pool(processes=3)
        #print("nrecs=",nrecs)
        #pool.map(proc_rttov,[z for z in range(nrecs)])
        proc_rttov(0)
        #pool.map(proc_rttov,[z for z in range(154,216)])
        
        # ------------------------------------------------------------------------
        # Deallocate memory
        # ------------------------------------------------------------------------
        
        # Because of Python's garbage collector, there should be no need to
        # explicitly deallocate memory
        del mhsRttov
        del iasiRttov
        del irAtlas
        del mwAtlas
        
        #quit()
