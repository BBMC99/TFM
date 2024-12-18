import pygrib
import numpy as np

def grib2np(filename,verbose=True):
    # Opening file with pygrib

    grbDic={}

    grbs = pygrib.open(filename)
    # We first need to parse all the pressure levels
    levDict={}
    for grb2 in grbs:
        if grb2.typeOfLevel=='isobaricInhPa':
            lev2=grb2.level
            if lev2 not in levDict:
                levDict[lev2]=lev2
    pLevels=np.array(sorted(levDict.keys()))
    grbDic['p'] = pLevels

    grbs = pygrib.open(filename)
    grbLevDic={}
    for grb in grbs:
    
        keyGrb=grb.shortName
        if keyGrb in grbLevDic:
            if grbLevDic[keyGrb] != grb.typeOfLevel:
                keyGrb = keyGrb + grb.typeOfLevel
                grbLevDic[keyGrb]=grb.typeOfLevel    
        else:
            grbLevDic[keyGrb]=grb.typeOfLevel    
        if verbose:
            print(keyGrb,'-->',grb)
    
        Ni=grb['Ni']
        #print("Ni=",Ni)
        Nj=grb['Nj']
        #print("Nj=",Nj)
        Nk=grb['numberOfVerticalCoordinateValues']//2
        #print("Nk=",Nk)
        #print("type of level ",grb.typeOfLevel)
        if grb.typeOfLevel == 'hybrid':
            # And the a_k and b_k coefficients which convert sigma levels to
            # pressure
            #pv=grb['pv']
            #print("pv=",pv)
            #ak=pv[0:Nk]
            #bk=pv[Nk:]
            #print("ak=",ak.shape)
            #print("bk=",bk.shape)
            lev=grb.level-1
            #print("lev=",lev)
            
            if keyGrb not in grbDic:
                grbDic[keyGrb]=np.zeros((Nj,Ni,Nk-1))
                grbDic[keyGrb][:,:,lev]=grb["values"]
            else:
                grbDic[keyGrb][:,:,lev]=grb["values"]
    
        elif grb.typeOfLevel=='surface':
            grbDic[keyGrb]=grb["values"]
        elif grb.typeOfLevel=='isobaricInhPa':
            #print("levels=",pLevels)
            #print("Nk=",Nk)
            #print("level=",grb.level)
            lev=grb.level
            ilev=np.where(pLevels==lev)[0][0]
            #print("ilev=",ilev)
            #print("values ",grb["values"].shape)
            if keyGrb not in grbDic:
                grbDic[keyGrb]=np.zeros((Nj,Ni,pLevels.size))
                grbDic[keyGrb][:,:,ilev]=grb["values"]
            else:
                grbDic[keyGrb][:,:,ilev]=grb["values"]
    
                    
    # And now we take care of the geographical coordinates
    grbDic['latInit']=grb['latitudeOfFirstGridPointInDegrees']
    grbDic['latEnd']=grb['latitudeOfLastGridPointInDegrees']
    grbDic['lonInit']=grb['longitudeOfFirstGridPointInDegrees']
    grbDic['lonEnd']=grb['longitudeOfLastGridPointInDegrees']
    # And the lats and lons
    grbDic['lat'],grbDic['lon']=grb.latlons()
    if verbose:
        print("latInit=",grbDic['latInit'])
        print("latEnd=",grbDic['latEnd'])
        print("lonInit=",grbDic['lonInit'])
        print("lonEnd=",grbDic['lonEnd'])

    grbs = pygrib.open(filename)
    # We first read sp
    for grb in grbs:
        keyGrb=grb.shortName
        if (keyGrb == 'sp'):
            #print(grb["values"])
            grbDic[keyGrb]=grb["values"]
            #print("grbDic ",keyGrb,grbDic.keys())
        #print("sp ",grbDic['sp'])
        if (grb.typeOfLevel == 'hybrid') and ('sp' in grbDic):
            # And the a_k and b_k coefficients which convert sigma levels to
            # pressure
            Nk=grb['numberOfVerticalCoordinateValues']//2
            pv=grb['pv']
            #print("pv=",pv)
            ak=pv[0:Nk]
            bk=pv[Nk:]
            #print("ak=",ak.shape)
            #print("bk=",bk.shape)
             # We calculate the pressure levels for each sigma level
            # at each grid point
            spre12=(bk*grbDic['sp'][:,:,np.newaxis]+ak)/100.0
            #print("spre12=",spre12.shape)
            spre1=spre12[:,:,0:-1]
            spre2=spre12[:,:,1:]
            grbDic['p']=(spre1+spre2)/2.0
            grbDic['sp']/=100.0
            #print("p=",p.shape)
            break
        else:
            if (grb.typeOfLevel == 'hybrid'):
                print("Warning: no pressure field calculated because no surface pressure available")
        
    return grbDic
    
# GRIB file to read
#filename='ecmwf/ECMWF_20000103_06+16.grib'

#ecmwf=grib2np(filename,verbose=True)
#print("ecmwf=",ecmwf)
