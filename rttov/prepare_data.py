import numpy as np
import grib2np
import netCDF4toNumpy
from pprint import pprint
import example_data as ex
import re
from datetime import datetime

# Función principal para preparar los datos de ECMWF y MHS
def prepare_all_data(ECMWF_data, MHS_data):
    # mhs_data = prepare_MHS_data(MHS_data)
    # ecmwf_data = prepare_ECMWF_data(ECMWF_data)

    mhs_data = netCDF4toNumpy.read_netcdf_to_dict(MHS_data)
    ecmwf_data = grib2np.grib2np(ECMWF_data)

    # get the date from the file name
    date = get_date(MHS_data)

    # Redondear latitudes y longitudes de MHS
    rounded_lat_matrix = np.vectorize(round_to_05)(mhs_data['lat'])
    rounded_lon_matrix = np.vectorize(round_to_05)(mhs_data['lon'])

    # Listas para almacenar los índices coincidentes
    indices = []

    # Iterar sobre cada par (lat, lon) en ECMWF
    for ecmwf_i, ecmwf_j in np.ndindex(ecmwf_data['lat'].shape):
        lat, lon = ecmwf_data['lat'][ecmwf_i, ecmwf_j], ecmwf_data['lon'][ecmwf_i, ecmwf_j]

        # Encontrar coincidencias en MHS
        match = np.where((rounded_lat_matrix == lat) & (rounded_lon_matrix == lon))

        # Añadir las coordenadas coincidentes
        for mhs_i, mhs_j in zip(*match):
            indices.append(((mhs_i, mhs_j), (ecmwf_i, ecmwf_j)))

    # Filtrar valores de ECMWF basados en los índices
    filtered_values_p, filtered_values_q, filtered_values_t = [], [], []
    filtered_values_solar_azimuth, filtered_values_solar_zenith = [], [] 
    filtered_values_satellite_azimuth, filtered_values_satellite_zenith = [], []
    filtered_values_lat, filtered_values_lon = [], []
    filtered_values_angles, filtered_values_skin = [], []
    filtered_values_surftype,filtered_values_surfgeom = [], []
    filtered_values_surface_properties, filtered_values_terrain_elevation = [], []
    filtered_values_2t, filtered_values_2d, filtered_values_sp = [], [], []
    datetimes = []

    for (mhs_idx, ecmwf_idx) in indices:
        ecmwf_i, ecmwf_j = ecmwf_idx
        mhs_i, mhs_j = mhs_idx
        
        # ECMWF
        filtered_values_p.append(ecmwf_data['p'][ecmwf_i, ecmwf_j])
        filtered_values_q.append(ecmwf_data['q'][ecmwf_i, ecmwf_j])
        filtered_values_t.append(ecmwf_data['t'][ecmwf_i, ecmwf_j])
        filtered_values_skin.append(ecmwf_data['skt'][ecmwf_i, ecmwf_j])
        filtered_values_2d.append(ecmwf_data['2d'][ecmwf_i, ecmwf_j])
        filtered_values_2t.append(ecmwf_data['2t'][ecmwf_i, ecmwf_j])
        filtered_values_sp.append(ecmwf_data['sp'][ecmwf_i, ecmwf_j])

        # MHS
        filtered_values_solar_azimuth.append(mhs_data['solar_azimuth'][mhs_i, mhs_j])
        filtered_values_solar_zenith.append(mhs_data['solar_zenith'][mhs_i, mhs_j])
        filtered_values_satellite_azimuth.append(mhs_data['satellite_azimuth'][mhs_i, mhs_j])
        filtered_values_satellite_zenith.append(mhs_data['solar_zenith'][mhs_i, mhs_j])

        filtered_values_lat.append(mhs_data['lat'][mhs_i, mhs_j])
        filtered_values_lon.append(mhs_data['lon'][mhs_i, mhs_j])
        filtered_values_terrain_elevation.append(mhs_data['terrain_elevation'][mhs_i, mhs_j])
        filtered_values_surface_properties.append(mhs_data['surface_properties'][mhs_i, mhs_j])
        
        # dates
        datetimes.append(date)

    # Convertir a arrays NumPy
    ## ECMWF
    filtered_values_p = np.array(filtered_values_p)
    filtered_values_q = np.array(filtered_values_q)
    filtered_values_t = np.array(filtered_values_t)
    filtered_values_2t = np.array(filtered_values_2t)
    filtered_values_2d = np.array(filtered_values_2d)
    filtered_values_sp = np.array(filtered_values_sp)
    filtered_values_skin = np.array(filtered_values_skin)

    ecmwf_data['p_filtered'] = filtered_values_p
    ecmwf_data['q_filtered'] = filtered_values_q
    ecmwf_data['t_filtered'] = filtered_values_t
    ecmwf_data['skin_filtered'] = format_skt(filtered_values_skin)
    ecmwf_data['s2m_filtered'] = format_s2m(filtered_values_2t, filtered_values_2d, filtered_values_sp)

    ## MHS
    filtered_values_angles = np.array(filtered_values_angles)
    filtered_values_surftype = np.array(filtered_values_surftype)
    filtered_values_surfgeom = np.array(filtered_values_surfgeom)

    # date
    datetimes = np.array(datetimes)
    # Combinar datos de ángulos
    mhs_data['angles'] = format_angles(filtered_values_solar_azimuth, filtered_values_solar_zenith, 
                                       filtered_values_satellite_azimuth, filtered_values_satellite_zenith)

    # Formatear geometría de la superficie
    mhs_data['surfgeom'] = format_surfgeom(filtered_values_lat, filtered_values_lon, filtered_values_terrain_elevation)

    # Formatear tipo de superficie
    mhs_data['surftype_filtered'] = format_surftype(filtered_values_surface_properties)
    mhs_data['angles_filtered'] = filtered_values_angles
    mhs_data['surfgeom_filtered'] = filtered_values_surfgeom
    
    # dates
    mhs_data['datetimes'] = datetimes

    # Prints para chequear
    #print('2t: ', ecmwf_data['2t'])
    #print('2d: ', ecmwf_data['2d'])
    #print('2q: ', ecmwf_data['2q'])
    return mhs_data, ecmwf_data


# Función para preparar datos de ECMWF
def prepare_ECMWF_data(ECMWF_data):
    ECMWF_dict = grib2np.grib2np(ECMWF_data)
    
    # Obtener datos de la superficie
    skin = format_skt(ECMWF_dict['skt'])
    ECMWF_dict['skin'] = skin
    return ECMWF_dict


# Función para preparar datos de MHS
def prepare_MHS_data(MHS_data):
    MHS_dict = netCDF4toNumpy.read_hdf5_to_numpy_arrays(MHS_data)

    # Combinar datos de ángulos
    angles = format_angles(MHS_dict['solar_azimuth'], MHS_dict['solar_zenith'],
                           MHS_dict['satellite_azimuth'], MHS_dict['satellite_zenith'])
    MHS_dict['angles'] = angles

    # Formatear geometría de la superficie
    surfgeom = format_surfgeom(MHS_dict['lat'], MHS_dict['lon'], MHS_dict['terrain_elevation'])
    MHS_dict['surfgeom'] = surfgeom

    # Formatear tipo de superficie
    surftype = format_surftype(MHS_dict['surface_properties'])
    MHS_dict['surftype'] = surftype

    return MHS_dict

# Redondear valores al múltiplo más cercano de 0.5
def round_to_05(value):
    return round(value * 2) / 2

# Combinar ángulos
def format_angles(solar_azi, solar_zen, sat_azi, sat_zen):
    if not (len(solar_azi) == len(solar_zen) == len(sat_azi) == len(sat_zen)):
        raise ValueError("Todos los arrays de entrada deben tener la misma longitud.")
    return np.array([sat_zen, sat_azi, solar_zen, solar_azi ]).T

# Formatear datos de temperatura superficial
def format_skt(skt):
    fixed_values = [35., 0., 0., 3.0, 5.0, 15.0, 0.1, 0.3]
    skt = np.asarray(skt).reshape(-1, 1)
    fixed_values = np.asarray(fixed_values).reshape(1, -1)
    return np.hstack((skt, np.tile(fixed_values, (skt.shape[0], 1))))

# Formatear datos de temp, presion, humedad etc. a dos metros
def format_s2m(t2, d2, sp):

    fixed_values = [0., 0., 0.]

    # Ensure all are column vectors (105012,1)
    t2 = np.asarray(t2).reshape(-1, 1)
    d2 = np.asarray(d2).reshape(-1, 1)
    sp = np.asarray(sp).reshape(-1, 1)

    surfRH = np.asarray(Tdew2RH(d2, t2)).reshape(-1, 1)
    sqB = np.asarray(RH2q(surfRH, sp, t2, 1)).reshape(-1, 1)
    fixed_values = np.asarray(fixed_values).reshape(1, -1)

    return np.hstack((sp, t2, sqB, np.tile(fixed_values, (t2.shape[0], 1))))


# Formatear geometría de la superficie
def format_surfgeom(lat, lon, elev):
    if not (len(lat) == len(lon) == len(elev)):
        raise ValueError("Todos los arrays de entrada deben tener la misma longitud.")
    elev_scaled = []
    # we need the elevation in km not in m
    for value in elev:
        elev_scaled.append(value / 1000)

    return np.array([lat, lon, elev_scaled]).T


# Formatear tipo de superficie
def format_surftype(surface_properties):
    fixed_values = 0
    surface_properties = np.asarray(surface_properties).reshape(-1, 1)
    return np.hstack((surface_properties, np.tile(fixed_values, (surface_properties.shape[0], 1))))

def get_date(filename):
    match = re.search(r'(\d{8})(\d{6})', filename)
    
    if match:
        date_part = match.group(1)  
        time_part = match.group(2)  

        # Convert date and time to the desired format
        try:
            date_obj = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
            return [
                date_obj.strftime("%Y"),
                date_obj.strftime("%m"),
                date_obj.strftime("%d"),
                date_obj.strftime("%H"),
                date_obj.strftime("%M"),
                date_obj.strftime("%S")
            ]
        except ValueError:
            return []

    return []  # Return empty list if no match is found


# -----------------------------------
# Humidity functions

def Tdew2RH(Tdew,T,*magnus):
        # Checked
        # Dew point temperature (K) plus temperature (K) to relative humidity (1)
        ## Para dwpt de HylandAndWexler: Tdew(e)
        ## Para dwpt de Magnus: Tdew(e,'magnus')
	if not magnus:
		e=esat_HylandAndWexler(Tdew)
		return e2RH(e,T,magnus)
	else:
		e=esat(Tdew)
		return e2RH(e,T,magnus)

def e2RH(e,T,*magnus):
        # Checked
        # PARTIAL PRESSURE (hPa) plus TEMPERATURE (K) TO RELATIVE HUMIDITY (1)
        if not magnus:
                RH=e/esat_HylandAndWexler(T)
        else:
                RH=e/esat(T)

        return RH

def esat_HylandAndWexler(t):
	# Hyland and Wexler 1983 water vapour saturation function
	# T in Kelvin
	# Output pressure in mb
	esat=np.exp(-0.58002206e4/t + 0.13914993e1 - 0.48640239e-1*t + 0.41764768e-4*t**2 - 0.14452093e-7*t**3 + 0.65459673e1*np.log(t))/100.0
	return esat

def esat(t):
	# Magnus water vapour saturation function
	# T in Kelvin
	# Output pressure in mb   
	esat=6.10*10**(7.4475*(t-273.15)/(234.07+t-273.15))
	return esat

def RH2q(RH,p,T,*magnus):
        # Checked
        # RELATIVE HUMIDITY (1) plus TEMPERATURE (K) plus PRESSURE (p) TO SPECIFIC HUMIDITY (kg/kg)
	e=RH2e(RH,T,magnus)
	q=e2q(e,p)
	return q

def RH2e(RH,T,*magnus):
        # Checked
        # RELATIVE HUMIDITY (1) plus TEMPERATUE (K) TO PARTIAL PRESSURE (hPa)
        if not magnus:
                e=RH*esat_HylandAndWexler(T)
        else:
                e=RH*esat(T)
        return e

def e2q(e,p):
        # Checked
        # PARTIAL PRESSURE (hPa) plus PRESSURE (hPa) TO SPECIFIC HUMIDITY (kg/kg)
	m=e2m(e,p)
	q=m2q(m)
	return q

def e2m(e,p):
        # Checked
        # PARTIAL PRESSURE WATER VAPOR (hPa) plus PRESSURE (hPa) TO MIXING RATIO (kg/kg)
	Mw=18.016
	Md=28.966
	m=(Mw*e)/(Md*(p-e))
	return m

def m2q(m):
        # Checked
        # MIXING RATIO (kg/kg) TO SPECIFIC HUMIDITY (kg/kg)
	q=m/(m+1.0)
	return q


