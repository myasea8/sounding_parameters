import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def lfc_metpy(ds):
    pressure = ds.pres.values * units.hPa
    temperature = ds.tdry.values * units.degC
    dewpoint = ds.dewpoint_Hardy.values * units.degC
    altitude = ds.alt.values * units.m

    try:
        LFC_pres, LFC_temp = mpcalc.lfc(pressure, temperature, dewpoint)
        LFC_height = np.interp(LFC_pres, pressure[::-1], altitude[::-1])
    except:
        LFC_pres, LFC_temp, LFC_height = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

    ds["LFCP_MetPy"] = LFC_pres.magnitude
    ds["LFCT_MetPy"] = LFC_temp.magnitude
    ds["LFCZ_MetPy"] = LFC_height.magnitude
    
    return ds
    
def lfc_Wyoming_surface(ds):
    
    lclz = ds.LCLZ_Bolton1980_surface.copy(deep=True)
    lclt = ds.LCLT_Bolton1980_surface.copy(deep=True)

    # Step 4: Find LFC - First height where parcel is warmer than the environment
    parcel_T = lclt  # Start at LCL
    lfc_height = None

    for i in range(0, len(ds.alt.values)):
        height = ds.alt.values
        temperature = ds.tdry.values
        pressure = ds.pres.values

        if height[i] >= lclz:
            parcel_T -= (6.5 / 1000) * (height[i] - height[i - 1])  # Lift parcel
            
            if parcel_T > temperature[i]:  # LFC is where parcel becomes warmer
                lfc_height = height[i]
                lfc_temp = temperature[i]
                lfc_pressure = pressure[i]
                
                ds['LFCZ_surface'] = lfc_height
                ds['LFCT_surface'] = lfc_temp
                ds['LFCP_surface'] = lfc_pressure

                break
        
        else:
            ds['LFCZ_surface'] = np.nan
            ds['LFCT_surface'] = np.nan
            ds['LFCP_surface'] = np.nan

    return ds

def lfc_Wyoming_averaged(ds):

    lclz = ds.LCLZ_Bolton1980_averaged.copy(deep=True)
    lclt = ds.LCLT_Bolton1980_averaged.copy(deep=True)

    # Step 4: Find LFC - First height where parcel is warmer than the environment
    parcel_T = lclt  # Start at LCL
    lfc_height = None

    for i in range(0, len(ds.alt.values)):
        height = ds.alt.values
        temperature = ds.tdry.values
        pressure = ds.pres.values

        if height[i] >= lclz:
            parcel_T -= (6.5 / 1000) * (height[i] - height[i - 1])  # Lift parcel
            
            if parcel_T > temperature[i]:  # LFC is where parcel becomes warmer
                lfc_height = height[i]
                lfc_temp = temperature[i]
                lfc_pressure = pressure[i]
                
                ds['LFCZ_averaged'] = lfc_height
                ds['LFCT_averaged'] = lfc_temp
                ds['LFCP_averaged'] = lfc_pressure

                break
        else:
            ds['LFCZ_averaged'] = np.nan
            ds['LFCT_averaged'] = np.nan
            ds['LFCP_averaged'] = np.nan

    return ds

def calculate_lfc(ds):
   """
   Calculate the LFC using multiple methods and add the results to the dataset.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LFC height (meters), pressure (hPa), and temperature (Celsius) calculated using various methods.
   """
   ds = lfc_metpy(ds)
   ds = lfc_Wyoming_averaged(ds)
   ds = lfc_Wyoming_surface(ds)
   
   return ds