import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def ccl_metpy(ds):
    pressure = ds.pres.values * units.hPa
    temperature = ds.tdry.values * units.degC
    dewpoint = ds.dewpoint_Hardy.values * units.degC
    altitude = ds.alt.values * units.m

    # Metpy finds the ccl where the ambient temperature profile intersects the line of constant mixing ratio starting at the surface,
    # using the surface dewpoint or the average dewpoint of a shallow layer near the surface
    CCL_pres, CCL_temp, _ = mpcalc.ccl(pressure, temperature, dewpoint)
    CCL_height = np.interp(CCL_pres, pressure[::-1], altitude[::-1])

    ds['CCLP_MetPy'] = CCL_pres.magnitude
    ds['CCLT_MetPy'] = CCL_temp.magnitude
    ds['CCLZ_MetPy'] = CCL_height.magnitude

    return ds

def ccl_metpy_layer(ds):
    pressure = ds.pres.values * units.hPa
    temperature = ds.tdry.values * units.degC
    dewpoint = ds.dewpoint_Hardy.values * units.degC
    altitude = ds.alt.values * units.m

    # Metpy finds the ccl where the ambient temperature profile intersects the line of constant mixing ratio starting at the surface,
    # using the surface dewpoint or the average dewpoint of a shallow layer near the surface
    CCL_pres, CCL_temp, _ = mpcalc.ccl(pressure, temperature, dewpoint, altitude, 150 * units.m)
    CCL_height = np.interp(CCL_pres, pressure[::-1], altitude[::-1])

    ds['CCLP_average_MetPy'] = CCL_pres.magnitude
    ds['CCLT_average_MetPy'] = CCL_temp.magnitude
    ds['CCLZ_average_MetPy'] = CCL_height.magnitude

    return ds

def ccl_assumption(ds):
    pressure = ds.pres.values
    temperature = ds.tdry.values
    dewpoint = ds.dewpoint_Hardy.values
    altitude = ds.alt.values
    
    CCL_height = 125 * (temperature[0] - dewpoint[0])
    CCL_temp = np.interp(CCL_height, altitude[::-1], temperature[::-1])
    CCL_pres = np.interp(CCL_height, altitude[::-1], pressure[::-1])

    ds['CCLZ_assumption'] = CCL_height
    ds['CCLT_assumption'] = CCL_temp
    ds['CCLP_assumption'] = CCL_pres

    return ds

def calculate_ccl(ds):
   """
   Calculate the CCL using multiple methods and add the results to the dataset.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added CCL height (meters), pressure (hPa), and temperature (Celsius) calculated using various methods.
   """
   ds = ccl_metpy(ds)
   ds = ccl_metpy_layer(ds)
   ds = ccl_assumption(ds)
   
   return ds