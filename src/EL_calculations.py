import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def el_metpy(ds):
    pressure = ds.pres.values * units.hPa
    temperature = ds.tdry.values * units.degC
    dewpoint = ds.dewpoint_Hardy.values * units.degC
    altitude = ds.alt.values * units.m

    # Create parcel path using mpcalc, the pressure profile, and surface conditions
    try:
        parcel = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0]).to('degC')
    except:
        parcel = np.nan

    try:
        EL_pres_top, EL_temp_top = mpcalc.el(pressure, temperature, dewpoint, parcel, 'top')
        EL_height_top = np.interp(EL_pres_top, pressure[::-1], altitude[::-1])
    except:
        EL_pres_top, EL_temp_top, EL_height_top = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

    try:
        EL_pres_bottom, EL_temp_bottom = mpcalc.el(pressure, temperature, dewpoint, parcel, 'bottom')
        EL_height_bottom = np.interp(EL_pres_bottom, pressure[::-1], altitude[::-1])
    except:
        EL_pres_bottom, EL_temp_bottom, EL_height_bottom = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

    ds['ELP_MetPy_top'] = EL_pres_top.magnitude
    ds['ELT_MetPy_top'] = EL_temp_top.magnitude
    ds['ELZ_MetPy_top'] = EL_height_top.magnitude

    ds['ELP_MetPy_bottom'] = EL_pres_bottom.magnitude
    ds['ELT_MetPy_bottom'] = EL_temp_bottom.magnitude
    ds['ELZ_MetPy_bottom'] = EL_height_bottom.magnitude

    return ds

def el_metpy_averaged(ds):
    pressure = ds.pres.values * units.hPa
    temperature = ds.tdry.values * units.degC
    dewpoint = ds.dewpoint_Hardy.values * units.degC
    altitude = ds.alt.values * units.m

    # Create parcel path using mpcalc, the pressure profile, and surface conditions
    mask = altitude <= (altitude[0] + 500 * units.m)
    avg_temperature = np.mean(temperature[mask])
    avg_dewpoint = np.mean(dewpoint[mask])
    parcel = mpcalc.parcel_profile(pressure[mask], avg_temperature, avg_dewpoint).to('degC')

    EL_pres, EL_temp = mpcalc.el(pressure[mask], temperature[mask], dewpoint[mask], parcel)
    EL_height = np.interp(EL_pres, pressure[mask][::-1], altitude[mask][::-1])
    # except:
    #     EL_pres, EL_temp, EL_height = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

    ds['ELP_MetPy_averaged'] = EL_pres.magnitude
    ds['ELT_MetPy_averaged'] = EL_temp.magnitude
    ds['ELZ_MetPy_averaged'] = EL_height.magnitude

    return ds

def imitate_vaisala(ds):
    # Attempt to raise moist adiabatically from closest to Vaisala LFC
    
    lfcz = ds.LFCZ_surface.copy(deep=True)
    lfct = ds.LFCT_surface.copy(deep=True)

    # Step 4: Find EL - First height where parcel is colder than the environment
    parcel_T = lfct  # Start at LFC
    lfc_height = None

    for i in range(0, len(ds.alt.values)):
        height = ds.alt.values
        temperature = ds.tdry.values
        pressure = ds.pres.values

        ELs = []

        if height[i] >= lfcz:
            parcel_T -= (6.5 / 1000) * (height[i] - height[i - 1])  # Lift parcel
            
            if parcel_T < temperature[i]:  # EL is where parcel becomes colder
                lfc_height = height[i]
                lfc_temp = temperature[i]
                lfc_pressure = pressure[i]
                
                ELs.append(lfc_pressure)
                ds['ELZ_surface'] = lfc_height
                ds['ELT_surface'] = lfc_temp
                ds['ELP_surface'] = lfc_pressure

                break
        
        else:
            ds['ELZ_surface'] = np.nan
            ds['ELT_surface'] = np.nan
            ds['ELP_surface'] = ELs[-1] if ELs else np.nan

    return ds

def calculate_el(ds):
   """
   Calculate the EL using multiple methods and add the results to the dataset.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added EL height (meters), pressure (hPa), and temperature (Celsius) calculated using various methods.
   """
   ds = el_metpy(ds)
   ds = el_metpy_averaged(ds)
   ds = imitate_vaisala(ds)
   return ds