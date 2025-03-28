import numpy as np
import scipy.special
import metpy.calc as mpcalc
from metpy.units import units

def lcl_Romps(ds):
   """
   Calculate the LCL using the Romps method.

   Reference:
   Romps, D. M. (2017). Exact expression for the lifting condensation level. 
   Journal of the Atmospheric Sciences, 74(12), 3891-3900.
   https://journals.ametsoc.org/view/journals/atsc/74/12/jas-d-17-0102.1.xml

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using the Romps method.
   """
   pressure = ds.pres.values[0] * 100  # Convert hPa to Pa
   temperature = ds.tdry.values[0] + 273.15  # Convert Celsius to Kelvin
   rh = ds.rh.values[0] / 100  # Convert % to fraction

   # Parameters
   Ttrip = 273.16     # K
   ptrip = 611.65     # Pa
   E0v   = 2.3740e6   # J/kg
   E0s   = 0.3337e6   # J/kg
   ggr   = 9.81       # m/s^2
   rgasa = 287.04     # J/kg/K 
   rgasv = 461        # J/kg/K 
   cva   = 719        # J/kg/K
   cvv   = 1418       # J/kg/K 
   cvl   = 4119       # J/kg/K 
   cvs   = 1861       # J/kg/K 
   cpa   = cva + rgasa
   cpv   = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T/Ttrip)**((cpv-cvl)/rgasv) * \
         np.exp( (E0v - (cvv-cvl)*Ttrip) / rgasv * (1/Ttrip - 1/T) )
   
   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T/Ttrip)**((cpv-cvs)/rgasv) * \
         np.exp( (E0v + E0s - (cvv-cvs)*Ttrip) / rgasv * (1/Ttrip - 1/T) )

   # The variable rh is assumed to be
   # with respect to liquid if temperature > Ttrip and
   # with respect to solid if temperature < Ttrip
   if temperature > Ttrip:
      pv = rh * pvstarl(temperature)
   else:
      pv = rh * pvstars(temperature)

   if pv > pressure:
      ds["LCLP_Romps"] = np.nan
      ds["LCLT_Romps"] = np.nan
      ds["LCLZ_Romps"] = np.nan

      return ds

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa * pv / (rgasv * pressure + (rgasa - rgasv) * pv)
   rgasm = (1 - qv) * rgasa + qv * rgasv
   cpm = (1 - qv) * cpa + qv * cpv

   aL = -(cpv - cvl) / rgasv + cpm / rgasm
   bL = -(E0v - (cvv - cvl) * Ttrip) / (rgasv * temperature)
   cL = pv / pvstarl(temperature) * np.exp(-(E0v - (cvv - cvl) * Ttrip) / (rgasv * temperature))
   LCL_temp = temperature * (1 / (scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real)) * bL / aL
   LCL_height = cpm * temperature / ggr * (1 - \
                                 bL / (aL * scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real))
   
   LCL_pres = pressure * (LCL_temp / temperature) ** (cpm / rgasm)

   ds["LCLP_Romps"] = LCL_pres / 100  # Convert Pa to hPa
   ds["LCLT_Romps"] = LCL_temp - 273.15  # Convert Kelvin to Celsius
   ds["LCLZ_Romps"] = LCL_height

   return ds

def lcl_metpy(ds):
   """
   Calculate the LCL using the iterative approach from MetPy.

   The basic algorithm is:
   1. Find the dewpoint from the LCL pressure and starting mixing ratio.
   2. Find the LCL pressure from the starting temperature and dewpoint.
   3. Iterate until convergence.

   Reference:
   https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lcl.html#examples-using-metpy-calc-lcl

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using MetPy.
   """
   pressure = ds.pres.values * units.hPa
   temperature = ds.tdry.values * units.degC
   dewpoint = ds.dewpoint_Hardy.values * units.degC
   altitude = ds.alt.values * units.m

   try:
      LCL_pres, LCL_temp = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
      LCL_height = np.interp(LCL_pres, pressure[::-1], altitude[::-1])
   except:
      LCL_pres, LCL_temp, LCL_height = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

   ds["LCLP_MetPy"] = LCL_pres.magnitude
   ds["LCLT_MetPy"] = LCL_temp.magnitude
   ds["LCLZ_MetPy"] = LCL_height.magnitude
   return ds

def lcl_Bolton1980_surface(ds):
   """
   Calculate the LCL temperature using the Bolton (1980) method, then use Poisson's equation to find the pressure.
   Calculate the LCL from only the surface point values.

   Reference:
   Bolton, D. (1980). The computation of equivalent potential temperature. Monthly Weather Review, 108(7), 1046-1053.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using the Bolton (1980) method.
   """
   pressure = ds.pres.values
   temperature = ds.tdry.values + 273.15
   dewpoint = ds.dewpoint_Hardy.values + 273.15
   altitude = ds.alt.values

   kappa = 0.286  # Rd/Cp

   # Calculate LCL temperature using Bolton (1980)
   LCL_temp = 1 / ((1 / (dewpoint[0] - 56)) + (np.log(temperature[0] / dewpoint[0]) / 800)) + 56

   # Calculate LCL pressure using Poisson’s equation
   LCL_pres = pressure[0] * (LCL_temp / temperature[0]) ** (1 / kappa)

   # Calculate LCL height using interpolation
   LCL_height = np.interp(LCL_pres, pressure[::-1], altitude[::-1])

   ds["LCLP_Bolton1980_surface"] = LCL_pres
   ds["LCLT_Bolton1980_surface"] = LCL_temp - 273.15
   ds["LCLZ_Bolton1980_surface"] = LCL_height

   return ds

def lcl_Bolton1980_averaged(ds):
   """
   Calculate the LCL temperature using the Bolton (1980) method, then use Poisson's equation to find the pressure.
   Average the lowest 500m of temperature and dewpoint values. Used by the University of Wyoming.

   Reference:
   Bolton, D. (1980). The computation of equivalent potential temperature. Monthly Weather Review, 108(7), 1046-1053.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using the Bolton (1980) method.
   """
   pressure = ds.pres.values
   temperature = ds.tdry.values + 273.15
   dewpoint = ds.dewpoint_Hardy.values + 273.15
   altitude = ds.alt.values
   height_above_ground = altitude - altitude[0]

   # Filter the values for the first 0-500 meters
   mask = height_above_ground <= 500
   avg_temperature = np.nanmean(temperature[mask])
   avg_dewpoint = np.nanmean(dewpoint[mask])
   avg_pressure = np.nanmean(pressure[mask])

   kappa = 0.286  # Rd/Cp

   # Calculate LCL temperature using Bolton (1980)
   LCL_temp = 1 / ((1 / (avg_dewpoint - 56)) + (np.log(avg_temperature / avg_dewpoint) / 800)) + 56

   # Calculate LCL pressure using Poisson’s equation
   LCL_pres = avg_pressure * (LCL_temp / avg_temperature) ** (1 / kappa)

   # Calculate LCL height using interpolation
   LCL_height = np.interp(LCL_pres, pressure[::-1], altitude[::-1])

   ds["LCLP_Bolton1980_averaged"] = LCL_pres
   ds["LCLT_Bolton1980_averaged"] = LCL_temp - 273.15
   ds["LCLZ_Bolton1980_averaged"] = LCL_height

   return ds

def lcl_Petty2008(ds):
   """
   LCL estimations from Grant Petty's 'A First Course in Atmospheric Thermodynamics' (2008).

   Reference:
   Petty, G. W. (2008). A First Course in Atmospheric Thermodynamics. Sundog Publishing.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using Petty's method.
   """
   pressure = ds.pres.values
   temperature = ds.tdry.values
   dewpoint = ds.dewpoint_Hardy.values

   LCL_pres = pressure[0] * np.exp(-.044 * (temperature[0] - dewpoint[0]))
   LCL_height = (temperature[0] - dewpoint[0]) / 8  # Stull (2000)
   LCL_temp = np.interp(LCL_pres, pressure[::-1], temperature[::-1])

   ds["LCLP_Petty2008"] = LCL_pres
   ds["LCLZ_Petty2008"] = LCL_height * 1000
   ds["LCLT_Petty2008"] = LCL_temp

   return ds

def lcl_Sharppy(ds):
   """
   Calculate the LCL using the Sharppy method by applying an empirical polynomial fit for 
   LCL temperature, which resembles Espy's approximation with added correction terms.

   Reference:
   https://github.com/sharppy/SHARPpy

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using the Sharppy method.
   """
   pressure = ds.pres.values
   temperature = ds.tdry.values
   dewpoint = ds.dewpoint_Hardy.values
   altitude = ds.alt.values

   s = temperature[0] - dewpoint[0]
   dlt = s * (1.2185 + 0.001278 * temperature[0] + s * (-0.00219 + 1.173e-5 * s -
                                           0.0000052 * temperature[0]))
   LCL_temp = temperature[0] - dlt

   theta = ((temperature[0] + 273.15) * np.power((1000. / pressure[0]), 0.28571426)) - 273.15
   theta = theta + 273.15
   LCL_temp += 273.15
   LCL_pres = 1000. / (np.power((theta / LCL_temp), (1. / 0.28571426)))

   # Calculate LCL height using interpolation
   LCL_height = np.interp(LCL_pres, pressure[::-1], altitude[::-1])

   ds["LCLP_Sharppy"] = LCL_pres
   ds["LCLT_Sharppy"] = LCL_temp - 273.15
   ds["LCLZ_Sharppy"] = LCL_height

   return ds

def calculate_lcl(ds):
   """
   Calculate the LCL using multiple methods and add the results to the dataset.

   Parameters:
   ds : xarray.Dataset
      Input dataset containing pressure (hPa), temperature (Celsius), and relative humidity (percent).

   Returns:
   xarray.Dataset
      Dataset with added LCL height (meters), pressure (hPa), and temperature (Celsius) calculated using various methods.
   """
   ds = lcl_Romps(ds)
   ds = lcl_metpy(ds)
   ds = lcl_Bolton1980_surface(ds)
   ds = lcl_Bolton1980_averaged(ds)
   #ds = lcl_Petty2008(ds)
   ds = lcl_Sharppy(ds)

   return ds