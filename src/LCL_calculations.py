import numpy as np
import scipy.special
import metpy.calc as mpcalc
from metpy.units import units

def lcl_Romps(ds):
   '''Calculate the LCL using the Romps method.
      https://journals.ametsoc.org/view/journals/atsc/74/12/jas-d-17-0102.1.xml
   '''

   p = ds.pres.values[0] * 100  # Convert hPa to Pa
   T = ds.tdry.values[0] + 273.15  # Convert Celsius to Kelvin
   rh = ds.rh.values[0] / 100  # Convert % to fraction

   # Parameters
   Ttrip = 273.16  # K
   ptrip = 611.65  # Pa
   E0v = 2.3740e6  # J/kg
   E0s = 0.3337e6  # J/kg
   ggr = 9.81  # m/s^2
   rgasa = 287.04  # J/kg/K
   rgasv = 461  # J/kg/K
   cva = 719  # J/kg/K
   cvv = 1418  # J/kg/K
   cvl = 4119  # J/kg/K
   cvs = 1861  # J/kg/K
   cpa = cva + rgasa
   cpv = cvv + rgasv

   # The saturation vapor pressure over liquid water
   def pvstarl(T):
      return ptrip * (T / Ttrip) ** ((cpv - cvl) / rgasv) * \
            np.exp((E0v - (cvv - cvl) * Ttrip) / rgasv * (1 / Ttrip - 1 / T))

   # The saturation vapor pressure over solid ice
   def pvstars(T):
      return ptrip * (T / Ttrip) ** ((cpv - cvs) / rgasv) * \
            np.exp((E0v + E0s - (cvv - cvs) * Ttrip) / rgasv * (1 / Ttrip - 1 / T))

   # The variable rh is assumed to be
   # with respect to liquid if T > Ttrip and
   # with respect to solid if T < Ttrip
   if T > Ttrip:
      pv = rh * pvstarl(T)
   else:
      pv = rh * pvstars(T)

   if pv > p:
      ds["LCL_height_Romps"] = LCL_height

   # Calculate lcl_liquid and lcl_solid
   qv = rgasa * pv / (rgasv * p + (rgasa - rgasv) * pv)
   rgasm = (1 - qv) * rgasa + qv * rgasv
   cpm = (1 - qv) * cpa + qv * cpv
   if rh == 0:
      ds["LCL_height_Romps"] = LCL_height
   aL = -(cpv - cvl) / rgasv + cpm / rgasm
   bL = -(E0v - (cvv - cvl) * Ttrip) / (rgasv * T)
   cL = pv / pvstarl(T) * np.exp(-(E0v - (cvv - cvl) * Ttrip) / (rgasv * T))
   LCL_temp = T * (1 / (scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real)) * bL / aL
   LCL_height = cpm * T / ggr * (1 - \
                          bL / (aL * scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real))
   LCL_pres = p * (LCL_temp / T) ** (cpm/rgasm)

   ds["LCLP_Romps"] = LCL_pres / 100
   ds["LCLT_Romps"] = LCL_temp - 273.15
   ds["LCLZ_Romps"] = LCL_height

   return ds

def lcl_metpy(ds):
   '''Calculate the LCL using the iterative approach, beginning with starting pressure, temperature and dewpoint.
   The basic algorithm is:
    1. Find the dewpoint from the LCL pressure and starting mixing ratio
    2. Find the LCL pressure from the starting temperature and dewpoint
    3. Iterate until convergence

   https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.lcl.html#examples-using-metpy-calc-lcl 
   '''
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
   """Calculate the LCL temperature using the Bolton (1980) method, then using Poisson's equation to find the pressure.
   Calculate the LCL from only the surface point values."""
   
   pressure = ds.pres.values
   temperature = ds.tdry.values + 273.15
   dewpoint = ds.dewpoint_Hardy.values + 273.15
   altitude = ds.alt.values

   kappa = 0.286  # Rd/Cp

   # Calculate LCL temperature using Bolton (1980)
   t_lcl = 1 / ((1 / (dewpoint[0] - 56)) + (np.log(temperature[0] / dewpoint[0]) / 800)) + 56

   # Calculate LCL pressure using Poisson’s equation
   p_lcl = pressure[0] * (t_lcl / temperature[0]) ** (1 / kappa)

   # Calculate LCL height using interpolation
   z_lcl = np.interp(p_lcl, pressure[::-1], altitude[::-1])

   ds["LCLP_Bolton1980_surface"] = p_lcl
   ds["LCLT_Bolton1980_surface"] = t_lcl - 273.15
   ds["LCLZ_Bolton1980_surface"] = z_lcl

   return ds

def lcl_Bolton1980_averaged(ds):
   """Calculate the LCL temperature using the Bolton (1980) method, then using Poisson's equation to find the pressure.
   Average the lowest 500m of temperature and dewpoint values. 
   Used by the University of Wyoming."""
   
   pressure = ds.pres.values
   temperature = ds.tdry.values + 273.15
   dewpoint = ds.dewpoint_Hardy.values + 273.15
   altitude = ds.alt.values

   # Filter the values for the first 0-500 meters
   mask = altitude <= 500
   avg_temperature = np.nanmean(temperature[mask])
   avg_dewpoint = np.nanmean(dewpoint[mask])
   avg_pressure = np.nanmean(pressure[mask])

   kappa = 0.286  # Rd/Cp

   # Calculate LCL temperature using Bolton (1980)
   t_lcl = 1 / ((1 / (avg_dewpoint - 56)) + (np.log(avg_temperature / avg_dewpoint) / 800)) + 56

   # Calculate LCL pressure using Poisson’s equation
   p_lcl = avg_pressure * (t_lcl / avg_temperature) ** (1 / kappa)

   # Calculate LCL height using interpolation
   z_lcl = np.interp(p_lcl, pressure[::-1], altitude[::-1])

   ds["LCLP_Bolton1980_averaged"] = p_lcl
   ds["LCLT_Bolton1980_averaged"] = t_lcl - 273.15
   ds["LCLZ_Bolton1980_averaged"] = z_lcl

   return ds

def lcl_Petty2008(ds):
   """LCL estimations from Grant Petty's 'A First Course in Atmospheric Thermodynamics' (2008)."""
   pressure = ds.pres.values
   temperature = ds.tdry.values
   dewpoint = ds.dewpoint_Hardy.values
   altitude = ds.alt.values

   LCL_pressure = pressure[0] * np.exp(-.044 * (temperature[0] - dewpoint[0]))
   LCL_height = (temperature[0] - dewpoint[0]) / 8
   LCL_temperature = np.interp(LCL_pressure, pressure[::-1], temperature[::-1])

   ds["LCLP_Petty2008"] = LCL_pressure
   ds["LCLZ_Petty2008"] = LCL_height * 1000
   ds["LCLT_Petty2008"] = LCL_temperature

   return ds

def calculate_lcl(ds):
   ds = lcl_Romps(ds)
   ds = lcl_metpy(ds)
   ds = lcl_Bolton1980_surface(ds)
   ds = lcl_Bolton1980_averaged(ds)
   ds = lcl_Petty2008(ds)

   return ds