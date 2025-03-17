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
   LCL_pres = np.nan
   LCL_temp = T * (1 / (scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real)) * bL / aL
   LCL_height = cpm * T / ggr * (1 - \
                          bL / (aL * scipy.special.lambertw(bL / aL * cL ** (1 / aL), -1).real))
   
   ds["LCL_pressure_Romps"] = LCL_pres
   ds["LCL_temperature_Romps"] = LCL_temp - 273.15
   ds["LCL_height_Romps"] = LCL_height
   return ds

def lcl_metpy(ds):
   pressure = ds.pres.values * units.hPa
   temperature = ds.tdry.values * units.degC
   dewpoint = ds.dewpoint_Hardy.values * units.degC
   altitude = ds.alt.values * units.m

   try:
      LCL_pres, LCL_temp = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
      LCL_height = np.interp(LCL_pres, pressure[::-1], altitude[::-1])
   except:
      LCL_pres, LCL_temp, LCL_height = np.nan * units.hPa, np.nan * units.degC, np.nan * units.m

   ds["LCL_pressure_MetPy"] = LCL_pres.magnitude
   ds["LCL_temperature_MetPy"] = LCL_temp.magnitude
   ds["LCL_height_MetPy"] = LCL_height.magnitude
   return ds

def calculate_lcl(ds):
   ds = lcl_Romps(ds)
   ds = lcl_metpy(ds)

   return ds