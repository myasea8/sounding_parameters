import numpy as np

def calculate_dewpoint(temperature, relative_humidity):
    """
    Calculate the dewpoint temperature given the air temperature and relative humidity.

    Parameters:
    temperature (float or np.ndarray): Air temperature in degrees Celsius.
    relative_humidity (float or np.ndarray): Relative humidity in percentage.

    Returns:
    float or np.ndarray: Dewpoint temperature in degrees Celsius.

    Notes:
    Calculations are based on Hardy 1998 approximation for the new ITS-90 scale.
    Reference: https://cires1.colorado.edu/~voemel/vp.html
    """

    # Convert temperature to Kelvin
    temperature_k = temperature + 273.15
    
    # Define the coefficients
    g = np.array([-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2, 
                  1.6261698e-5, 7.0229056e-10, -1.8680009e-13, 2.7150305])
    
    # Calculate es for each temperature
    es_value = np.exp(sum(g[i] * temperature_k**(i-2) for i in range(7)) + g[7] * np.log(temperature_k))
    
    # Calculate e for each relative humidity
    e_value = (relative_humidity / 100) * es_value
    
    # Calculate dewpoint for each e_value
    dewpoint = ((1/273.15) - (1.844e-4 * np.log(e_value/611.3)))**(-1) - 273.15  # Convert back to Celsius
    
    return dewpoint

def preprocess_dataset(profile_dataset):
    # Drop all times with nan pressure values
    profile_dataset = profile_dataset.dropna(dim='time', subset=['pres'])

    # Make sure that at least 75% of the pressure values are in decreasing order
    if np.sum(np.diff(profile_dataset.pres.values) > 0) / len(profile_dataset.pres.values) > 0.75:
        # Invert the dataset along the time dimension
        profile_dataset = profile_dataset.isel(time=slice(None, None, -1))

    # Calculate dewpoint and add it to the dataset
    dewpoint = calculate_dewpoint(profile_dataset.tdry.values, profile_dataset.rh.values)
    profile_dataset['dewpoint_Hardy'] = (('time',), dewpoint)

    return profile_dataset
