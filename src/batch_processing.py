import xarray as xr
from pathlib import Path
import argparse
from src.preprocessing import preprocess_dataset
from src.LCL_calculations import calculate_lcl
# from src.visualization import visualize_comparisons

def list_netCDF_files(directory):
    """
    List all NetCDF files in a given directory.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing NetCDF files.

    Returns:
    --------
    list of Path objects representing NetCDF files.
    """
    directory = Path(directory)
    return sorted([f for f in directory.iterdir() if f.suffix == ".nc"])

def process_field_campaign(data_path, limit):
    """
    Reads, preprocesses, and processes all NetCDF files in a field campaign.

    Parameters:
    -----------
    data_path : str
        Path to the directory containing NetCDF files.
    limit : int, optional
        Number of files to process (useful for testing).

    Returns:
    --------
    List of processed datasets or results.
    """
    files = list_netCDF_files(data_path)
    if limit:
        files = files[:limit]  # Process only a subset if specified

    results = []

    for file in files:
        print(f"Processing: {file}")

        # 1. Load dataset
        ds = xr.open_dataset(file)

        # 2. Preprocess
        ds = preprocess_dataset(ds)

        # 3. Calculate all parameters 
        ds = calculate_lcl(ds)

        # 4. Store results
        results.append(ds)

        ds.close()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NetCDF files from a field campaign.")
    parser.add_argument("data_path", type=str, help="Path to the directory containing NetCDF files.")
    parser.add_argument("--limit", type=int, default=5, help="Number of files to process (useful for testing).")

    args = parser.parse_args()

    process_field_campaign(args.data_path, limit=args.limit)
