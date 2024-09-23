#! /DSNNAS/Repro_Temp/users/jacksont/miniconda3/envs/AVHRR_FCDR/bin/python
# Written by: Thomas Jackson 
# Organisation: EUMETSAT
# Purpose:  This extracts macropixels of maximum entropy and performs some statistical analysis to 
#           store along with the tile in the database.  This can then be used for QC checking of 
#           outputs (was originally designed for use with AVHRR due to issues with NOAA 15 data.)
# Date: August 2023
# Notes: 
# 
#  
import argparse
import skimage
from skimage.util import view_as_blocks
import numpy as np
import pandas as pd
import time
import os
import netCDF4 as nc
import datetime
import glob
import xarray as xr

def numpy_to_datetime64(numpy_dt):
    """Convert numpy.datetime64 to Python datetime."""
    return pd.to_datetime(numpy_dt).to_pydatetime()


def calculate_entropy(tile):
    """Calculate entropy of a tile."""
    hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def analyze_tiles(image, tile_size):
    """Analyze tiles of the given image."""
    # Determine the shape of the tiles
    tiles_shape = (image.shape[0] // tile_size, image.shape[1] // tile_size)
    
    # Create a view of the image as blocks
    tiles = view_as_blocks(image, block_shape=(tile_size, tile_size))
    
    # Initialize variables to track the maximum entropy and its position
    max_entropy = -np.inf
    max_entropy_tile = None
    max_entropy_position = None
    
    # Iterate over all tiles
    for i in range(tiles_shape[0]):
        for j in range(tiles_shape[1]):
            tile = tiles[i, j]
            entropy = calculate_entropy(tile)
            
            if entropy > max_entropy:
                max_entropy = entropy
                max_entropy_tile = tile
                max_entropy_position = (i * tile_size, j * tile_size)
    
    return max_entropy, max_entropy_tile, max_entropy_position

def analyze_tiles_with_edges(image, tile_size):
    """Analyze tiles of the given image, including edge cases."""
    height, width = image.shape
    max_entropy = -np.inf
    max_entropy_tile = None
    max_entropy_position = None
    
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = image[i:i+tile_size, j:j+tile_size]
            
            # Check the percentage of non-NaN values
            non_nan_percentage = np.sum(~np.isnan(tile)) / tile.size
            
            if non_nan_percentage > 0.75:
                # Calculate the entropy for this tile
                # Handle tiles that are smaller than the desired size
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    tile = np.pad(tile, ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1])), mode='constant')

                entropy = calculate_entropy(tile)

                if entropy > max_entropy:
                    max_entropy = entropy
                    max_entropy_tile = tile
                    max_entropy_position = (i, j)            

    
    return max_entropy, max_entropy_tile, max_entropy_position


def main():

    arg_parse=argparse.ArgumentParser(description="Give this script a directory, \
    a filestring pattern and an output filename and it will create an output dataset of QC checks")
    arg_parse.add_argument('--input_dir', '-i', help='input directory e.g \
        /DSNNAS2/Repro/processing/pygac/release1_extension_2024/output_2024_extension_products/N15/2023', required=True)
    arg_parse.add_argument('--file_regex', '-f',required=True, help='input file regex to match')
    arg_parse.add_argument('--output_file', '-o',required=True, help='output file to write the results to THIS SHOULD BE A NETCDF')
    arg_parse.add_argument('--tile_size','-t',  type=int, help='size to slice data into for checks (integer)', default=50)
    arg_parse.add_argument('--analysis_band','-b',  default='reflectance_channel_1', help='Band name that we are extracting and analysing')

    args=arg_parse.parse_args()

    output_file=args.output_file
    if os.path.basename(output_file).endswith('.nc'):
        print(f'Will write out to {output_file}')
    else:
        raise ValueError('input filename for ouput file is not a netcdf (needs to end with .nc)')

    tile_size=args.tile_size

    if os.path.exists(output_file):
        dataset=nc.Dataset(output_file, 'a', format='NETCDF4')
        #check compatability of key metadata
        if not dataset.tile_size==tile_size:
            raise AssertionError('You are trying to append tiles to a file that already has different size tiles')
        if not np.all([x in dataset.variables.keys() for x in ['time', 'x', 'y', 'entropy', args.analysis_band, 'noise_sigma']]):
            print('vars FOUND:', dataset.variables.keys() )
            print('vars EXPECTED: ', ['time', 'x', 'y', 'entropy', args.analysis_band, 'noise_sigma'])
            raise AssertionError('File to be appended to doesnt seem to have correct variables')  
        time_idx=len(dataset.variables['time'])
        times = dataset.variables['time']
        filenames = dataset.variables['filename']
        x_loc = dataset.variables['x']
        y_loc = dataset.variables['y'] 
        entropy_values = dataset.variables['entropy']
        arrays = dataset.variables[args.analysis_band]
        noise_sigma_values = dataset.variables['noise_sigma']
    else:
        dataset=nc.Dataset(output_file, 'w', format='NETCDF4')
        # Define dimensions
        time_dim = dataset.createDimension('time', None)  # Unlimited time dimension
        x_dim = dataset.createDimension('x', 1)  # x location dimension
        y_dim = dataset.createDimension('y', 1)  # y location dimension
        row_dim = dataset.createDimension('row', tile_size)  # Row dimension for the array
        col_dim = dataset.createDimension('col', tile_size)  # Column dimension for the array   
        # Define variables
        times = dataset.createVariable('time', np.float64, ('time',))
        filenames = dataset.createVariable('filename', str, ('time',))
        x_loc = dataset.createVariable('x', np.int32, ('time', 'x'))
        y_loc = dataset.createVariable('y', np.int32, ('time', 'y'))
        entropy_values = dataset.createVariable('entropy', np.float64, ('time',))
        arrays = dataset.createVariable(args.analysis_band, np.float64, ('time', 'row', 'col'))
        noise_sigma_values = dataset.createVariable('noise_sigma', np.float64, ('time',))
        #Set metadata
        dataset.description = 'A netcdf of extracted stats from some AVHRR files'
        dataset.history = f'Created on {str(datetime.datetime.now())}'
        # Add metadata to variables
        times.units = 'days since 1970-01-01'
        times.standard_name = 'time'
        dataset.tile_size=tile_size    
        time_idx=0
    
    Indir=args.input_dir
    file_glob=args.file_regex
    Matched_files = list(glob.glob(Indir+'/'+file_glob))
    Matched_files.sort()
    if len(Matched_files)>0:
        n_files=len(Matched_files)
        print(f'Found {n_files} matching supplied regex {Indir}/{file_glob}')
    else:
        print(f'No files found using regex: {Indir}/{file_glob}')
    print(f'Adding files from index {time_idx} onwards')
    for current_file in Matched_files:
        if os.path.basename(current_file) in dataset.variables['filename']:
            print(f'{current_file} already processed so skipping')
        else:
            print(f'Processing: {current_file}')
            dat = xr.open_dataset(current_file)
            midish_row = int(len(dat['acq_time'].values) / 2)
            # Convert numpy.datetime64 to Python datetime
            file_mid_time_np = dat['acq_time'][midish_row].values
            file_mid_time = numpy_to_datetime64(file_mid_time_np)
            # Convert Python datetime to seconds since epoch
            file_mid_time_num = (file_mid_time - datetime.datetime(1970, 1, 1)).total_seconds()
            max_entropy, max_entropy_tile, position = analyze_tiles_with_edges(dat[args.analysis_band].values, tile_size)
            if max_entropy_tile is None:
                print(f'No valid data in file {current_file}')  
            else:
                # Assign data to NetCDF variables
                times[time_idx] = file_mid_time_num
                filenames[time_idx] = os.path.basename(current_file)
                x_loc[time_idx, 0] = position[0]
                y_loc[time_idx, 0] = position[1]
                entropy_values[time_idx] = max_entropy
                arrays[time_idx, :, :] = max_entropy_tile
                noise_sigma_values[time_idx]=skimage.restoration.estimate_sigma(np.nan_to_num(max_entropy_tile, nan=np.nanmean(max_entropy_tile)))
                time_idx += 1                          
            dat.close()
    dataset.close()



if __name__=="__main__":
    main()

