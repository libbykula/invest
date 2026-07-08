import numpy as np
import tempfile
import os
import pygeoprocessing
import shutil
import logging
from osgeo import gdal
import geopandas as gpd
import re
import pygeoprocessing.kernels


intermediate_dir = 'C:/Users/kibby/OneDrive/Desktop/Research/DOI/Code/cooling workspace/UrbanWatch/Minneapolis-St. Paul-Bloomington, MN-WI_upd_invest_v02/intermediate'

data_dir = 'C:/Users/kibby/OneDrive/Desktop/Research/DOI/Data'

# 1. Run model once to get baseline - done!

# 2. Evaluate dt_nomix_d_shade at every pixel → raster

# 3. Convolve once → dt_air_mix_d_shade raster (spatial sensitivity map)

# 4. Rank pixels by their value in that raster

def convolve_2d_by_exponential(
        decay_kernel_distance, signal_raster_path,
        target_convolve_raster_path):
    """Convolve signal by an exponential decay of a given radius.

    Args:
        decay_kernel_distance (float): radius of 1/e cutoff of decay kernel
            raster in pixels.
        signal_raster_path (str): path to single band signal raster.
        target_convolve_raster_path (str): path to convolved raster.

    Returns:
        None.

    """
    print(f"Starting a convolution over {signal_raster_path} with a "
                f"decay distance of {decay_kernel_distance}")
    temporary_working_dir = tempfile.mkdtemp(
        dir=os.path.dirname(target_convolve_raster_path))
    exponential_kernel_path = os.path.join(
        temporary_working_dir, 'exponential_decay_kernel.tif')
    pygeoprocessing.kernels.exponential_decay_kernel(
        target_kernel_path=exponential_kernel_path,
        max_distance=decay_kernel_distance * 5,
        expected_distance=decay_kernel_distance)
    pygeoprocessing.convolve_2d(
        (signal_raster_path, 1), (exponential_kernel_path, 1),
        target_convolve_raster_path, working_dir=temporary_working_dir,
        ignore_nodata_and_edges=True)
    shutil.rmtree(temporary_working_dir)

def get_dt_air_mix_d_shade(dt_nomix_d_shade, i, grid_shape, pixel_size, 
                            projection_wkt, d_air_mix_pixels,
                            target_raster_path):
    """
    Args:
        dt_nomix_d_shade (float): evaluated scalar derivative from MATLAB
        i (tuple): (row, col) of the pixel of interest
        grid_shape (tuple): (rows, cols) of the full grid
        pixel_size (tuple): (pixel_width, pixel_height) in map units
        projection_wkt (str): projection of the raster
        d_air_mix_pixels (float): decay distance in pixels
        target_raster_path (str): path for output derivative raster
    """
    tmpdir = tempfile.mkdtemp()
    spike_path = os.path.join(tmpdir, 'spike.tif')

    # Write spike raster: scalar derivative at pixel i, 0 elsewhere
    spike = np.zeros(grid_shape)
    spike[i[0], i[1]] = dt_nomix_d_shade

    pygeoprocessing.numpy_array_to_raster(
        spike, None, pixel_size, 
        (0, 0),  # origin — match your actual raster origin
        projection_wkt, spike_path)

    convolve_2d_by_exponential(
        decay_kernel_distance=d_air_mix_pixels,
        signal_raster_path=spike_path,
        target_convolve_raster_path=target_raster_path)

    shutil.rmtree(tmpdir)

# UHI_max is a constant for Minneapolis, what is it?
# Minneapolis-St. Paul-Bloomington, MN-WI	6	Afton city27	19.93050003	0.826599121
# Minneapolis-St. Paul-Bloomington, MN-WI	7	Afton city27	22.44921303	0.90438652
# Minneapolis-St. Paul-Bloomington, MN-WI	8	Afton city27	21.17599678	0.768802643

uhi_max_dict = {'06': 0.826599121,
                '07': 0.90438652,
                '08': 0.768802643}
month = '06'
uhi_max = uhi_max_dict[month]

def get_dt_effect(GA_i):
    dt_effect = -uhi_max*(3/(5*(np.exp(15/2 - (3*GA_i)/2) + 1)) + 1/5) # So I want to apply this to each pixel of green_area! 
    return dt_effect

def get_tif_max(filename):
    # Open the dataset
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Could not open file: {filename}")

    # Get the first raster band (index starts at 1)
    band = dataset.GetRasterBand(1)
    if band is None:
        raise ValueError("No raster band found in the file.")

    # Option 1: Use GDAL's built-in max
    max_val = band.GetMaximum()

    return max_val

# ref_et_raster_path = os.path.join(intermediate_dir, 'ref_eto_upd_invest_v02.tif')
ref_et_raster_path = os.path.join(data_dir, 'Evapotranspiration/et0_V3_' + month + '_aligned_wgs84_clipped_Minneapolis-St. Paul-Bloomington, MN-WI_linear.tif')

# et_max = get_tif_max(ref_et_raster_path)

# def get_dt_effect_kc(GA_i, et0_i):
#    dt_effect_kc = -(et0_i*uhi_max*(3/(np.exp(15/2 - (3*GA_i)/2) + 1) + 1))/(5*et_max) # So I want to apply this to each pixel of green_area! 
#    return dt_effect_kc

# do I need raster_calculator???    
ga_raster = os.path.join(intermediate_dir, 'green_area_sum_upd_invest_v02.tif')
dt_effect_unweighted_path = os.path.join(intermediate_dir, 'dt_effect_unweighted.tif')
dt_effect_kc_path = os.path.join(intermediate_dir, 'dt_effect_kc.tif')

# # for just dt
# pygeoprocessing.raster_calculator(base_raster_path_band_const_list = [(ga_raster, 1)],
#                                   local_op = get_dt_effect, target_raster_path = dt_effect_unweighted_path,
#                                   datatype_target = gdal.GDT_Float32, nodata_target = 9999)

# # for dt from kc, hmmmmm the maximum is giving infinity???? wtf!! 
# pygeoprocessing.raster_calculator(base_raster_path_band_const_list = [(ga_raster, 1),
#                                                                       (ref_et_raster_path, 1)],
#                                   local_op = get_dt_effect_kc, target_raster_path = dt_effect_kc_path,
#                                   datatype_target = gdal.GDT_Float32, nodata_target = 9999)


# Something weird going on with evapotranspiration... need to rerun InVEST, but in the meantime just clip the results here to be less weird

boundary = gpd.read_file(os.path.join(data_dir, 'Polygon_vectors/MINN_03.gpkg'))
boundary_buffed = boundary.buffer(-28)
buff_path = os.path.join(data_dir, 'Polygon_vectors/MINN_03_buffed.gpkg')
boundary_buffed.to_file(buff_path)

gdal.Warp(re.sub('.tif', '_buff.tif', dt_effect_kc_path), 
                dt_effect_kc_path, cutlineDSName= buff_path, 
                cropToCutline=True,
                            format="GTiff",  # Output format
                creationOptions=[
                    f"COMPRESS=LZW",  # Compression type
                    "TILED=YES",                # Enable tiling for better performance
                    "BIGTIFF=IF_SAFER"          # Allow BigTIFF if needed
                ])

# dt_nomix_d_shade  = -uhio_max*(9/(5*(exp(15/2 - (3*GA_i)/2) + 1)) + 3/5)
# dt_nomix_d_albedo = -uhi_max*(3/(5*(exp(15/2 - (3*GA_i)/2) + 1)) + 1/5)
# dt_nomix_d_eti = -uhi_max*(3/(5*(exp(15/2 - (3*GA_i)/2) + 1)) + 1/5)

# dt_nomix_d_kc = -(et0_i*uhi_max*(3/(exp(15/2 - (3*GA_i)/2) + 1) + 1))/(5*et_max) 
# where do I get et0 and et_max? 𝐸⁢𝑇⁢𝑚⁢𝑎⁢𝑥 = the maximum value of the 𝐸⁢𝑇⁢𝑜 raster in the area of interest. So I want ET0_i and then calc the max of it for the area. That's not that bad! 

# Ok, I need GA_i. What is that in my thing? Is it green_area_sum.tif or green_area.tif? i think the first one???

# uhi_max = scalar, what is it???
# eti_upd_invest_v02.tif et0 is ref_eto_upd_invest_v02.tif??? what is et_max??? oooh, I think with eti and can skip the initial inputs of et_max and et0 so it should just be same as albedo 
# % really the dt_air_mix_d_shade = 3*dt_air_mix_d_eti =
# % 3*dt_air_mix_d_albedo! 

# % So, I just need one equation, and then can do that scalar work later on!
# now I need to convolve 
convolve_2d_by_exponential(
        decay_kernel_distance = 600, signal_raster_path = dt_effect_unweighted_path,
        target_convolve_raster_path = re.sub('.tif', '_convolved.tif', dt_effect_unweighted_path)) # hmmmmmm..... why isn't pygeoprocessing working?!?!?!?!?!?!?

convolve_2d_by_exponential(
        decay_kernel_distance = 600, signal_raster_path = re.sub('.tif', '_buff.tif', dt_effect_kc_path),
        target_convolve_raster_path = re.sub('.tif', '_convolved.tif', re.sub('.tif', '_buff.tif', dt_effect_kc_path)))