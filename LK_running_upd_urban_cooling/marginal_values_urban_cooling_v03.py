import numpy as np
import tempfile
import os
import pygeoprocessing
import shutil
import logging
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import re
import pygeoprocessing.kernels
gdal.UseExceptions()

output_dir = 'C:/Users/kibby/OneDrive/Desktop/Research/DOI/Code/cooling workspace/UrbanWatch/Minneapolis-St. Paul-Bloomington, MN-WI_upd_invest_v02'
intermediate_dir = os.path.join(output_dir, 'intermediate')

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

# boundary = gpd.read_file(os.path.join(data_dir, 'Polygon_vectors/MINN_03.gpkg'))
# boundary_buffed = boundary.buffer(-28)
# buff_path = os.path.join(data_dir, 'Polygon_vectors/MINN_03_buffed.gpkg')
# boundary_buffed.to_file(buff_path)

# gdal.Warp(re.sub('.tif', '_buff.tif', dt_effect_kc_path), 
#                 dt_effect_kc_path, cutlineDSName= buff_path, 
#                 cropToCutline=True,
#                             format="GTiff",  # Output format
#                 creationOptions=[
#                     f"COMPRESS=LZW",  # Compression type
#                     "TILED=YES",                # Enable tiling for better performance
#                     "BIGTIFF=IF_SAFER"          # Allow BigTIFF if needed
#                 ])

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
# convolve_2d_by_exponential(
#         decay_kernel_distance = 600, signal_raster_path = dt_effect_unweighted_path,
#         target_convolve_raster_path = re.sub('.tif', '_convolved.tif', dt_effect_unweighted_path)) 

# convolve_2d_by_exponential(
#         decay_kernel_distance = 600, signal_raster_path = re.sub('.tif', '_buff.tif', dt_effect_kc_path),
#         target_convolve_raster_path = re.sub('.tif', '_convolved.tif', re.sub('.tif', '_buff.tif', dt_effect_kc_path)))

#### Getting the partial Temp/partial LC for barren to forest 
def partial_temp_partial_lc(dt_effect_path,
                             dt_effect_kc_path,
                             out_path=os.path.join(intermediate_dir, "partial_temp_partial_lc.tif"),
                             change_shade=1, # Wait!!! should these not be just constants? Is this where these should be applied only to the cells we want to change from one land use class to another, and 0 everywhere else????
                             change_albedo=-0.01762,
                             change_kc=0.390475):

    # --- Read shade/albedo effect raster ---
    dt_effect_ds = gdal.Open(dt_effect_path, gdal.GA_ReadOnly)
    dt_band = dt_effect_ds.GetRasterBand(1)
    dt_effect = dt_band.ReadAsArray().astype(float)
    nodata = dt_band.GetNoDataValue()

    # --- Read ET/kc effect raster ---
    dt_effect_kc_ds = gdal.Open(dt_effect_kc_path, gdal.GA_ReadOnly)
    dt_effect_kc = dt_effect_kc_ds.GetRasterBand(1).ReadAsArray().astype(float)

    # --- Mask nodata pixels (using the first raster's nodata value/location) ---
    if nodata is not None:
        mask = (dt_effect == nodata)
    else:
        mask = np.zeros_like(dt_effect, dtype=bool)
        nodata = -9999.0

    # --- Compute weighted sum ---
    result = dt_effect * 3 * change_shade + dt_effect * change_albedo + dt_effect_kc * change_kc
    result[mask] = nodata

    # --- Write output, copying georeferencing from the first input raster ---
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        out_path,
        dt_effect_ds.RasterXSize,
        dt_effect_ds.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(dt_effect_ds.GetGeoTransform())
    out_ds.SetProjection(dt_effect_ds.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result.astype(np.float32))
    out_band.SetNoDataValue(nodata)
    out_band.FlushCache()

    # --- Clean up ---
    out_band = None
    out_ds = None
    dt_effect_ds = None
    dt_effect_kc_ds = None

    return result


boundary = gpd.read_file(os.path.join(data_dir, 'Polygon_vectors/MINN_03.gpkg'))
boundary_buffed = boundary.buffer(-28)
buff_path = os.path.join(data_dir, 'Polygon_vectors/MINN_03_buffed.gpkg')
boundary_buffed.to_file(buff_path)

def buff_raster_in(raster_path):
    gdal.Warp(re.sub('.tif', '_buff.tif', raster_path), 
                    raster_path, cutlineDSName= buff_path, 
                    cropToCutline=True,
                                format="GTiff",  # Output format
                    creationOptions=[
                        f"COMPRESS=LZW",  # Compression type
                        "TILED=YES",                # Enable tiling for better performance
                        "BIGTIFF=IF_SAFER"          # Allow BigTIFF if needed
                    ])
# buff_raster_in(os.path.join(intermediate_dir, 'dt_effect_unweighted_convolved.tif'))

# partial_temp_partial_lc(
#     dt_effect_path = os.path.join(intermediate_dir, 'dt_effect_unweighted_convolved_buff.tif'),
#     dt_effect_kc_path = os.path.join(intermediate_dir, 'dt_effect_kc_buff_convolved.tif')
# )


### Need to get (\partial CDD)/(\partial Temp)

def temp_c_to_F(T):
    return 1.8 * T + 32

def smoothstep_w(F, window=1.0):
    """Returns (w, t) — w is the blend weight, t is the raw (unclipped) 
    normalized position, needed by w_prime."""
    t = (F - (70 - window)) / (2 * window)
    t_clip = np.clip(t, 0.0, 1.0)
    w = 3 * t_clip**2 - 2 * t_clip**3
    return w, t_clip  # pass the clipped t forward — see note below

def smoothstep_w_prime(t_clip, window=1.0):
    # t_clip already in [0,1]; formula naturally goes to 0 at the boundaries
    return (3 * t_clip * (1 - t_clip)) / window

def A(F):
    return 1.07e-18 * F**10.96

def B(F):
    return 29.58 * F - 1905.0

def A_prime(F):
    return 1.1733e-17 * F**9.96

B_prime = 29.58

def partial_cdd_partial_T(T_raster, window=1.0):
    """
    T_raster: numpy array of baseline temps in °C.
    Returns: array of ∂CDD/∂T, same shape as T_raster.
    """
    F = temp_c_to_F(T_raster)
    w, t_clip = smoothstep_w(F, window)
    wp = smoothstep_w_prime(t_clip, window)

    dCDD_dT = (1 - w) * A_prime(F) + w * B_prime + (B(F) - A(F)) * wp
    return dCDD_dT

# # ---- Read baseline temperature raster ----
# src_path = os.path.join(output_dir, 'T_air_upd_invest_v02.tif')
# src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
# band = src_ds.GetRasterBand(1)

# T = band.ReadAsArray().astype(float)   # °C
# nodata = band.GetNoDataValue()

# mask = (T == nodata) if nodata is not None else np.zeros_like(T, dtype=bool)

# # ---- Compute derivative ----
# dCDD_dT = partial_cdd_partial_T(T, window=1.0)

# if nodata is not None:
#     dCDD_dT[mask] = nodata   # preserve nodata pixels using the same nodata value
# else:
#     nodata = -9999.0
#     dCDD_dT[mask] = nodata

# # ---- Write output raster, copying georeferencing/projection from source ----
# driver = gdal.GetDriverByName("GTiff")
# out_ds = driver.Create(
#     os.path.join(output_dir, "dCDD_dT.tif"),
#     src_ds.RasterXSize,
#     src_ds.RasterYSize,
#     1,
#     gdal.GDT_Float32
# )
# out_ds.SetGeoTransform(src_ds.GetGeoTransform())
# out_ds.SetProjection(src_ds.GetProjection())

# out_band = out_ds.GetRasterBand(1)
# out_band.WriteArray(dCDD_dT.astype(np.float32))
# out_band.SetNoDataValue(nodata)
# out_band.FlushCache()

# # ---- Clean up ----
# out_band = None
# out_ds = None
# src_ds = None

#### Combining
# hmmmm.... I have scalars for the partial rasters... but I need T celcius for partial_cdd_partial_T... baseline air temps? or what? I think so! 
# Yay!! ok, I just need to get the rasters in the same bb and multiply! 
# buff_raster_in(os.path.join(output_dir, 'dCDD_dT.tif'))

def creating_ranking_raster(partial_cdd_partial_t_raster_path,
                             partial_t_partial_lc_raster_path,
                             out_path=os.path.join(output_dir, "ranking_raster_dollars.tif"),
                             price_const = 0.1108,
                             mc_const = 0.0106):

    # --- Read first raster ---
    ds1 = gdal.Open(partial_cdd_partial_t_raster_path, gdal.GA_ReadOnly)
    band1 = ds1.GetRasterBand(1)
    arr1 = band1.ReadAsArray().astype(float)
    nodata1 = band1.GetNoDataValue()

    # --- Read second raster ---
    ds2 = gdal.Open(partial_t_partial_lc_raster_path, gdal.GA_ReadOnly)
    band2 = ds2.GetRasterBand(1)
    arr2 = band2.ReadAsArray().astype(float)
    nodata2 = band2.GetNoDataValue()

    # --- Mask nodata pixels from either raster ---
    mask = np.zeros_like(arr1, dtype=bool)
    if nodata1 is not None:
        mask |= (arr1 == nodata1)
    if nodata2 is not None:
        mask |= (arr2 == nodata2)

    out_nodata = nodata1 if nodata1 is not None else (nodata2 if nodata2 is not None else -9999.0)

    # --- Pixelwise multiply ---
    result = arr1 * arr2 * price_const * mc_const
    result[mask] = out_nodata

    # --- Write output, copying georeferencing from the first raster ---
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        out_path,
        ds1.RasterXSize,
        ds1.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_ds.SetProjection(ds1.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result.astype(np.float32))
    out_band.SetNoDataValue(out_nodata)
    out_band.FlushCache()

    # --- Clean up ---
    out_band = None
    out_ds = None
    ds1 = None
    ds2 = None

    return result

# creating_ranking_raster(partial_cdd_partial_t_raster_path=os.path.join(output_dir, 'dCDD_dT_buff.tif'),
#                         partial_t_partial_lc_raster_path=os.path.join(intermediate_dir, "partial_temp_partial_lc.tif"))


#### getting the change in each of the biophysical parameters when going to tree
# biophysical_table = pd.read_csv(os.path.join(data_dir, 'Biophysical Tables/urbanwatch_biophys_v02.csv'))

# tree_canopy_parameters = biophysical_table.loc[
#     biophysical_table['lulc_desc'] == 'Tree Canopy', ['shade', 'kc', 'albedo']
# ].iloc[0]   # <-- .iloc[0] turns the 1-row DataFrame into a Series

# print(tree_canopy_parameters)

# # changing from lulc to tree canopy 
# biophysical_table_changes = tree_canopy_parameters - biophysical_table[['shade', 'kc', 'albedo']].astype(float)
# print(biophysical_table_changes)

# ##### getting each land cover class as it's own binary raster
# def make_binary_lu_raster(lulc_array, value):
#     '''
#     Modify ranking array to only have a value where condition_array = 1
#     '''
#     binary_raster = np.where(
#         lulc_array == value,
#         1,
#         999.0
#     )
#     return binary_raster


# lulc_path = os.path.join(data_dir, 'LULC/UrbanWatch/MINN_03_LULC_pct_al.tif')
# buff_raster_in(lulc_path)
# buff_lulc_path = os.path.join(data_dir, 'LULC/UrbanWatch/MINN_03_LULC_pct_al_buff.tif')

# for i in range(0, 9):
#     pygeoprocessing.raster_calculator([(buff_lulc_path, 1), (i, 'raw')],
#                 make_binary_lu_raster,
#                 os.path.join(output_dir, 'urban_watch_binary_' + str(i) + '.tif'),
#                 gdal.GDT_Float32,
#                 nodata_target=999.0)
    

#### ok I have the ranking raster... do I want the more negative ones or the more positive ones??? more negative ones mean more possibilities for savings! 
## Ok, I can look at just the barren pixels now and see where the priority would be and what the savings would be!!! 
# how do I bring ESI into it?

# buff_raster_in('C:/Users/kibby/OneDrive/Desktop/Research/DOI/Data/LULC/UrbanWatch/MINN_03_LULC_pct_al_only_barren.tif')

# Ok, now I need to get only the pixels in that raster, did this in raster_calculator in QGIS, but would be better to do here with pygeo.raster_calculator 

def modify_classification(ranking_array, condition_array):
    '''
    Modify ranking array to only have a value where condition_array = 1
    '''
    ranking_array_condition = np.where(
        condition_array == 1,
        ranking_array,
        999.0
    )
    return ranking_array_condition

ranking_array_path = os.path.join(output_dir, "ranking_raster_dollars.tif")
building_path = os.path.join(output_dir, 'urban_watch_binary_0.tif')

pygeoprocessing.raster_calculator([(ranking_array_path, 1), (building_path, 1)],
            modify_classification,
            os.path.join(output_dir, 'ranking_dollars_building_only.tif'),
            gdal.GDT_Float32,
            nodata_target=999.0)
