# Libby Kula
# Ch. 3
# 3/25/2026
# Purpose: 
# Update: running Urban Cooling for UrbanWatch Minneapolis section



import matplotlib
matplotlib.use('Agg')  # non-interactive, no GUI needed
import logging
import numpy as np
import os
import pygeoprocessing as pygeo
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from osgeo import gdalconst
from osgeo import gdalnumeric
from osgeo import ogr
# import matplotlib.pyplot as plt
# import multiprocessing as mp
import rasterio
from rasterio.mask import mask
import re 
from shapely.geometry import Point
import shutil
import sys

import natcap.invest.urban_cooling_model.urban_cooling_model_orig
import natcap.invest.utils

import fiona
# from fiona.ogrext import SHAPE_RESTORE_SHX

fiona.Env(SHAPE_RESTORE_SHX='YES')

os.chdir("C:/Users/kibby/OneDrive/Desktop/Research/DOI/Data")

#### Parallel processing setup ####
# print("Number of processors: ", mp.cpu_count())

######## RUNNING INVEST COOLING MODEL, LOOPING THROUGH CITIES ##########
c = 'Minneapolis-St. Paul-Bloomington, MN-WI'
month_list = ['06']
uhi_df = pd.read_csv('../Results/uhi_2_20_2026.csv')

for month in month_list:
    print(f"Processing {month}")
    
    # Getting rural reference temperature and UHIs
    ##### PRE-CALCULATED

    rural_ref_temp = uhi_df.loc[
        (uhi_df['metro'] == c) & (uhi_df['month'] == int(month)),
        't_ref'
    ].iloc[0]

    uhi = uhi_df.loc[
        (uhi_df['metro'] == c) & (uhi_df['month'] == int(month)),
        'uhi'
    ].iloc[0]

    # Masking input rasters to aoi and getting in right units
    evapotrans_path = 'Evapotranspiration/et0_V3_' + month + '_aligned_wgs84.tif'

    gdal.Warp(re.sub(r'\.(.*?)$', r"_clipped_{}.\1".format(c), evapotrans_path), 
                    evapotrans_path, cutlineDSName= f"Polygon_vectors/Metros/{c}.shp", cropToCutline=True)
    gdal.Warp(re.sub(r'\.(.*?)$', r"_clipped_{}_linear.\1".format(c), evapotrans_path), 
                    re.sub(r'\.(.*?)$', r"_clipped_{}.\1".format(c), evapotrans_path), dstSRS='EPSG:5070')

    nlcd_wgs84_path = 'LULC/SNAPP_Public_NLCD_2021_Minneapolis-St. Paul-Bloomington, MN-WI.tif'
    nlcd_path = re.sub('.tif', '_nad83.tif', nlcd_wgs84_path)

    gdal.Warp(nlcd_path, nlcd_wgs84_path, dstSRS='EPSG:5070')

    # Runnng the cooling Urban InVEST model
    args = {
        'aoi_vector_path': f"Polygon_vectors/Metros/{c}.shp",
        'biophysical_table_path': 'C:\\Users\\kibby\\OneDrive\\Desktop\\Research\\DOI\\Data\\Biophysical '
                                                    'Tables\\nlcd_biophysical_table_all_nature_water.csv',
        'cc_method': 'factors',
        'cc_weight_albedo': '',
        'cc_weight_eti': '',
        'cc_weight_shade': '',
        'do_energy_valuation': False,
        'do_productivity_valuation': False,
        'green_area_cooling_distance': '50',
        'lulc_raster_path': nlcd_path,
        'n_workers': '-1',
        'ref_eto_raster_path': re.sub(r'\.(.*?)$', r"_clipped_{}_linear.\1".format(c), evapotrans_path),
        'results_suffix': '_06_2021_upd_invest_v02',
        't_air_average_radius': '600',
        't_ref': str(rural_ref_temp),
        'uhi_max': str(uhi),
        'workspace_dir': 'C:\\Users\\kibby\\OneDrive\\Desktop\\Research\\DOI\\Code\\cooling '
                                            'workspace\\' + c + '_06_2021_upd_invest_v02', # hmmmmm.... it did not seem to do anything differently... 
        }
        
    natcap.invest.urban_cooling_model.execute(args)

    # Removing city-clipped inputs to save on space 