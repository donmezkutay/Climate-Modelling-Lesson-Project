from datetime import datetime

import cartopy
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot
import requests
import rioxarray
import xarray as xr
from bs4 import BeautifulSoup
from matplotlib.colors import BoundaryNorm
from scipy.constants import g
from shapely.geometry import mapping
from visjobs.datas import get_ERA5

from visualization_codes import plot_facet_map


def define_credentials():
    
    # credentials
    credentials = pd.read_csv(r'credentials.txt')
    username = credentials['username'][0]
    password = credentials['password'][0]
    
    return username, password

def define_era5_var_matching_name():
    
    var_map_dict =  {
    '2t': 'VAR_2T', #single
    'msl': 'MSL', #single
    '10u': 'VAR_10U', #single
    '10v': 'VAR_10V', #single
    'tp': 'TP', #single
    't': 'T', # pressure
    'z': 'Z' # pressure
    }
    
    return var_map_dict

def adjust_unit(data, variable):
    
    days_int_month = data['time']\
                        .dt\
                        .days_in_month\
                        .values[:, 
                                np.newaxis,
                                np.newaxis]
    unit_conversion = {
        'PMSL': 0.01,
        'RELHUM_2M': 1,
        'TOT_PREC': 1,
        'T_2M': 273.15,
        'U_10M': 3.6,
        'V_10M': 3.6,
        'FI': 1/g,
        'RELHUM': 1,
        'T': 273.15,
        'VAR_2T': 273.15,
        'MSL': 0.01,
        'VAR_10U': 3.6,
        'VAR_10V': 3.6,
        'TP': 1e3*days_int_month, # since monthly accumulation tp is m/per day in nature
        'T': 273.15, 
        'Z': 1/g,
        'u10':3.6,
        'v10':3.6,
        't2m':273.15,
        'msl':0.01,
        'tp':1e3,
        'z':1/g,
        't':273.15
        
    }
    
    if variable in ['T_2M', 'T', 'VAR_2T', 'T', 't2m', 't']:
        return data - unit_conversion[variable]
    
    return data * unit_conversion[variable]

def get_average(data, variable, month_start,
                month_end, *args, **kwargs):
    
    data = adjust_unit(data, variable)
    
    if variable in ['tp', 'TP', 'TOT_PREC']:
        return data.sel(time = (data['time.month'] >= month_start) &
                                   (data['time.month'] <= month_end),
                                   *args, **kwargs) \
                                   .sum(dim = 'time')
    else:
        return data.sel(time = (data['time.month'] >= month_start) &
                                   (data['time.month'] <= month_end),
                                   *args, **kwargs) \
                                   .mean(dim = 'time')
    
def get_average_bymonth(data, variable, month_start,
                        month_end, *args, **kwargs):
    
    data = adjust_unit(data, variable)
        
    if variable in ['tp', 'TP', 'TOT_PREC']:
        return data.groupby('time.month') \
                       .sum(dim='time') \
                       .sel(month = range(month_start,
                                          month_end+1),
                           *args, **kwargs)
    else:
        return data.groupby('time.month') \
                       .mean(dim='time') \
                       .sel(month = range(month_start,
                                          month_end+1),
                           *args, **kwargs)

def assing_proj_info(data, crs_data,
                     x_dims, y_dims):
    
    # set spatial dimension names
    data = data.rio.set_spatial_dims(x_dim = x_dims,
                                     y_dim = y_dims)
    # write crs info
    data = data.rio.write_crs(crs_data)
    
    return data

def session_accredition(username, password):
    
    #start the session
    session = requests.Session()
    session.auth = (username, password)
    
    return session

def retrieve_era5_data_link(model_level, var_name, year):
    
    # server link
    if var_name == 'tp':
        server_link = fr'https://rda.ucar.edu/thredds/catalog/files/g/ds633.1_nc/e5.moda.fc.{model_level}.accumu/{year}/catalog.html'
    else:
        server_link = fr'https://rda.ucar.edu/thredds/catalog/files/g/ds633.1_nc/e5.moda.an.{model_level}/{year}/catalog.html'
    
    # Make a request to the nomads server link
    page = requests.get(server_link)

    # View the page content
    soup = BeautifulSoup(page.content)

    # Page content under <a> tag
    a_tag = soup.find_all('a')

    # get data links
    data_links = [a.get_text() for a in a_tag if not str(a).find('.nc') == -1]
    
    var_link = server_link.replace('catalog.html', '').replace('catalog', 'dodsC') + \
               [link for link in data_links if var_name in link][0]  
    
    return var_link

def calculate_season_means(data):
    
    seasons = {
        'spring':slice(3,4,5),
        'summer':slice(6,7,8),
        'autumn':slice(9,10,11)
    }

    seasonal_dt = []

    for season in seasons.keys():
        seasonal_mean = data.sel(month=seasons[season]).mean(dim='month')
        seasonal_mean = seasonal_mean.assign_coords(coords={'season':str(season)})
        seasonal_dt.append(seasonal_mean)

    return xr.concat(seasonal_dt, dim='season')

def clip_to_city(data, shapefile, crs_data, x_dims, y_dims):
    data= data.rio.set_spatial_dims(x_dim=x_dims, y_dim=y_dims)

    data = data.rio.write_crs(crs_data)
    
    clipped = data.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs,
                            all_touched=True, invert=False)
    
    return clipped

def higher_resolution(data, x_dim, y_dim, res):
    """
    Interpolates given dataset
    """
    
    # new resolution
    new_lon = np.linspace(data[x_dim][0], data[x_dim][-1], len(data[x_dim]) * res)
    new_lat = np.linspace(data[y_dim][0], data[y_dim][-1], len(data[y_dim]) * res)
    
    # interpolate
    new = data.interpolate_na(dim = x_dim).interpolate_na(dim = y_dim)
    new = new.interp({x_dim:new_lon, y_dim:new_lat})
    
    return new
