import numpy as np
# ========================================================
def read_gmt_boundary(filename):
    '''
    Read boundary data from text files that are extracted by GMT
    
    Input:
        filename is the filename for the boundary data file
    
    Output:
        lat_list is a list of latitudes
        lon_list is a list of longitude
    
    '''
    fid = open(filename,'r')
    lon_list, lat_list = [], []
    temp_lon, temp_lat = [], []
    for aline in fid:
        words = aline.split()
        if words[0] != '>':
            temp_lon.append(float(words[0]))
            temp_lat.append(float(words[1]))
        else:
            lon_list.append(temp_lon)
            lat_list.append(temp_lat)
            temp_lon, temp_lat = [], []
    fid.close()
    lon_list.append(temp_lon)
    lat_list.append(temp_lat)
    return lat_list, lon_list
# ========================================================
def read_boundary_data(country_boundary_file='./utility/country.xy',\
                       marine_boundary_file='./utility/marine.xy',\
                       shoreline_boundary_file='./utility/shorelines.xy',\
                       state_boundary_file='./utility/states.xy'):
    '''
    Read different types of boundaries from text files
    
    Input:
        country_boundary_file (default is './utility/country.xy') is the filename for countary boundary data
        marine_boundary_file ('./utility/marine.xy') is the filename for marine boundary data
        shoreline_boundary_file ('./utility/shorelines.xy') is the filename for shoreline boundary data
        state_boundary_file ('./utility/states.xy') is the filename for state boundary data
        
    Output:
        boundary_data is a dictionary contains four types of boundary data
    
    '''
    country_lat_list, country_lon_list = read_gmt_boundary(country_boundary_file)
    marine_lat_list, marine_lon_list = read_gmt_boundary(marine_boundary_file)
    shoreline_lat_list, shoreline_lon_list = read_gmt_boundary(shoreline_boundary_file)
    state_lat_list, state_lon_list = read_gmt_boundary(state_boundary_file)
    boundary_data = {}
    temp = {}
    temp['latitude'] = country_lat_list
    temp['longitude'] = country_lon_list
    boundary_data['country'] = temp
    temp = {}
    temp['latitude'] = marine_lat_list
    temp['longitude'] = marine_lon_list
    boundary_data['marine'] = temp
    temp = {}
    temp['latitude'] = shoreline_lat_list
    temp['longitude'] = shoreline_lon_list
    boundary_data['shoreline'] = temp
    temp = {}
    temp['latitude'] = state_lat_list
    temp['longitude'] = state_lon_list
    boundary_data['state'] = temp
    return boundary_data
# ========================================================
def get_color_list(value_list,color_min,color_max,palette,nan_value,nan_color):
    '''
    Convert a list of numbers to a list of colors
    
    Input:
        value_list is a list of numbers
        color_min is the minimum value to be converted
        color_max is the maximum value to be converted
        palette is the reference color list
        nan_value is used to label missing values
        nan_color is the color for missing values
    
    Output:
        color_list is a list of colors
    '''
    color_step = (color_max-color_min)*1./len(palette)
    color_list = []
    for i in range(len(value_list)):
        value = value_list[i]
        if value <= color_min:
            color = palette[0]
        elif value >= color_max and not value == nan_value:
            color = palette[-1]
        elif np.isnan(value):
            color = nan_color
        else:
            color_index = int(np.floor((value - color_min)/color_step))
            color = palette[color_index]
        color_list.append(color)
    return color_list