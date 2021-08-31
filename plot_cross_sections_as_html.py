# -*- coding: utf-8 -*-
# Cross-Section Interactive Viewer
# 
# by Chengping Chai, University of Tennessee, October 4, 2017
# 
# Version 1.4
#
# Updates:
#       V1.0, Chengping Chai, University of Tennessee, October 4, 2017
#       V1.1, Chengping Chai, University of Tennessee, October 4, 2017
#         some changes for bokeh 0.12.9
#       V1.2, Chengping Chai, University of Tennessee, October 6, 2017
#         minor changes
#       V1.3, Chengping Chai, University of Tennessee, December 2, 2017
#         change the reference, replace the interpolation function
#       V1.4, Chengping Chai, Oak Ridge National Laboratory, December 31, 2018
#         update color scaling, minor changes to work with latest libraries.
#
# This script is prepared for a paper named as Interactive Visualization of
#  Complex Seismic Data and Models Using Bokeh submitted to SRL.
#
# Requirement:
#       numpy 1.15.3
#       scipy 1.1.0
#       bokeh 1.0.2
#
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from bokeh.plotting import Figure, output_file, save
from bokeh.palettes import RdYlBu11 as palette
from bokeh.plotting import ColumnDataSource
from bokeh.models import CustomJS
from bokeh.models import Column, Row
from bokeh.models.widgets import Slider, Button
from bokeh.models import FixedTicker, PrintfTickFormatter
from bokeh.models import Rect
from bokeh.models import Range1d
from bokeh.models.widgets import Div
from utility import *
import matplotlib.pyplot as plt
temp = plt.cm.get_cmap('RdYlBu',20)
my_palette = []
from matplotlib.colors import rgb2hex
for i in range(temp.N):
    rgb = temp(i)[:3] # will return rgba, we take only first 3 so we get rgb
    my_palette.append(rgb2hex(rgb))
# ========================================================
def read_3D_output_model(modelfile, nx=30, ny=30, nz=99):
    '''
    Parse 3d model from the text file.
    
    Input:
        modelfile is the filename of a 3D velocity model
        nx is the number of grid points along longitude
        ny is the number of grid points along latitude
        nz is the number of layers
        
    Output:
        results is a list of arrays, which contain model parameters of the 3D model
    '''
    fid = open(modelfile,'r')
    prd_vp = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_vs = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_thicks = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_rho = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_lons = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_lats = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_smooth = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_vsap = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_weight = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_ap = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_ae = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_tops = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_celln = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    prd_geo = [[[0 for k in range(nz)] for j in range(ny)] for i in range(nx)]
    for ilat in range(nx):
        for ilon in range(ny):
            line1 = fid.readline()
            words = line1.split()
            lat = words[10]
            lon = words[12]
            geo = words[-1]
            celln = words[5]
            line2 = fid.readline()
            words = line2.split()
            nlay = int(words[0])
            for ilay in range(nlay):
                line = fid.readline()
                words = line.split()
                prd_vp[ilat][ilon][ilay] = float(words[1])
                prd_vs[ilat][ilon][ilay] = float(words[2])
                prd_rho[ilat][ilon][ilay] = float(words[3])
                prd_thicks[ilat][ilon][ilay] = float(words[4])
                prd_smooth[ilat][ilon][ilay] = float(words[5])
                prd_vsap[ilat][ilon][ilay] = float(words[6])
                prd_weight[ilat][ilon][ilay] = float(words[7])
                prd_ap[ilat][ilon][ilay] = float(words[8])
                prd_ae[ilat][ilon][ilay] = float(words[9])
                prd_lons[ilat][ilon][ilay] = float(lon)
                prd_lats[ilat][ilon][ilay] = float(lat)
                prd_tops[ilat][ilon][ilay] = float(words[10])
                prd_celln[ilat][ilon][ilay] = celln
                prd_geo[ilat][ilon][ilay] = geo
    fid.close()
    result = [prd_vp, prd_vs, prd_rho, prd_thicks,prd_smooth,prd_vsap,prd_weight,\
prd_ap,prd_ae,prd_lons,prd_lats, prd_tops, prd_celln, prd_geo]
    return result
# ========================================================
def select_parameters(model_raw):
    '''
    Select model parameters
    
    Input:
        model_raw is a list of arrays, which contain model parameters of the 3D model
        
    Output:
        model_3D is a dictionary that contains vp, vs, rho, and layer top values
    '''
    vp_3D = model_raw[0]
    vs_3D = model_raw[1]
    rho_3D = model_raw[2]
    top_3D = model_raw[11]
    lat_3D = model_raw[10]
    lon_3D = model_raw[9]
    geo_3D = model_raw[-1]
    model_3D = {}
    model_3D['vp'] = vp_3D
    model_3D['vs'] = vs_3D
    model_3D['rho'] = rho_3D
    model_3D['top'] = top_3D
    model_3D['lat'] = lat_3D
    model_3D['lon'] = lon_3D
    model_3D['geo'] = geo_3D
    return model_3D
# ========================================================
def prepare_profile_data(model_3D):
    '''
    Prepare data for profile plots

    Input:
        model_3D is a dictionary that contains vp, vs, rho, lat, lon, 
            geological label, and layer top values

    Output:
        profile_data_all is a list of dictionaries that contain vp, vs, 
            rho, lat, lon, geological label, and layer top values
    '''
    vs_3D = model_3D['vs']
    nlat = len(vs_3D)
    profile_data_all = []
    for ilat in range(nlat):
        temp_profile = vs_3D[ilat]
        nlon = len(temp_profile)
        for ilon in range(nlon):
            vs_profile = temp_profile[ilon]
            vp_profile = model_3D['vp'][ilat][ilon]
            rho_profile = model_3D['rho'][ilat][ilon]
            top_profile = model_3D['top'][ilat][ilon]
            geo_code = model_3D['geo'][ilat][ilon][0]
            lat = model_3D['lat'][ilat][ilon][0]
            lon = model_3D['lon'][ilat][ilon][0]
            profile = {}
            profile['vs'] = vs_profile
            profile['vp'] = vp_profile
            profile['rho'] = rho_profile
            profile['top'] = top_profile
            profile['lat'] = lat
            profile['lon'] = lon
            profile['geo'] = geo_code
            profile_data_all.append(profile)
    return profile_data_all
# ========================================================
def interpolate_map_data(map_data_one_slice, nlat=30, nlon=30, \
                         dlat_interp=0.2, dlon_interp=0.2, lat_order='descend',\
                         lon_order='ascend'):
    '''
    Interpolate shear velocity values on a depth slice

    Input:
        map_data_one_slice is a dictionary that constains Vs, lat, and lon of a depth slice
        nlat is the total number of grid points along latitude
        nlon is the total number of grid points along longitude
        dlat_interp is the interpolation spacing along latitude
        dlon_interp is the interpolation spacing along longitude
        lat_order is the ordering of gird points in the latitude direction
        lon_order is the ordering of grid points in the longitude direction

    Output:
        map_data_one_slice_interpolated is a dictionary that constains interpolated Vs, lat, 
            and lon of the depth slice
    '''
    map_vs = map_data_one_slice['vs']
    map_lat = map_data_one_slice['lat']
    map_lon = map_data_one_slice['lon']
    z_list = np.zeros((nlat,nlon))
    for ilat in range(nlat):
        for ilon in range(nlon):
            index = ilat * nlon + ilon
            z_list[ilat][ilon] = map_vs[index]
    #
    x_temp = map_lon[0:nlon]
    y_temp = [map_lat[i*nlon] for i in range(nlat)]
    f = interpolate.interp2d(x_temp,y_temp,z_list,kind='cubic')
    if lon_order == 'ascend':
        lon_new = np.arange(min(x_temp),max(x_temp)+dlon_interp, dlon_interp)
    elif lon_order == 'descend':
        lon_new = np.arange(max(x_temp),min(x_temp)-dlon_interp, -dlon_interp)
    if lat_order == 'ascend':
        lat_new = np.arange(min(y_temp),max(y_temp)+dlat_interp, dlat_interp)
    elif lat_order == 'descend':
        lat_new = np.arange(max(y_temp),min(y_temp)-dlat_interp, -dlat_interp)
    z_new = f(lon_new,lat_new)
    map_data_one_slice_interpolated = {}
    map_data_one_slice_interpolated['vs'] = z_new
    map_data_one_slice_interpolated['lat'] = lat_new
    map_data_one_slice_interpolated['lon'] = lon_new
    return map_data_one_slice_interpolated
# ========================================================
def clip_map_data(map_data_one_slice_interpolated, vs_min, vs_max):
    '''
    Clip velocity values based on vs_min and vs_max

    Input:
        map_data_one_slice_interpolated is a dictionary that constains interpolated Vs, lat, 
            and lon of the depth slice
        vs_min is the minimum Vs value of the color range
        vs_max is the maximum Vs value of the color range

    Output:
        map_data_one_slice_interpolated is a dictionary similar to the input dictionary, but
            with Vs values clipped with vs_min and vs_max.
    '''
    vs_list = map_data_one_slice_interpolated['vs']
    for i in range(len(vs_list)):
        vs_temp = vs_list[i]
        for j in range(len(vs_temp)):
            vs = vs_temp[j]
            if vs > vs_max:
                vs_list[i][j] = vs_max
            if vs < vs_min:
                vs_list[i][j] = vs_min
    #
    map_data_one_slice_interpolated['vs'] = vs_list
    return map_data_one_slice_interpolated
# ========================================================
def compute_color_range(map_vs_one_slice, map_geo_one_slice, style_parameter):
    '''
    Compute color ranges using on-land values. The color range is computed based on the average
        velocity value, the standard derivation and a control constant.

    Input:
        map_vs_one_slice is a list of Vs values on a depth slice
        map_geo_one_slice is a list of geological labels on a depth slice
        style_parameter is a dictionary of options that are used to change the style of plots

    Output:
        vs_min is the minimum value of the color range
        vs_max is the maximum value of the color range
    '''
    continental_list = []
    for i in range(len(map_geo_one_slice)):
        geo_code = map_geo_one_slice[i]
        if geo_code == 'co':
            continental_list.append(map_vs_one_slice[i])
    vs_mean = np.mean(continental_list)
    vs_std = np.std(continental_list)
    if vs_std > style_parameter['min_vs_range']/2.:
        vs_half_spread = vs_std
    else:
        vs_half_spread = style_parameter['min_vs_range']/2.
    vs_min = vs_mean - vs_half_spread * style_parameter['spread_factor']
    vs_max = vs_mean + vs_half_spread * style_parameter['spread_factor']
    return vs_min, vs_max
# ========================================================
def prepare_map_data(profile_data_all, ndepth=54):
    '''
    Prepare data for map view plots

    Input:
        profile_data_all is a list of dictionaries that contain vp, vs, 
            rho, lat, lon, geological label, and layer top values

        ndepth is the total number of depth slices

    Output:
        map_data_all is a list of dictionaries that contain Vs, lat, lon values and geological labels
        map_depth_all is a list of depths of the slices
        color_range_all_slices is a list of color ranges of the depth slices
    '''
    map_data_all = []
    color_range_all_slices = []
    nprofile = len(profile_data_all)
    map_depth_all = []
    for idepth in range(ndepth):
        map_data_one_slice = {}
        map_vs_one_slice = []
        map_lat_one_slice = []
        map_lon_one_slice = []
       
        map_geo_one_slice = []
        for iprofile in range(nprofile):
            profile = profile_data_all[iprofile]
            vs = profile['vs']
            lat = profile['lat']
            lon = profile['lon']
            geo_code = profile['geo']
            map_vs_one_slice.append(vs[idepth])
            map_lat_one_slice.append(lat)
            map_lon_one_slice.append(lon)
            map_geo_one_slice.append(geo_code)
        #
        layer_mid_depth = profile['top'][idepth] + (profile['top'][idepth+1] - \
                                                    profile['top'][idepth])*0.5 
        map_depth_all.append(layer_mid_depth)
        map_data_one_slice['vs'] = map_vs_one_slice
        map_data_one_slice['lat'] = map_lat_one_slice
        map_data_one_slice['lon'] = map_lon_one_slice
        
        # interpolate a depth slice
        map_data_one_slice_interpolated = interpolate_map_data(map_data_one_slice)
        # compute color ranges
        vs_min, vs_max = compute_color_range(map_vs_one_slice, map_geo_one_slice, \
                                             style_parameter)
        color_range_all_slices.append((vs_min, vs_max))
        # clip values that are out of color ranges
        #map_data_one_slice_clipped = clip_map_data(map_data_one_slice_interpolated, \
        #                                          vs_min, vs_max)
        #
        map_data_all.append(map_data_one_slice_interpolated['vs'])
    return map_data_all, map_depth_all, color_range_all_slices
# ========================================================
def prepare_lat_cross_data(model_3D, style_parameter):
    ndepth = style_parameter['map_view_ndepth']
    vs_min = style_parameter['cross_view_vs_min']
    vs_max = style_parameter['cross_view_vs_max']
    dlon_interp = style_parameter['cross_dlon']
    ddepth_interp = style_parameter['cross_ddepth']
    cross_lat_data_all = []
    cross_lat_all = sorted(list(set(np.array(model_3D['lat']).flatten())))
    data_lon_all = []
    data_depth_all = []
    depth = np.arange(200) * ddepth_interp
    #
    x_list = [-model_3D['lat'][i][0][0] for i in range(np.shape(model_3D['lat'])[0])]
    y_list = [model_3D['lon'][0][i][0]+360  for i in range(np.shape(model_3D['lon'])[1])]
    z_list = [model_3D['top'][0][0][i] for i in range(np.shape(model_3D['top'])[2])]
    #
    interpolator_vs = RegularGridInterpolator((x_list, y_list, z_list), model_3D['vs'])
    #
    for ilat in range(len(cross_lat_all)):
        cross_data_one_slice = {}
        cross_vs_one_slice = []
        cross_lon_one_slice = []
        cross_depth_one_slice = []
        lat = cross_lat_all[ilat]
        lon_min = np.min(model_3D['lon'])
        lon_max = np.max(model_3D['lon'])
        nlon = int(abs(lon_max - lon_min) / dlon_interp) + 1
        lon_list = lon_max - np.arange(nlon) * dlon_interp
        lat_list = np.array([lat]*nlon)
        for i in range(len(lon_list)):
            lat = lat_list[i]
            lon = lon_list[i]
            temp_list = []
            for ik in range(len(depth)-1, -1, -1):
                coords = [-lat, lon+360, depth[ik]]
                temp = interpolator_vs(coords)[0]
                temp_list.append(temp)
            cross_vs_one_slice.append(temp_list)
        #
        # for i in range(len(cross_vs_one_slice)):
        #     temp = cross_vs_one_slice[i]
        #     for j in range(len(temp)):
        #         vs = temp[j]
        #         if vs > vs_max:
        #             cross_vs_one_slice[i][j] = vs_max
        #         if vs < vs_min:
        #             cross_vs_one_slice[i][j] = vs_min
        #
        transposed = []
        for i in range(np.shape(cross_vs_one_slice)[1]):
            temp_list = []
            temp_len = np.shape(cross_vs_one_slice)[0]
            for j in range(np.shape(cross_vs_one_slice)[0]):
                temp_list.append(cross_vs_one_slice[temp_len-j-1][i])
            transposed.append(temp_list)
        cross_lat_data_all.append(transposed)
    return cross_lat_data_all, cross_lat_all
# ========================================================
def prepare_lon_cross_data(model_3D, style_parameter):
    ndepth = style_parameter['map_view_ndepth']
    vs_min = style_parameter['cross_view_vs_min']
    vs_max = style_parameter['cross_view_vs_max']
    dlat_interp = style_parameter['cross_dlat']
    ddepth_interp = style_parameter['cross_ddepth']
    cross_lon_data_all = []
    cross_lon_all = sorted(list(set(np.array(model_3D['lon']).flatten())))
    data_lat_all = []
    data_depth_all = []
    depth = np.arange(200) * ddepth_interp
    #
    x_list = [-model_3D['lat'][i][0][0] for i in range(np.shape(model_3D['lat'])[0])]
    y_list = [model_3D['lon'][0][i][0]+360  for i in range(np.shape(model_3D['lon'])[1])]
    z_list = [model_3D['top'][0][0][i] for i in range(np.shape(model_3D['top'])[2])]
    #
    interpolator_vs = RegularGridInterpolator((x_list, y_list, z_list), model_3D['vs'])
    #
    for ilon in range(len(cross_lon_all)):
        cross_data_one_slice = {}
        cross_vs_one_slice = []
        cross_lat_one_slice = []
        cross_depth_one_slice = []
        lon = cross_lon_all[ilon]
        lat_min = np.min(model_3D['lat'])
        lat_max = np.max(model_3D['lat'])
        nlat = int(abs(lat_max - lat_min) / dlat_interp)
        lat_list = lat_max - np.arange(nlat) * dlat_interp
        lon_list = np.array([lon]*nlat)
        for i in range(len(lat_list)):
            lat = lat_list[i]
            lon = lon_list[i]
            temp_list = []
            for ik in range(len(depth)-1, -1, -1):
                coords = [-lat, lon+360, depth[ik]]
                temp = interpolator_vs(coords)[0]
                temp_list.append(temp)
            cross_vs_one_slice.append(temp_list)
        #
        # for i in range(len(cross_vs_one_slice)):
        #     temp = cross_vs_one_slice[i]
        #     for j in range(len(temp)):
        #         vs = temp[j]
        #         if vs > vs_max:
        #             cross_vs_one_slice[i][j] = vs_max
        #         if vs < vs_min:
        #             cross_vs_one_slice[i][j] = vs_min
        #
        transposed = []
        for i in range(np.shape(cross_vs_one_slice)[1]):
            temp_list = []
            temp_len = np.shape(cross_vs_one_slice)[0]
            for j in range(np.shape(cross_vs_one_slice)[0]):
                temp_list.append(cross_vs_one_slice[temp_len-j-1][i])
            transposed.append(temp_list)
        cross_lon_data_all.append(transposed)
    return cross_lon_data_all, cross_lon_all
# ========================================================
def val_to_rgb(map_data_one_slice, palette, vmin, vmax):
    color_data = np.zeros((np.shape(map_data_one_slice)[0], np.shape(map_data_one_slice)[1],4), dtype=np.uint8)
    min_map = vmin
    max_map = vmax
    ncolor = len(palette)
    dc_map = (max_map - min_map)/ncolor
    for ix in range(np.shape(map_data_one_slice)[0]):
        for iy in range(np.shape(map_data_one_slice)[1]):
            val = map_data_one_slice[ix][iy]
            index = int(np.floor((val - min_map)/dc_map))
            if index < 0:
                index = 0
            if index >= ncolor:
                index = ncolor - 1
            color = palette[index].lstrip('#')
            red = int(color[0:0+2], 16)
            green = int(color[2:2+2], 16)
            blue = int(color[4:4+2], 16)
            color_data[ix][iy][0] = red
            color_data[ix][iy][1] = green
            color_data[ix][iy][2] = blue
            color_data[ix][iy][3] = 255
    #
    return color_data
# ========================================================
def plot_cross_section_bokeh(filename, map_data_all_slices, map_depth_all_slices, \
                             color_range_all_slices, profile_data_all, cross_lat_data_all, \
                             lat_value_all, cross_lon_data_all, lon_value_all, boundary_data, \
                             style_parameter):
    '''
    Plot shear velocity maps and cross-sections using bokeh

    Input:
        filename is the filename of the resulting html file
        map_data_all_slices contains the velocity model parameters saved for map view plots
        map_depth_all_slices is a list of depths
        color_range_all_slices is a list of color ranges
        profile_data_all is a list of velocity profiles
        cross_lat_data_all is a list of cross-sections along latitude
        lat_value_all is a list of corresponding latitudes for these cross-sections
        cross_lon_data_all is a list of cross-sections along longitude
        lon_value_all is a list of corresponding longitudes for these cross-sections
        boundary_data is a list of boundaries
        style_parameter contains parameters to customize the plots

    Output:
        None
    
    '''
    xlabel_fontsize = style_parameter['xlabel_fontsize']
    #
    colorbar_data_all_left = []
    colorbar_data_all_right = []
    map_view_ndepth = style_parameter['map_view_ndepth']
    palette_r = palette[::-1]
    ncolor = len(palette_r)
    colorbar_top = [0.1 for i in range(ncolor)]
    colorbar_bottom = [0 for i in range(ncolor)]
    map_data_all_slices_depth = []
    for idepth in range(map_view_ndepth): 
        color_min = color_range_all_slices[idepth][0]
        color_max = color_range_all_slices[idepth][1]
        color_step = (color_max - color_min)*1./ncolor
        colorbar_left = np.linspace(color_min,color_max-color_step,ncolor)
        colorbar_right = np.linspace(color_min+color_step,color_max,ncolor)
        colorbar_data_all_left.append(colorbar_left)
        colorbar_data_all_right.append(colorbar_right)
        map_depth = map_depth_all_slices[idepth]
        map_data_all_slices_depth.append('Depth: {0:8.0f} km'.format(map_depth))
    # data for the colorbar
    colorbar_data_one_slice = {}
    colorbar_data_one_slice['colorbar_left'] = colorbar_data_all_left[style_parameter['map_view_default_index']]
    colorbar_data_one_slice['colorbar_right'] = colorbar_data_all_right[style_parameter['map_view_default_index']]
    colorbar_data_one_slice_bokeh = ColumnDataSource(data=dict(colorbar_top=colorbar_top,colorbar_bottom=colorbar_bottom,\
                                                               colorbar_left=colorbar_data_one_slice['colorbar_left'],\
                                                               colorbar_right=colorbar_data_one_slice['colorbar_right'],\
                                                               palette_r=palette_r))
    colorbar_data_all_slices_bokeh = ColumnDataSource(data=dict(colorbar_data_all_left=colorbar_data_all_left,\
                                                                colorbar_data_all_right=colorbar_data_all_right))
    #
    map_view_label_lon = style_parameter['map_view_depth_label_lon']
    map_view_label_lat = style_parameter['map_view_depth_label_lat']
    map_data_one_slice_depth = map_data_all_slices_depth[style_parameter['map_view_default_index']]
    map_data_one_slice_depth_bokeh = ColumnDataSource(data=dict(lat=[map_view_label_lat], lon=[map_view_label_lon],
                                                           map_depth=[map_data_one_slice_depth]))
    
    #
    map_view_default_index = style_parameter['map_view_default_index']
    #map_data_one_slice = map_data_all_slices[map_view_default_index]
    map_color_all_slices = []
    for i in range(len(map_data_all_slices)):
        vmin, vmax = color_range_all_slices[i]
        map_color = val_to_rgb(map_data_all_slices[i], palette_r, vmin, vmax)
        map_color_2d = map_color.view('uint32').reshape(map_color.shape[:2])
        map_color_all_slices.append(map_color_2d)
    map_color_one_slice = map_color_all_slices[map_view_default_index]
    #
    map_data_one_slice_bokeh = ColumnDataSource(data=dict(x=[style_parameter['map_view_image_lon_min']],\
                   y=[style_parameter['map_view_image_lat_min']],dw=[style_parameter['nlon']],\
                   dh=[style_parameter['nlat']],map_data_one_slice=[map_color_one_slice]))
    map_data_all_slices_bokeh = ColumnDataSource(data=dict(map_data_all_slices=map_color_all_slices,\
                                                           map_data_all_slices_depth=map_data_all_slices_depth))
    #
    cross_lat_label_all = []
    for a in lat_value_all:
        cross_lat_label_all.append('Lat: '+str(a))
    #cross_lat_data_one_slice = cross_lat_data_all[style_parameter['cross_lat_default_index']]
    cross_lat_color_all = []
    for i in range(len(cross_lat_data_all)):
        vmin = style_parameter['cross_view_vs_min']
        vmax = style_parameter['cross_view_vs_max']
        cross_color = val_to_rgb(cross_lat_data_all[i], palette_r, vmin, vmax)
        cross_color_2d = cross_color.view('uint32').reshape(cross_color.shape[:2])
        cross_lat_color_all.append(cross_color_2d)
    cross_lat_color_one_slice = cross_lat_color_all[style_parameter['cross_lat_default_index']]
    #
    plot_depth = np.shape(cross_lat_color_one_slice)[0] * style_parameter['cross_ddepth']
    cross_lat_data_one_slice_bokeh = ColumnDataSource(data=dict(x=[style_parameter['map_view_image_lon_min']],\
                   y=[plot_depth],dw=[style_parameter['nlon']],\
                   dh=[plot_depth],cross_lat_data_one_slice=[cross_lat_color_one_slice]))
    cross_lat_data_all_slices_bokeh = ColumnDataSource(data=dict(cross_lat_data_all=cross_lat_color_all))
    cross_lat_view_label_lon = style_parameter['cross_lat_view_label_lon']
    cross_lat_view_label_depth = style_parameter['cross_lat_view_label_depth']
    default_lat = lat_value_all[style_parameter['cross_lat_default_index']]
    cross_lat_value_all_bokeh = ColumnDataSource(data=dict(lat_value_all=lat_value_all))
    cross_lat_label_all_bokeh = ColumnDataSource(data=dict(lat_label_all=cross_lat_label_all))
    cross_lat_label_bokeh = ColumnDataSource(data=dict(lon=[cross_lat_view_label_lon],\
                                                      depth=[cross_lat_view_label_depth],\
                                                      lat_label=[cross_lat_label_all[style_parameter['cross_lat_default_index']]]))
    map_lat_line_bokeh = ColumnDataSource(data=dict(lat=[default_lat, default_lat],\
                                                    lon=[style_parameter['map_view_image_lon_min'], style_parameter['map_view_image_lon_max']]))
    #
    cross_lon_label_all = []
    for a in lon_value_all:
        cross_lon_label_all.append('Lon: '+str(a))
    #cross_lon_data_one_slice = cross_lon_data_all[style_parameter['cross_lon_default_index']]
    cross_lon_color_all = []
    for i in range(len(cross_lon_data_all)):
        vmin = style_parameter['cross_view_vs_min']
        vmax = style_parameter['cross_view_vs_max']
        cross_color = val_to_rgb(cross_lon_data_all[i], palette_r, vmin, vmax)
        cross_color_2d = cross_color.view('uint32').reshape(cross_color.shape[:2])
        cross_lon_color_all.append(cross_color_2d)
    cross_lon_color_one_slice = cross_lon_color_all[style_parameter['cross_lon_default_index']]
    #
    plot_depth = np.shape(cross_lon_color_one_slice)[0] * style_parameter['cross_ddepth']
    cross_lon_data_one_slice_bokeh = ColumnDataSource(data=dict(x=[style_parameter['map_view_image_lat_min']],\
                   y=[plot_depth],dw=[style_parameter['nlat']],\
                   dh=[plot_depth],cross_lon_data_one_slice=[cross_lon_color_one_slice]))
    cross_lon_data_all_slices_bokeh = ColumnDataSource(data=dict(cross_lon_data_all=cross_lon_color_all))
    cross_lon_view_label_lat = style_parameter['cross_lon_view_label_lat']
    cross_lon_view_label_depth = style_parameter['cross_lon_view_label_depth']
    default_lon = lon_value_all[style_parameter['cross_lon_default_index']]
    cross_lon_value_all_bokeh = ColumnDataSource(data=dict(lon_value_all=lon_value_all))
    cross_lon_label_all_bokeh = ColumnDataSource(data=dict(lon_label_all=cross_lon_label_all))
    cross_lon_label_bokeh = ColumnDataSource(data=dict(lat=[cross_lon_view_label_lat],\
                                                      depth=[cross_lon_view_label_depth],\
                                                      lon_label=[cross_lon_label_all[style_parameter['cross_lon_default_index']]]))
    map_lon_line_bokeh = ColumnDataSource(data=dict(lat=[style_parameter['map_view_image_lat_min'], style_parameter['map_view_image_lat_max']],\
                                                    lon=[default_lon, default_lon]))
    #
    ncolor_cross = len(my_palette)
    colorbar_top_cross = [0.1 for i in range(ncolor_cross)]
    colorbar_bottom_cross = [0 for i in range(ncolor_cross)]
    color_min_cross = style_parameter['cross_view_vs_min']
    color_max_cross = style_parameter['cross_view_vs_max']
    color_step_cross = (color_max_cross - color_min_cross)*1./ncolor_cross
    colorbar_left_cross = np.linspace(color_min_cross, color_max_cross-color_step_cross, ncolor_cross)
    colorbar_right_cross = np.linspace(color_min_cross+color_step_cross, color_max_cross, ncolor_cross)
    # ==============================
    map_view = Figure(plot_width=style_parameter['map_view_plot_width'], plot_height=style_parameter['map_view_plot_height'], \
                      tools=style_parameter['map_view_tools'], title=style_parameter['map_view_title'], \
                      y_range=[style_parameter['map_view_figure_lat_min'], style_parameter['map_view_figure_lat_max']],\
                      x_range=[style_parameter['map_view_figure_lon_min'], style_parameter['map_view_figure_lon_max']])
    #
    map_view.image_rgba('map_data_one_slice',x='x',\
                   y='y',dw='dw',dh='dh',
                   source=map_data_one_slice_bokeh, level='image')

    depth_slider_callback = CustomJS(args=dict(map_data_one_slice_bokeh=map_data_one_slice_bokeh,\
                                               map_data_all_slices_bokeh=map_data_all_slices_bokeh,\
                                               colorbar_data_all_slices_bokeh=colorbar_data_all_slices_bokeh,\
                                               colorbar_data_one_slice_bokeh=colorbar_data_one_slice_bokeh,\
                                               map_data_one_slice_depth_bokeh=map_data_one_slice_depth_bokeh), code="""

        var d_index = Math.round(cb_obj.value)
        
        var map_data_all_slices = map_data_all_slices_bokeh.data
        
        map_data_one_slice_bokeh.data['map_data_one_slice'] = [map_data_all_slices['map_data_all_slices'][d_index]]
        map_data_one_slice_bokeh.change.emit()
        
        var color_data_all_slices = colorbar_data_all_slices_bokeh.data
        colorbar_data_one_slice_bokeh.data['colorbar_left'] = color_data_all_slices['colorbar_data_all_left'][d_index]
        colorbar_data_one_slice_bokeh.data['colorbar_right'] = color_data_all_slices['colorbar_data_all_right'][d_index]
        colorbar_data_one_slice_bokeh.change.emit()
        
        map_data_one_slice_depth_bokeh.data['map_depth'] = [map_data_all_slices['map_data_all_slices_depth'][d_index]]
        map_data_one_slice_depth_bokeh.change.emit()
        
    """) 
    depth_slider = Slider(start=0, end=style_parameter['map_view_ndepth']-1, \
                          value=map_view_default_index, step=1, \
                          width=style_parameter['map_view_plot_width'],\
                          title=style_parameter['depth_slider_title'], height=50)
    depth_slider.js_on_change('value', depth_slider_callback)
    depth_slider_callback.args["depth_index"] = depth_slider
    # ------------------------------
    # add boundaries to map view
    # country boundaries
    map_view.multi_line(boundary_data['country']['longitude'],\
                        boundary_data['country']['latitude'],color='black',\
                        line_width=2, level='underlay',nonselection_line_alpha=1.0,\
                        nonselection_line_color='black')
    # marine boundaries
    map_view.multi_line(boundary_data['marine']['longitude'],\
                        boundary_data['marine']['latitude'],color='black',\
                        level='underlay',nonselection_line_alpha=1.0,\
                        nonselection_line_color='black')
    # shoreline boundaries
    map_view.multi_line(boundary_data['shoreline']['longitude'],\
                        boundary_data['shoreline']['latitude'],color='black',\
                        line_width=2, level='underlay',nonselection_line_alpha=1.0,\
                        nonselection_line_color='black')
    # state boundaries
    map_view.multi_line(boundary_data['state']['longitude'],\
                        boundary_data['state']['latitude'],color='black',\
                        level='underlay',nonselection_line_alpha=1.0,\
                        nonselection_line_color='black')
     # ------------------------------
    # add depth label
    map_view.rect(style_parameter['map_view_depth_box_lon'], style_parameter['map_view_depth_box_lat'], \
                  width=style_parameter['map_view_depth_box_width'], height=style_parameter['map_view_depth_box_height'], \
                  width_units='screen',height_units='screen', color='#FFFFFF', line_width=1., line_color='black', level='underlay')
    map_view.text('lon', 'lat', 'map_depth', source=map_data_one_slice_depth_bokeh,\
                  text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    # ------------------------------
    map_view.line('lon', 'lat', source=map_lat_line_bokeh, line_dash=[8,2,8,2], line_color='#00ff00',\
                        nonselection_line_alpha=1.0, line_width=5.,\
                        nonselection_line_color='black')
    map_view.line('lon', 'lat', source=map_lon_line_bokeh, line_dash=[8,2,8,2], line_color='#00ff00',\
                        nonselection_line_alpha=1.0, line_width=5.,\
                        nonselection_line_color='black')
    # ------------------------------
    # change style
    map_view.title.text_font_size = style_parameter['title_font_size']
    map_view.title.align = 'center'
    map_view.title.text_font_style = 'normal'
    map_view.xaxis.axis_label = style_parameter['map_view_xlabel']
    map_view.xaxis.axis_label_text_font_style = 'normal'
    map_view.xaxis.axis_label_text_font_size = xlabel_fontsize
    map_view.xaxis.major_label_text_font_size = xlabel_fontsize
    map_view.yaxis.axis_label = style_parameter['map_view_ylabel']
    map_view.yaxis.axis_label_text_font_style = 'normal'
    map_view.yaxis.axis_label_text_font_size = xlabel_fontsize
    map_view.yaxis.major_label_text_font_size = xlabel_fontsize
    map_view.xgrid.grid_line_color = None
    map_view.ygrid.grid_line_color = None
    map_view.toolbar.logo = None
    map_view.toolbar_location = 'above'
    map_view.toolbar_sticky = False
    # ==============================
    # plot colorbar
    colorbar_fig = Figure(tools=[], y_range=(0,0.1),plot_width=style_parameter['map_view_plot_width'], \
                      plot_height=style_parameter['colorbar_plot_height'],title=style_parameter['colorbar_title'])
    colorbar_fig.toolbar_location=None
    colorbar_fig.quad(top='colorbar_top',bottom='colorbar_bottom',left='colorbar_left',right='colorbar_right',\
                  color='palette_r',source=colorbar_data_one_slice_bokeh)
    colorbar_fig.yaxis[0].ticker=FixedTicker(ticks=[])
    colorbar_fig.xgrid.grid_line_color = None
    colorbar_fig.ygrid.grid_line_color = None
    colorbar_fig.xaxis.axis_label_text_font_size = xlabel_fontsize
    colorbar_fig.xaxis.major_label_text_font_size = xlabel_fontsize
    colorbar_fig.xaxis[0].formatter = PrintfTickFormatter(format="%5.2f")
    colorbar_fig.title.text_font_size = xlabel_fontsize
    colorbar_fig.title.align = 'center'
    colorbar_fig.title.text_font_style = 'normal'
     # ==============================
    # annotating text
    annotating_fig01 = Div(text=style_parameter['annotating_html01'], \
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    # ==============================
    # plot cross-section along latitude
    cross_lat_view = Figure(plot_width= style_parameter['cross_lat_plot_width'], plot_height=style_parameter['cross_lat_plot_height'], \
                      tools=style_parameter['cross_lat_view_tools'], title=style_parameter['cross_lat_view_title'], \
                      y_range=[plot_depth, -30],\
                      x_range=[style_parameter['map_view_image_lon_min'], style_parameter['map_view_image_lon_max']])
    cross_lat_view.image_rgba('cross_lat_data_one_slice',x='x',\
                   y='y',dw='dw',dh='dh',\
                   source=cross_lat_data_one_slice_bokeh, level='image')
    cross_lat_view.text('lon', 'depth', 'lat_label', source=cross_lat_label_bokeh,\
                   text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    cross_lat_view.text([style_parameter['map_view_image_lon_min']+2], [style_parameter['cross_lat_view_label_depth']],\
                        text=['E'], text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    cross_lat_view.text([style_parameter['map_view_image_lon_max']-2], [style_parameter['cross_lat_view_label_depth']],\
                        text=['W'], text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    # ------------------------------
    # change style
    cross_lat_view.title.text_font_size = style_parameter['title_font_size']
    cross_lat_view.title.align = 'center'
    cross_lat_view.title.text_font_style = 'normal'
    cross_lat_view.xaxis.axis_label = style_parameter['cross_lat_view_xlabel']
    cross_lat_view.xaxis.axis_label_text_font_style = 'normal'
    cross_lat_view.xaxis.axis_label_text_font_size = xlabel_fontsize
    cross_lat_view.xaxis.major_label_text_font_size = xlabel_fontsize
    cross_lat_view.yaxis.axis_label = style_parameter['cross_lat_view_ylabel']
    cross_lat_view.yaxis.axis_label_text_font_style = 'normal'
    cross_lat_view.yaxis.axis_label_text_font_size = xlabel_fontsize
    cross_lat_view.yaxis.major_label_text_font_size = xlabel_fontsize
    cross_lat_view.xgrid.grid_line_color = None
    cross_lat_view.ygrid.grid_line_color = None
    cross_lat_view.toolbar.logo = None
    cross_lat_view.toolbar_location = 'right'
    cross_lat_view.toolbar_sticky = False
    # ==============================
    # plot cross-section along latitude
    cross_lon_view = Figure(plot_width= style_parameter['cross_lon_plot_width'], plot_height=style_parameter['cross_lon_plot_height'], \
                      tools=style_parameter['cross_lon_view_tools'], title=style_parameter['cross_lon_view_title'], \
                      y_range=[plot_depth, -30],\
                      x_range=[style_parameter['map_view_image_lat_min'], style_parameter['map_view_image_lat_max']])
    cross_lon_view.image_rgba('cross_lon_data_one_slice',x='x',\
                   y='y',dw='dw',dh='dh',\
                   source=cross_lon_data_one_slice_bokeh, level='image')
    cross_lon_view.text('lat', 'depth', 'lon_label', source=cross_lon_label_bokeh,\
                   text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    cross_lon_view.text([style_parameter['map_view_image_lat_min']+2], [style_parameter['cross_lat_view_label_depth']],\
                        text=['S'], text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    cross_lon_view.text([style_parameter['map_view_image_lat_max']-2], [style_parameter['cross_lat_view_label_depth']],\
                        text=['N'], text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    # ------------------------------
    # change style
    cross_lon_view.title.text_font_size = style_parameter['title_font_size']
    cross_lon_view.title.align = 'center'
    cross_lon_view.title.text_font_style = 'normal'
    cross_lon_view.xaxis.axis_label = style_parameter['cross_lon_view_xlabel']
    cross_lon_view.xaxis.axis_label_text_font_style = 'normal'
    cross_lon_view.xaxis.axis_label_text_font_size = xlabel_fontsize
    cross_lon_view.xaxis.major_label_text_font_size = xlabel_fontsize
    cross_lon_view.yaxis.axis_label = style_parameter['cross_lon_view_ylabel']
    cross_lon_view.yaxis.axis_label_text_font_style = 'normal'
    cross_lon_view.yaxis.axis_label_text_font_size = xlabel_fontsize
    cross_lon_view.yaxis.major_label_text_font_size = xlabel_fontsize
    cross_lon_view.xgrid.grid_line_color = None
    cross_lon_view.ygrid.grid_line_color = None
    cross_lon_view.toolbar.logo = None
    cross_lon_view.toolbar_location = 'right'
    cross_lon_view.toolbar_sticky = False
    # ==============================
    colorbar_fig_right = Figure(tools=[], y_range=(0,0.1),plot_width=style_parameter['cross_lat_plot_width'], \
                      plot_height=style_parameter['colorbar_plot_height'],title=style_parameter['colorbar_title'])
    colorbar_fig_right.toolbar_location=None
    
    colorbar_fig_right.quad(top=colorbar_top_cross,bottom=colorbar_bottom_cross,\
                            left=colorbar_left_cross,right=colorbar_right_cross,\
                            color=my_palette)
    colorbar_fig_right.yaxis[0].ticker=FixedTicker(ticks=[])
    colorbar_fig_right.xgrid.grid_line_color = None
    colorbar_fig_right.ygrid.grid_line_color = None
    colorbar_fig_right.xaxis.axis_label_text_font_size = xlabel_fontsize
    colorbar_fig_right.xaxis.major_label_text_font_size = xlabel_fontsize
    colorbar_fig_right.xaxis[0].formatter = PrintfTickFormatter(format="%5.2f")
    colorbar_fig_right.title.text_font_size = xlabel_fontsize
    colorbar_fig_right.title.align = 'center'
    colorbar_fig_right.title.text_font_style = 'normal'
    # ==============================
    lat_slider_callback = CustomJS(args=dict(cross_lat_data_one_slice_bokeh=cross_lat_data_one_slice_bokeh,\
                                             cross_lat_data_all_slices_bokeh=cross_lat_data_all_slices_bokeh,\
                                             cross_lat_label_all_bokeh=cross_lat_label_all_bokeh,\
                                             cross_lat_label_bokeh=cross_lat_label_bokeh,\
                                             map_lat_line_bokeh=map_lat_line_bokeh,\
                                             cross_lat_value_all_bokeh=cross_lat_value_all_bokeh), code="""

        var lat_index = Math.round(cb_obj.value)
        
        var cross_lat_data_all_slices = cross_lat_data_all_slices_bokeh.data
        
        cross_lat_data_one_slice_bokeh.data['cross_lat_data_one_slice'] = [cross_lat_data_all_slices['cross_lat_data_all'][lat_index]]
        cross_lat_data_one_slice_bokeh.change.emit()
        
        cross_lat_label_bokeh.data['lat_label'] = [cross_lat_label_all_bokeh.data['lat_label_all'][lat_index]]
        cross_lat_label_bokeh.change.emit()
        
        var lat_value = cross_lat_value_all_bokeh.data['lat_value_all'][lat_index]
        map_lat_line_bokeh.data['lat'] = [lat_value, lat_value]
        map_lat_line_bokeh.change.emit()
    """) 
    lat_slider = Slider(start=0, end=style_parameter['nlat']-1, \
                          value=style_parameter['cross_lat_default_index'], step=1, \
                          width=style_parameter['cross_lat_plot_width'],\
                          title=style_parameter['lat_slider_title'], height=50)
    lat_slider.js_on_change('value', lat_slider_callback)
    lat_slider_callback.args['lat_index'] = lat_slider
    # ==============================
    lon_slider_callback = CustomJS(args=dict(cross_lon_data_one_slice_bokeh=cross_lon_data_one_slice_bokeh,\
                                             cross_lon_data_all_slices_bokeh=cross_lon_data_all_slices_bokeh,\
                                             cross_lon_label_all_bokeh=cross_lon_label_all_bokeh,\
                                             cross_lon_label_bokeh=cross_lon_label_bokeh,\
                                             map_lon_line_bokeh=map_lon_line_bokeh,\
                                             cross_lon_value_all_bokeh=cross_lon_value_all_bokeh), code="""

        var lon_index = Math.round(cb_obj.value)
        
        var cross_lon_data_all_slices = cross_lon_data_all_slices_bokeh.data
        
        cross_lon_data_one_slice_bokeh.data['cross_lon_data_one_slice'] = [cross_lon_data_all_slices['cross_lon_data_all'][lon_index]]
        cross_lon_data_one_slice_bokeh.change.emit()
        
        cross_lon_label_bokeh.data['lon_label'] = [cross_lon_label_all_bokeh.data['lon_label_all'][lon_index]]
        cross_lon_label_bokeh.change.emit()
        
        var lon_value = cross_lon_value_all_bokeh.data['lon_value_all'][lon_index]
        map_lon_line_bokeh.data['lon'] = [lon_value, lon_value]
        map_lon_line_bokeh.change.emit()
    """) 
    lon_slider = Slider(start=0, end=style_parameter['nlon']-1, \
                          value=style_parameter['cross_lon_default_index'], step=1, \
                          width=style_parameter['cross_lon_plot_width'],\
                          title=style_parameter['lon_slider_title'], height=50)
    lon_slider.js_on_change('value', lon_slider_callback)
    lon_slider_callback.args['lon_index'] = lon_slider
    # ==============================
    output_file(filename,title=style_parameter['html_title'], mode=style_parameter['library_source'])
    left_column = Column(depth_slider, map_view, colorbar_fig, annotating_fig01, width=style_parameter['left_column_width'])
    
    right_column = Column(Column(lat_slider,cross_lat_view), Column(lon_slider, cross_lon_view), colorbar_fig_right, width=style_parameter['right_column_width'])
    layout = Row(left_column, right_column, height=800)
    save(layout)
if __name__ == '__main__':
    # parameters used to customize figures
    style_parameter = {}
    style_parameter['html_title'] = 'Cross-section Viewer'
    style_parameter['xlabel_fontsize'] = '12pt'
    style_parameter['xtick_label_fontsize'] = '12pt'
    style_parameter['title_font_size'] = '14pt'
    style_parameter['annotating_text_font_size'] = '12pt'
    style_parameter['map_view_ndepth'] = 54
    style_parameter['nlat'] = 30
    style_parameter['nlon'] = 30
    style_parameter['dlat'] = 1.
    style_parameter['dlon'] = 1.
    style_parameter['map_view_figure_lat_min'] = 25
    style_parameter['map_view_figure_lat_max'] = 58.5
    style_parameter['map_view_figure_lon_min'] = -127
    style_parameter['map_view_figure_lon_max'] = -97
    style_parameter['map_view_image_lat_min'] = 25
    style_parameter['map_view_image_lat_max'] = 55
    style_parameter['map_view_image_lon_min'] = -127
    style_parameter['map_view_image_lon_max'] = -97
    style_parameter['map_view_plot_width'] = 500
    style_parameter['map_view_plot_height'] = 550
    style_parameter['map_view_title'] = 'Shear Velocity Map'
    style_parameter['map_view_xlabel'] = 'Longitude (degree)'
    style_parameter['map_view_ylabel'] = 'Latitude (degree)'
    style_parameter['map_view_tools'] = ['save', 'crosshair']
    style_parameter['map_view_default_index'] = 15
    style_parameter['map_view_depth_label_lon'] = -117
    style_parameter['map_view_depth_label_lat'] = 56.5
    style_parameter['map_view_depth_box_lon'] = -112.5
    style_parameter['map_view_depth_box_lat'] = 57.2
    style_parameter['map_view_depth_box_width'] = 150
    style_parameter['map_view_depth_box_height'] = 30
    style_parameter['map_view_grid_width'] = 13
    style_parameter['map_view_grid_height'] = 12
    style_parameter['spread_factor'] = 2
    style_parameter['min_vs_range'] = 0.4
    #
    style_parameter['colorbar_title'] = 'Shear Velocity (km/s)'
    style_parameter['colorbar_plot_height'] = 70
    #
    style_parameter['depth_slider_title'] = 'Depth Index (drag to change depth)'
    #
    style_parameter['annotation_plot_width'] = 550
    style_parameter['annotation_plot_height'] = 150
    #
    style_parameter['cross_lat_plot_width'] = 550
    style_parameter['cross_lat_plot_height'] = 350
    style_parameter['cross_lat_default_index'] = 14
    style_parameter['lat_slider_title'] = 'Latitude Index (drag to change latitude)'
    style_parameter['cross_lat_view_label_lon'] = -114
    style_parameter['cross_lat_view_label_depth'] = -5
    style_parameter['cross_view_vs_min'] = 2.8
    style_parameter['cross_view_vs_max'] = 4.6
    style_parameter['cross_ddepth'] = 1.0
    style_parameter['cross_dlat'] = 0.25
    style_parameter['cross_dlon'] = 0.25
    style_parameter['cross_lat_view_title'] = 'Cross-sections Along Latitude'
    style_parameter['cross_lat_view_tools'] = ['save', 'crosshair']
    #
    style_parameter['cross_lat_view_xlabel'] = 'Longitude (degree)'
    style_parameter['cross_lat_view_ylabel'] = 'Depth (km)'
    #
    style_parameter['cross_lon_plot_width'] = 550
    style_parameter['cross_lon_plot_height'] = 350
    style_parameter['cross_lon_default_index'] = 14
    style_parameter['lon_slider_title'] = 'Longitude Index (drag to change longitude)'
    style_parameter['cross_lon_view_label_lat'] = 38
    style_parameter['cross_lon_view_label_depth'] = -5
    style_parameter['cross_lon_view_title'] = 'Cross-sections Along Longitude'
    style_parameter['cross_lon_view_tools'] = ['save', 'crosshair']
    #
    style_parameter['cross_lon_view_xlabel'] = 'Latitude (degree)'
    style_parameter['cross_lon_view_ylabel'] = 'Depth (km)'
    #
    style_parameter['annotating_html01'] = """<p style="font-size:16px">
        <b> Reference:</b> <br>
        Chai, C., Ammon, C.J., Maceira, M., Herrmann, R.B. 2015, Inverting interpolated receiver functions \
        with surface wave dispersion and gravity: Application to the western U.S. and adjacent Canada and Mexico, \
        Geophysical Research Letters, 42(11), 4359–4366, doi:10.1002/2015GL063733.</br>
        Chai, C., Ammon, C.J., Maceira, M., Herrmann, R.B., 2018. Interactive Visualization of Complex Seismic Data \
        and Models Using Bokeh. Seismol. Res. Lett. 89, 668–676. https://doi.org/10.1785/0220170132. </p>
        <b> Tips:</b> <br>
        Drag a slider to change the depth or the cross-section location. <br>
        The dashed lines show cross-section locations on the Shear Velocity Map. <br>
        Note the color scale is different."""
    #
    style_parameter['left_column_width'] = 600
    style_parameter['right_column_width'] = 600
    # inline for embeded libaries; CDN for online libaries
    style_parameter['library_source'] = 'inline' #'CDN'
    #
    style_parameter['vmodel_filename'] = './WUS-CAMH-2015/2015GL063733-ds02.txt'
    style_parameter['html_filename'] = 'cross_section_viewer.html'
    #
    initial_model_raw = read_3D_output_model(style_parameter['vmodel_filename'])
    model_3D = select_parameters(initial_model_raw)
    profile_data_all = prepare_profile_data(model_3D)
    # read boundary data
    boundary_data = read_boundary_data()
    # convert profile data into map view data
    map_data_all_slices, map_depth_all_slices, color_range_all_slices = \
            prepare_map_data(profile_data_all, ndepth=style_parameter['map_view_ndepth'])
    # convert profile data into cross-section along latitude
    cross_lat_data_all, lat_label_all = prepare_lat_cross_data(model_3D, style_parameter)
     # convert profile data into cross-section along longitude
    cross_lon_data_all, lon_label_all = prepare_lon_cross_data(model_3D, style_parameter)
    #
    plot_cross_section_bokeh(style_parameter['html_filename'], map_data_all_slices, map_depth_all_slices, \
                       color_range_all_slices, profile_data_all, cross_lat_data_all, lat_label_all, \
                       cross_lon_data_all, lon_label_all, boundary_data, style_parameter)