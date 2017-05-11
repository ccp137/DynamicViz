# Velocity Model Interactive Viewer
# 
# by Chengping Chai, Penn State, 2016
# 
# Version 1.0
#
# This script is prepared for a paper named as Interactive Seismic Visualization Using HTML.
#
# Requirement:
#       numpy 1.10.4
#       scipy 0.17.0
#       bokeh 0.12.0
#
import numpy as np
from scipy import interpolate
from bokeh.plotting import Figure, output_file, save
from bokeh.palettes import RdYlBu11 as palette
from bokeh.plotting import ColumnDataSource
from bokeh.models import CustomJS
from bokeh.models import HBox, VBox
from bokeh.models.widgets import Slider, Button
from bokeh.models import FixedTicker, PrintfTickFormatter
from bokeh.models import Rect
from bokeh.models import Range1d
from bokeh.models.widgets import Div
from utility import *
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
    prd_vp = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_vs = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_thicks = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_rho = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_lons = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_lats = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_smooth = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_vsap = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_weight = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_ap = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_ae = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_tops = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_celln = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
    prd_geo = [[[0 for k in xrange(nz)] for j in xrange(ny)] for i in xrange(nx)]
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
    z_list = np.zeros((nlon,nlat))
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
        map_data_one_slice_clipped = clip_map_data(map_data_one_slice_interpolated, \
                                                  vs_min, vs_max)
        #
        map_data_all.append(map_data_one_slice_clipped['vs'])
    return map_data_all, map_depth_all, color_range_all_slices

def plot_3DModel_bokeh(filename, map_data_all_slices, map_depth_all_slices, \
                       color_range_all_slices, profile_data_all, boundary_data, \
                       style_parameter):
    '''
    Plot shear velocity maps and velocity profiles using bokeh

    Input:
        filename is the filename of the resulting html file
        map_data_all_slices contains the velocity model parameters saved for map view plots
        map_depth_all_slices is a list of depths
        color_range_all_slices is a list of color ranges
        profile_data_all constains the velocity model parameters saved for profile plots
        boundary_data is a list of boundaries
        style_parameter contains plotting parameters

    Output:
        None
    
    '''
    xlabel_fontsize = style_parameter['xlabel_fontsize']
    #
    colorbar_data_all_slices_left_right = []
    map_view_ndepth = style_parameter['map_view_ndepth']
    ncolor = len(palette)
    colorbar_top = [0.1 for i in range(ncolor)]
    colorbar_bottom = [0 for i in range(ncolor)]
    map_data_all_slices_depth = []
    for idepth in range(map_view_ndepth):
        colorbar_data_one_slice = {}
        color_min = color_range_all_slices[idepth][0]
        color_max = color_range_all_slices[idepth][1]
        color_step = (color_max - color_min)*1./ncolor
        colorbar_left = np.linspace(color_min,color_max-color_step,ncolor)
        colorbar_right = np.linspace(color_min+color_step,color_max,ncolor)
        colorbar_data_one_slice['colorbar_left'] = colorbar_left
        colorbar_data_one_slice['colorbar_right'] = colorbar_right
        colorbar_data_all_slices_left_right.append(colorbar_data_one_slice)
        map_depth = map_depth_all_slices[idepth]
        map_data_all_slices_depth.append('Depth: {0:8.0f} km'.format(map_depth))
    #
    # data for the colorbar
    colorbar_data_one_slice = colorbar_data_all_slices_left_right[style_parameter['map_view_default_index']]
    colorbar_data_one_slice_bokeh = ColumnDataSource(data=dict(colorbar_left=colorbar_data_one_slice['colorbar_left'],\
                                                               colorbar_right=colorbar_data_one_slice['colorbar_right']))
    colorbar_data_all_slices_left_right_bokeh = ColumnDataSource(data=dict(colorbar_data_all_slices_left_right=colorbar_data_all_slices_left_right))
    #
    map_view_label_lon = style_parameter['map_view_depth_label_lon']
    map_view_label_lat = style_parameter['map_view_depth_label_lat']
    map_data_one_slice_depth = map_data_all_slices_depth[style_parameter['map_view_default_index']]
    map_data_one_slice_depth_bokeh = ColumnDataSource(data=dict(lat=[map_view_label_lat], lon=[map_view_label_lon],
                                                           map_depth=[map_data_one_slice_depth]))
    palette_r = palette[::-1]
    #
    map_view_default_index = style_parameter['map_view_default_index']
    map_data_one_slice = map_data_all_slices[map_view_default_index]

    map_data_one_slice_bokeh = ColumnDataSource(data=dict(map_data_one_slice=[map_data_one_slice]))
    map_data_all_slices_bokeh = ColumnDataSource(data=dict(map_data_all_slices=map_data_all_slices,\
                                                           map_data_all_slices_depth=map_data_all_slices_depth))
    # ------------------------------
    nprofile = len(profile_data_all)
    grid_lat_list = []
    grid_lon_list = []
    for iprofile in range(nprofile):
        aprofile = profile_data_all[iprofile]
        grid_lat_list.append(aprofile['lat'])
        grid_lon_list.append(aprofile['lon'])
    grid_data_bokeh = ColumnDataSource(data=dict(lon=grid_lon_list,lat=grid_lat_list))
    profile_default_index = style_parameter['profile_default_index']
    selected_dot_on_map_bokeh = ColumnDataSource(data=dict(lat=[grid_lat_list[profile_default_index]], \
                                                           lon=[grid_lon_list[profile_default_index]], \
                                                           index=[profile_default_index]))
    # ------------------------------
    profile_vs_all = []
    profile_depth_all = []
    profile_ndepth = style_parameter['profile_ndepth']
    profile_lat_label_list = []
    profile_lon_label_list = []
    for iprofile in range(nprofile):
        aprofile = profile_data_all[iprofile]
        vs_raw = aprofile['vs']
        top_raw = aprofile['top']
        profile_lat_label_list.append('Lat: {0:12.1f}'.format(aprofile['lat']))
        profile_lon_label_list.append('Lon: {0:12.1f}'.format(aprofile['lon']))
        vs_plot = []
        depth_plot = []
        for idepth in range(profile_ndepth):
            vs_plot.append(vs_raw[idepth])
            depth_plot.append(top_raw[idepth])
            vs_plot.append(vs_raw[idepth])
            depth_plot.append(top_raw[idepth+1])
        profile_vs_all.append(vs_plot)
        profile_depth_all.append(depth_plot)
    profile_data_all_bokeh = ColumnDataSource(data=dict(profile_vs_all=profile_vs_all, \
                                                        profile_depth_all=profile_depth_all))
    selected_profile_data_bokeh = ColumnDataSource(data=dict(vs=profile_vs_all[profile_default_index],\
                                                             depth=profile_depth_all[profile_default_index]))
    selected_profile_lat_label_bokeh = ColumnDataSource(data=\
                                dict(lat_label=[profile_lat_label_list[profile_default_index]]))
    selected_profile_lon_label_bokeh = ColumnDataSource(data=\
                                dict(lon_label=[profile_lon_label_list[profile_default_index]]))
    all_profile_lat_label_bokeh = ColumnDataSource(data=dict(profile_lat_label_list=profile_lat_label_list))
    all_profile_lon_label_bokeh = ColumnDataSource(data=dict(profile_lon_label_list=profile_lon_label_list))
    #
    button_ndepth = style_parameter['button_ndepth']
    button_data_all = []
    for iprofile in range(nprofile):
        aprofile = profile_data_all[iprofile]
        bprofile = {}
        for akey in ['vs','vp','rho','top']:
            bprofile[akey] = aprofile[akey][:button_ndepth]
        button_data_all.append(bprofile)
    button_data_all_bokeh = ColumnDataSource(data=dict(button_data_all=button_data_all))
    # ==============================
    map_view = Figure(plot_width=style_parameter['map_view_plot_width'], plot_height=style_parameter['map_view_plot_height'], \
                      tools=style_parameter['map_view_tools'], webgl=True, title=style_parameter['map_view_title'], \
                      y_range=[style_parameter['map_view_figure_lat_min'], style_parameter['map_view_figure_lat_max']],\
                      x_range=[style_parameter['map_view_figure_lon_min'], style_parameter['map_view_figure_lon_max']])
    #
    map_view.image('map_data_one_slice',x=[style_parameter['map_view_image_lon_min']],\
                   y=[style_parameter['map_view_image_lat_min']],dw=[style_parameter['nlon']],\
                   dh=[style_parameter['nlat']],palette=palette_r,\
                   source=map_data_one_slice_bokeh, level='image')

    depth_slider_callback = CustomJS(args=dict(map_data_one_slice_bokeh=map_data_one_slice_bokeh,\
                                               map_data_all_slices_bokeh=map_data_all_slices_bokeh,\
                                               colorbar_data_all_slices_left_right_bokeh=colorbar_data_all_slices_left_right_bokeh,\
                                               colorbar_data_one_slice_bokeh=colorbar_data_one_slice_bokeh,\
                                               map_data_one_slice_depth_bokeh=map_data_one_slice_depth_bokeh), code="""

        var d_index = depth_index.get('value')
        
        var map_data_all_slices = map_data_all_slices_bokeh.get('data')
        
        map_data_one_slice_bokeh.get('data')['map_data_one_slice'] = [map_data_all_slices['map_data_all_slices'][d_index]]
        map_data_one_slice_bokeh.trigger('change')
        
        var color_data_all_slices = colorbar_data_all_slices_left_right_bokeh.get('data')['colorbar_data_all_slices_left_right']
        var colorbar_data_new_slice = color_data_all_slices[d_index]
        colorbar_data_one_slice_bokeh.get('data')['colorbar_left'] = colorbar_data_new_slice['colorbar_left']
        colorbar_data_one_slice_bokeh.get('data')['colorbar_right'] = colorbar_data_new_slice['colorbar_right']
        colorbar_data_one_slice_bokeh.trigger('change')
        
        map_data_one_slice_depth_bokeh.get('data')['map_depth'] = [map_data_all_slices['map_data_all_slices_depth'][d_index]]
        map_data_one_slice_depth_bokeh.trigger('change')
        
    """) 
    depth_slider = Slider(start=0, end=style_parameter['map_view_ndepth']-1, \
                          value=map_view_default_index, step=1, \
                          width=style_parameter['map_view_plot_width'],\
                          title=style_parameter['depth_slider_title'], height=50, \
                          callback=depth_slider_callback)
    depth_slider_callback.args["depth_index"] = depth_slider
    # ------------------------------
    # add boundaries to map view
    # country boundaries
    map_view.multi_line(boundary_data['country']['longitude'],\
                        boundary_data['country']['latitude'],color='black',\
                        line_width=2, level='image')
    # marine boundaries
    map_view.multi_line(boundary_data['marine']['longitude'],\
                        boundary_data['marine']['latitude'],color='black',\
                        level='image')
    # shoreline boundaries
    map_view.multi_line(boundary_data['shoreline']['longitude'],\
                        boundary_data['shoreline']['latitude'],color='black',\
                        line_width=2, level='image')
    # state boundaries
    map_view.multi_line(boundary_data['state']['longitude'],\
                        boundary_data['state']['latitude'],color='black',\
                        level='image')
     # ------------------------------
    # add period label
    map_view.rect(style_parameter['map_view_depth_box_lon'], style_parameter['map_view_depth_box_lat'], \
                  width=style_parameter['map_view_depth_box_width'], height=style_parameter['map_view_depth_box_height'], \
                  width_units='screen',height_units='screen', color='#FFFFFF', line_width=1., line_color='black', level='image')
    map_view.text('lon', 'lat', 'map_depth', source=map_data_one_slice_depth_bokeh,\
                  text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='image')
    # ------------------------------
    map_view.rect('lon', 'lat', width=style_parameter['map_view_grid_width'], \
                  width_units='screen', height=style_parameter['map_view_grid_height'], \
                  height_units='screen', line_color='gray', line_alpha=0.5, \
                  selection_line_color='gray', selection_line_alpha=0.5, selection_fill_color=None,\
                  nonselection_line_color='gray',nonselection_line_alpha=0.5, nonselection_fill_color=None,\
                  source=grid_data_bokeh, color=None, line_width=1, level='glyph')
    map_view.rect('lon', 'lat',width=style_parameter['map_view_grid_width'], \
                  width_units='screen', height=style_parameter['map_view_grid_height'], \
                  height_units='screen', line_color='#00ff00', line_alpha=1.0, \
                  source=selected_dot_on_map_bokeh, fill_color=None, line_width=3.,level='glyph')
    # ------------------------------
    grid_data_bokeh.callback = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh, \
                                                  grid_data_bokeh=grid_data_bokeh,\
                                                  profile_data_all_bokeh=profile_data_all_bokeh,\
                                                  selected_profile_data_bokeh=selected_profile_data_bokeh,\
                                                  selected_profile_lat_label_bokeh=selected_profile_lat_label_bokeh,\
                                                  selected_profile_lon_label_bokeh=selected_profile_lon_label_bokeh, \
                                                  all_profile_lat_label_bokeh=all_profile_lat_label_bokeh, \
                                                  all_profile_lon_label_bokeh=all_profile_lon_label_bokeh, \
                                                 ), code="""
        
        var inds = cb_obj.get('selected')['1d'].indices
        
        var grid_data = grid_data_bokeh.get('data')
        selected_dot_on_map_bokeh.get('data')['lat'] = [grid_data['lat'][inds]]
        selected_dot_on_map_bokeh.get('data')['lon'] = [grid_data['lon'][inds]]
        selected_dot_on_map_bokeh.get('data')['index'] = [inds]
        selected_dot_on_map_bokeh.trigger('change')
        
        var profile_data_all = profile_data_all_bokeh.get('data')
        selected_profile_data_bokeh.get('data')['vs'] = profile_data_all['profile_vs_all'][inds]
        selected_profile_data_bokeh.get('data')['depth'] = profile_data_all['profile_depth_all'][inds]
        selected_profile_data_bokeh.trigger('change')
        
        var all_profile_lat_label = all_profile_lat_label_bokeh.get('data')['profile_lat_label_list']
        var all_profile_lon_label = all_profile_lon_label_bokeh.get('data')['profile_lon_label_list']
        selected_profile_lat_label_bokeh.get('data')['lat_label'] = [all_profile_lat_label[inds]]
        selected_profile_lon_label_bokeh.get('data')['lon_label'] = [all_profile_lon_label[inds]]
        selected_profile_lat_label_bokeh.trigger('change')
        selected_profile_lon_label_bokeh.trigger('change')
    """)
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
    colorbar_fig.quad(top=colorbar_top,bottom=colorbar_bottom,left='colorbar_left',right='colorbar_right',\
                  color=palette_r,source=colorbar_data_one_slice_bokeh)
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
    profile_xrange = Range1d(start=style_parameter['profile_plot_xmin'], end=style_parameter['profile_plot_xmax'])
    profile_yrange = Range1d(start=style_parameter['profile_plot_ymax'], end=style_parameter['profile_plot_ymin'])
    profile_fig = Figure(plot_width=style_parameter['profile_plot_width'], plot_height=style_parameter['profile_plot_height'],\
                         x_range=profile_xrange, y_range=profile_yrange, tools=style_parameter['profile_tools'],\
                         title=style_parameter['profile_title'])
    profile_fig.line('vs','depth',source=selected_profile_data_bokeh, line_width=2, line_color='black')
    # ------------------------------
    # add lat, lon
    profile_fig.rect([style_parameter['profile_label_box_x']], [style_parameter['profile_label_box_y']],\
                     width=style_parameter['profile_label_box_width'], height=style_parameter['profile_label_box_height'],\
                     width_units='screen', height_units='screen', color='#FFFFFF', line_width=1., line_color='black',\
                     level='underlay')
    profile_fig.text([style_parameter['profile_lat_label_x']], [style_parameter['profile_lat_label_y']], \
                     'lat_label', source=selected_profile_lat_label_bokeh)
    profile_fig.text([style_parameter['profile_lon_label_x']], [style_parameter['profile_lon_label_y']], \
                     'lon_label', source=selected_profile_lon_label_bokeh)
    # ------------------------------
    # change style
    profile_fig.xaxis.axis_label = style_parameter['profile_xlabel']
    profile_fig.xaxis.axis_label_text_font_style = 'normal'
    profile_fig.xaxis.axis_label_text_font_size = xlabel_fontsize
    profile_fig.xaxis.major_label_text_font_size = xlabel_fontsize
    profile_fig.yaxis.axis_label = style_parameter['profile_ylabel']
    profile_fig.yaxis.axis_label_text_font_style = 'normal'
    profile_fig.yaxis.axis_label_text_font_size = xlabel_fontsize
    profile_fig.yaxis.major_label_text_font_size = xlabel_fontsize
    profile_fig.xgrid.grid_line_dash = [4, 2]
    profile_fig.ygrid.grid_line_dash = [4, 2]
    profile_fig.title.text_font_size = style_parameter['title_font_size']
    profile_fig.title.align = 'center'
    profile_fig.title.text_font_style = 'normal'
    profile_fig.toolbar_location = 'above'
    profile_fig.toolbar_sticky = False
    profile_fig.toolbar.logo = None
    # ==============================
    profile_slider_callback = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                                 grid_data_bokeh=grid_data_bokeh, \
                                                 profile_data_all_bokeh=profile_data_all_bokeh, \
                                                 selected_profile_data_bokeh=selected_profile_data_bokeh,\
                                                 selected_profile_lat_label_bokeh=selected_profile_lat_label_bokeh,\
                                                 selected_profile_lon_label_bokeh=selected_profile_lon_label_bokeh, \
                                                 all_profile_lat_label_bokeh=all_profile_lat_label_bokeh, \
                                                 all_profile_lon_label_bokeh=all_profile_lon_label_bokeh), code="""
        var p_index = profile_index.get('value')
        
        var grid_data = grid_data_bokeh.get('data')
        selected_dot_on_map_bokeh.get('data')['lat'] = [grid_data['lat'][p_index]]
        selected_dot_on_map_bokeh.get('data')['lon'] = [grid_data['lon'][p_index]]
        selected_dot_on_map_bokeh.get('data')['index'] = [p_index]
        selected_dot_on_map_bokeh.trigger('change')
        
        var profile_data_all = profile_data_all_bokeh.get('data')
        selected_profile_data_bokeh.get('data')['vs'] = profile_data_all['profile_vs_all'][p_index]
        selected_profile_data_bokeh.get('data')['depth'] = profile_data_all['profile_depth_all'][p_index]
        selected_profile_data_bokeh.trigger('change')
        
        var all_profile_lat_label = all_profile_lat_label_bokeh.get('data')['profile_lat_label_list']
        var all_profile_lon_label = all_profile_lon_label_bokeh.get('data')['profile_lon_label_list']
        selected_profile_lat_label_bokeh.get('data')['lat_label'] = [all_profile_lat_label[p_index]]
        selected_profile_lon_label_bokeh.get('data')['lon_label'] = [all_profile_lon_label[p_index]]
        selected_profile_lat_label_bokeh.trigger('change')
        selected_profile_lon_label_bokeh.trigger('change')
        
    """)
    profile_slider = Slider(start=0, end=nprofile-1, value=style_parameter['profile_default_index'], \
                           step=1, title=style_parameter['profile_slider_title'], \
                           width=style_parameter['profile_plot_width'], height=50,\
                           callback=profile_slider_callback)
    profile_slider_callback.args['profile_index'] = profile_slider
    # ==============================
    simple_text_button_callback = CustomJS(args=dict(button_data_all_bokeh=button_data_all_bokeh,\
                                                    selected_dot_on_map_bokeh=selected_dot_on_map_bokeh), \
                                           code="""
        var index = selected_dot_on_map_bokeh.get('data')['index']
        
        var button_data = button_data_all_bokeh.get('data')['button_data_all'][index]
        
        var csvContent = "data:text;charset=utf-8,"
        var i = 0
        var temp = csvContent
        temp += "# Layer Top (km)      Vs(km/s)    Vp(km/s)    Rho(g/cm^3) \\n"
        while(button_data['vp'][i]) {
            temp+=button_data['top'][i].toPrecision(6) + "    " + button_data['vs'][i].toPrecision(4) + "   " + \
                    button_data['vp'][i].toPrecision(4) + "   " + button_data['rho'][i].toPrecision(4) + "\\n"
            i = i + 1
        }
        var encodedUri = encodeURI(temp)
        window.open(encodedUri)
        
    """)

    simple_text_button = Button(label=style_parameter['simple_text_botton_label'], button_type='default', width=style_parameter['button_width'],\
                                callback=simple_text_button_callback)
    # ------------------------------
    model96_button_callback = CustomJS(args=dict(button_data_all_bokeh=button_data_all_bokeh,\
                                                    selected_dot_on_map_bokeh=selected_dot_on_map_bokeh), \
                                           code="""
        var index = selected_dot_on_map_bokeh.get('data')['index']
        var lat = selected_dot_on_map_bokeh.get('data')['lat']
        var lon = selected_dot_on_map_bokeh.get('data')['lon']
        
        var button_data = button_data_all_bokeh.get('data')['button_data_all'][index]
        
        var csvContent = "data:text;charset=utf-8,"
        var i = 0
        var temp = csvContent
        temp +=  "MODEL." + index + " \\n"
        temp +=  "ShearVelocityModel Lat: "+ lat +"  Lon: " + lon + "\\n"
        temp +=  "ISOTROPIC \\n"
        temp +=  "KGS \\n"
        temp +=  "SPHERICAL EARTH \\n"
        temp +=  "1-D \\n"
        temp +=  "CONSTANT VELOCITY \\n"
        temp +=  "LINE08 \\n"
        temp +=  "LINE09 \\n"
        temp +=  "LINE10 \\n"
        temp +=  "LINE11 \\n"
        temp +=  "      H(KM)   VP(KM/S)   VS(KM/S) RHO(GM/CC)     QP         QS       ETAP       ETAS      FREFP      FREFS \\n"
        while(button_data['vp'][i+1]) {
            var thickness = button_data['top'][i+1] - button_data['top'][i]
            temp+="      " +thickness.toPrecision(6) + "    " + button_data['vp'][i].toPrecision(4) + "      " + button_data['vs'][i].toPrecision(4) \
                 + "      " + button_data['rho'][i].toPrecision(4) + "     0.00       0.00       0.00       0.00       1.00       1.00" + "\\n"
            i = i + 1
        }
        var encodedUri = encodeURI(temp)
        window.open(encodedUri)                                   
    """)                                       
    model96_button = Button(label=style_parameter['model96_botton_label'], button_type='default', width=style_parameter['button_width'],\
                                callback=model96_button_callback)
    # ==============================
    # annotating text
    annotating_fig01 = Div(text=style_parameter['annotating_html01'], \
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    annotating_fig02 = Div(text=style_parameter['annotating_html02'],\
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    # ==============================
    output_file(filename,title=style_parameter['html_title'], mode=style_parameter['library_source'])
    left_column = VBox(depth_slider, map_view, colorbar_fig, annotating_fig01, width=style_parameter['left_column_width'])
    button_pannel = HBox(simple_text_button, model96_button)
    right_column = VBox(profile_slider, profile_fig, button_pannel, annotating_fig02, width=style_parameter['right_column_width'])
    layout = HBox(left_column, right_column)
    save(layout)
if __name__ == '__main__':
    style_parameter = {}
    style_parameter['html_title'] = 'Model Viewer'
    style_parameter['xlabel_fontsize'] = '12pt'
    style_parameter['xtick_label_fontsize'] = '12pt'
    style_parameter['title_font_size'] = '14pt'
    style_parameter['annotating_text_font_size'] = '12pt'
    style_parameter['marker_size'] = 10
    style_parameter['selected_marker_size'] = 20
    style_parameter['map_view_ndepth'] = 54
    style_parameter['nlat'] = 30
    style_parameter['nlon'] = 30
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
    style_parameter['map_view_tools'] = ['tap', 'save', 'crosshair']
    style_parameter['map_view_default_index'] = 15
    style_parameter['map_view_depth_label_lon'] = -117
    style_parameter['map_view_depth_label_lat'] = 56.5
    style_parameter['map_view_depth_box_lon'] = -112.5
    style_parameter['map_view_depth_box_lat'] = 57.2
    style_parameter['map_view_depth_box_width'] = 150
    style_parameter['map_view_depth_box_height'] = 30
    style_parameter['map_view_grid_width'] = 13
    style_parameter['map_view_grid_height'] = 12
    style_parameter['spread_factor'] = 3
    style_parameter['min_vs_range'] = 0.4
    #
    style_parameter['colorbar_title'] = 'Shear Velocity (km/s)'
    style_parameter['colorbar_plot_height'] = 70
    #
    style_parameter['profile_ndepth'] = 54
    style_parameter['profile_default_index'] = 380
    style_parameter['profile_plot_width'] = 500
    style_parameter['profile_plot_height'] = 550
    style_parameter['profile_plot_xmin'] = 0.
    style_parameter['profile_plot_xmax'] = 6.
    style_parameter['profile_plot_ymin'] = 0
    style_parameter['profile_plot_ymax'] = 100
    style_parameter['profile_tools'] = ['save','ywheel_zoom','xwheel_zoom','reset','crosshair','pan']
    style_parameter['profile_title'] = '1D Shear Velocity Profile'
    style_parameter['profile_xlabel'] = 'Shear Velocity (km/s)'
    style_parameter['profile_ylabel'] = 'Depth (km)'
    style_parameter['profile_lat_label_x'] = 0.2
    style_parameter['profile_lat_label_y'] = 20
    style_parameter['profile_lon_label_x'] = 0.2
    style_parameter['profile_lon_label_y'] = 25
    style_parameter['profile_label_box_x'] = 1
    style_parameter['profile_label_box_y'] = 20
    style_parameter['profile_label_box_width'] = 120
    style_parameter['profile_label_box_height'] = 45
    #
    style_parameter['depth_slider_title'] = "Depth Index (drag to change depth)"
    style_parameter['profile_slider_title'] = 'Profile Index (drag to change profile location)'
    #
    style_parameter['button_ndepth'] = 65
    width=style_parameter['button_width'] = 250
    style_parameter['simple_text_botton_label'] = 'Download the profile as a simple text file'
    style_parameter['model96_botton_label'] = 'Download the profile as the model96 format'
    #
    style_parameter['annotation_plot_width'] = 550
    style_parameter['annotation_plot_height'] = 150
    style_parameter['annotation_tools'] = []
    #
    style_parameter['annotating_html01'] = """<p style="font-size:16px">
        <b> References:</b> <br>
        Chai et al. (<a href="http://onlinelibrary.wiley.com/doi/10.1002/2015GL063733/full">GRL</a>, 2015, 
        <a href="http://eqseis.geosc.psu.edu/~cchai/01research/01westernUS.html"> Website</a>)</p>"""
    #
    style_parameter['annotating_html02'] = """<p style="font-size:16px">
        <b> Tips:</b> <br>
        Drag the sliders to change the depth or the profile location. <br>
        Click a box in the map to a profile at the location. <br>
        Click the buttons to download the selected velocity profile."""
    #
    style_parameter['left_column_width'] = 600
    style_parameter['right_column_width'] = 600
    # inline for embeded libaries; CDN for online libaries
    style_parameter['library_source'] = 'inline' #'CDN'
    #
    initial_model_raw = read_3D_output_model('./WUS-CAMH-2015/2015GL063733-ds02.txt')
    model_3D = select_parameters(initial_model_raw)
    profile_data_all = prepare_profile_data(model_3D)
    # read boundary data
    boundary_data = read_boundary_data()
    # convert profile data into map view data
    map_data_all_slices, map_depth_all_slices, color_range_all_slices = \
            prepare_map_data(profile_data_all, ndepth=style_parameter['map_view_ndepth'])
    #
    html_filename = 'model_viewer.html'
    #
    plot_3DModel_bokeh(html_filename, map_data_all_slices, map_depth_all_slices, \
                       color_range_all_slices, profile_data_all, boundary_data, \
                       style_parameter)