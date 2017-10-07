# -*- coding: utf-8 -*-
# Dipserion Interactive Viewer
# 
# by Chengping Chai, Penn State, 2016
# 
# Version 1.2
#
# Updates:
#       V1.0  Chengping Chai, Penn State, 2016
#       V1.1, Chengping Chai, University of Tennessee, September 20, 2017
#         some changes for bokeh 0.12.9
#       V1.2, Chengping Chai, University of Tennessee, October 6, 2017
#         minor changes
#
# This script is prepared for a paper named as Interactive Visualization of  Complex Seismic Data and Models Using Bokeh submitted to SRL.
#
# Requirement:
#       numpy 1.10.4
#       bokeh 0.12.9
#
import numpy as np
from bokeh.plotting import Figure, output_file, save
from bokeh.plotting import ColumnDataSource
from bokeh.palettes import RdYlBu11 as palette
from bokeh.models.widgets import Slider
from bokeh.models import CustomJS
from bokeh.models import Column, Row
from bokeh.models import FixedTicker, PrintfTickFormatter
from bokeh.models.widgets import Div
from utility import *
# ========================================================
def read_dispersion_from_text(filename,ray_length_limit=100.,round_to=3,lat_min=25,lat_max=55,\
                             lon_min=-127,lon_max=-97):
    '''
    Read dispersion data from a text file.
    
    Input:
        filename is the path to the text file
        ray_length_limit (default is 100) is the minimum of acceptable ray path lengths
        round_to (3) specifies precision of latitude and longitude
        lat_min (25) is the minimum latitude
        lat_max (55) is the maximum latitude 
        lon_min (-127) is the minimum longitude
        lon_max (-97) is the maximum longitude
    
    Output:
        selected_lats is a list of latitudes for selected data points
        selected_lons is a list of longitudes for selected data points
        selected_vels is a list of group velocities for selected data points
        selected_rays is a list of ray path length for selected data points
    '''
    data = np.loadtxt(filename)
    lats = data[:,0]
    lons = data[:,1]
    vels = data[:,2]
    rays = data[:,3]
    selected_points = []
    selected_vels, selected_rays = [], []
    for i in range(len(rays)):
        ray_length = rays[i]
        lat = lats[i]
        lon = lons[i]
        if ray_length > ray_length_limit and lat > lat_min and lat < lat_max \
           and lon > lon_min and lon < lon_max:
            selected_points.append((np.round(lats[i],round_to),np.round(lons[i],round_to)))
            selected_vels.append(vels[i])
            selected_rays.append(rays[i])
    return selected_points, selected_vels, selected_rays
# ========================================================
def read_period_and_match_filename(period_filename,period_min=7,period_max=80):
    '''
    Read period from a text file and match each period with its dispersion data file
    
    Input:
        period_filename is the path to the text file that contains period values
        period_min (default is 7) is the minimum period accepted
        period_max (100) is the maximum period accepted
    
    Output:
        selected_period_array is a list of period values
        selected_vel_filename_list is a list of filenames of the dispersion data file that corresponds to each period
    '''
    period_array = np.loadtxt(period_filename)
    dir_path = '/'.join(period_filename.split('/')[:-1])
    nperiod = len(period_array)
    selected_period_array = []
    selected_vel_filename_list = []
    for i in range(nperiod):
        period = period_array[i]
        if period > period_min and period < period_max:
            filename = dir_path+'/vc_'+str(i).zfill(3)+'_001.xyz'
            selected_vel_filename_list.append(filename)
            selected_period_array.append(period)
    return selected_period_array, selected_vel_filename_list
# ========================================================
def read_all_period_data(period_filename):
    '''
    Read all acceptable dispersion data points
    
    Input:
        period_filename is the path to the text file that contains period values
        
    Output:
        period_array is an array of period values
        period_lat_array is a list of latitudes
        period_lon_array is a list of longitudes
        period_vel_array is a list of group velocities
    '''
    period_array, period_vel_filename_list = read_period_and_match_filename(period_filename)
    nperiod = len(period_array)
    period_point_array = []
    period_vel_array = []
    period_ray_array = []
    for i in range(nperiod):
        filename = period_vel_filename_list[i]
        points, vels, rays = read_dispersion_from_text(filename)
        period_point_array.append(points)
        period_vel_array.append(vels)
        period_ray_array.append(rays)
    return period_array,period_point_array, period_vel_array
# ========================================================
def get_unique_locations(period_point_array):
    '''
    Get unique locations (grid points) from surface-wave data
    
    Input:
        period_lat_array is an array of latitude
        period_lon_array is an array of longitude
    
    Output:
        point_list is an array of tuples with (lat, lon) as elements
    '''
    nperiod = len(period_point_array)
    unique_point_list = []
    for ip in range(nperiod):
        points = period_point_array[ip]
        npoint = len(points)
        for ipoint in range(npoint):
            point = points[ipoint]
            if point not in unique_point_list:
                unique_point_list.append(point)
    return unique_point_list
# ========================================================
def map_to_curve(unique_point_list, period_array, period_map_point_array, period_map_data_array):
    '''
    Convert dispersion map data into dispersion curves 
    
    Input:
        period_array is an array of period values
        period_map_point_array is an array of locations corresponding to period_map_data_array
        period_data_array is an array of dispersion values saved for map view
        
    Output:
        curve_point_array is an array of locations correspond to curve_data_array
        curve_data_array is a list of dispersion curves
    '''
    nperiod = len(period_array)
    curve_data_array = []
    npoint = len(unique_point_list)
    for ipoint in range(npoint):
        apoint = unique_point_list[ipoint]
        curve_data = []
        curve_period = []
        for iperiod in range(nperiod):
            period = period_array[iperiod]
            map_point = period_map_point_array[iperiod]
            map_data = period_map_data_array[iperiod]
            if apoint in map_point:
                point_index = map_point.index(apoint)
                curve_data.append(map_data[point_index])
                curve_period.append(period)
                
        #
        if curve_data:
            acurve = {}
            lat = apoint[0]
            lon = apoint[1]
            acurve['latitude'] = lat
            acurve['longitude'] = lon
            acurve['period'] = curve_period
            acurve['velocity'] = curve_data
            curve_data_array.append(acurve)
    return curve_data_array
# ========================================================
def plot_dispersion_bokeh(filename, period_array, curve_data_array, boundary_data, style_parameter):
    '''
    Plot dispersion maps and curves using bokeh
    
    Input:
        filename is the filename of the resulting html file
        period_array is a list of period
        curve_data_array is a list of dispersion curves
        boundary_data is a list of boundaries
        style_parameter contains plotting parameters 
    
    Output:
        None
        
    '''
    xlabel_fontsize = style_parameter['xlabel_fontsize']
    # ==============================
    # prepare data
    map_data_all_slices_velocity = []
    map_data_all_slices_period = []
    map_data_all_slices_color = []
    colorbar_data_all_left = []
    colorbar_data_all_right = []
    nperiod = len(period_array)
    ncurve = len(curve_data_array)
    ncolor = len(palette)
    palette_r = palette[::-1]
    colorbar_top = [0.1 for i in range(ncolor)]
    colorbar_bottom = [0 for i in range(ncolor)]
    for iperiod in range(nperiod):
        one_slice_lat_list = []
        one_slice_lon_list = []
        one_slice_vel_list = []
        
        map_period = period_array[iperiod]
        for icurve in range(ncurve):
            acurve = curve_data_array[icurve]
            curve_lat = acurve['latitude']
            curve_lon = acurve['longitude']
            curve_vel = acurve['velocity']
            curve_period = acurve['period']
            one_slice_lat_list.append(curve_lat)
            one_slice_lon_list.append(curve_lon)
            if map_period in curve_period:
                curve_period_index = curve_period.index(map_period)
                one_slice_vel_list.append(curve_vel[curve_period_index])
            else:
                one_slice_vel_list.append(style_parameter['nan_value'])
        # get color for dispersion values
        one_slice_vel_mean = np.nanmean(one_slice_vel_list)
        one_slice_vel_std = np.nanstd(one_slice_vel_list)
        
        color_min = one_slice_vel_mean - one_slice_vel_std * style_parameter['spread_factor']
        color_max = one_slice_vel_mean + one_slice_vel_std * style_parameter['spread_factor']
        color_step = (color_max - color_min)*1./ncolor
        one_slice_color_list = get_color_list(one_slice_vel_list,color_min,color_max,palette_r,\
                                             style_parameter['nan_value'],style_parameter['nan_color'])
        colorbar_left = np.linspace(color_min,color_max-color_step,ncolor)
        colorbar_right = np.linspace(color_min+color_step,color_max,ncolor)
        if one_slice_lat_list:
            map_data_all_slices_velocity.append(one_slice_vel_list)
            map_data_all_slices_period.append('Period: {0:6.1f} s'.format(map_period))
            map_data_all_slices_color.append(one_slice_color_list)
            colorbar_data_all_left.append(colorbar_left)
            colorbar_data_all_right.append(colorbar_right)
    # get location for all points
    map_lat_list, map_lon_list = [], []
    map_lat_label_list, map_lon_label_list = [], []
    for i in range(ncurve):
        acurve = curve_data_array[i]
        map_lat_list.append(acurve['latitude'])
        map_lon_list.append(acurve['longitude'])
        map_lat_label_list.append('Lat: {0:12.3f}'.format(acurve['latitude']))
        map_lon_label_list.append('Lon: {0:12.3f}'.format(acurve['longitude']))
    # data for the map view plot
    map_view_label_lon = style_parameter['map_view_period_label_lon']
    map_view_label_lat = style_parameter['map_view_period_label_lat']

    map_data_one_slice = map_data_all_slices_color[style_parameter['map_view_default_index']]
    map_data_one_slice_period = map_data_all_slices_period[style_parameter['map_view_default_index']]
    map_data_one_slice_bokeh = ColumnDataSource(data=dict(map_lat_list=map_lat_list,\
                                                          map_lon_list=map_lon_list,\
                                                          map_data_one_slice=map_data_one_slice))
    map_data_one_slice_period_bokeh = ColumnDataSource(data=dict(lat=[map_view_label_lat], lon=[map_view_label_lon],
                                                       map_period=[map_data_one_slice_period]))
    map_data_all_slices_bokeh = ColumnDataSource(data=dict(map_data_all_slices_color=map_data_all_slices_color,\
                                                          map_data_all_slices_period=map_data_all_slices_period))

    # data for the colorbar
    colorbar_data_one_slice = {}
    colorbar_data_one_slice['colorbar_left'] = colorbar_data_all_left[style_parameter['map_view_default_index']]
    colorbar_data_one_slice['colorbar_right'] = colorbar_data_all_right[style_parameter['map_view_default_index']]
    colorbar_data_one_slice_bokeh = ColumnDataSource(data=dict(colorbar_top=colorbar_top,colorbar_bottom=colorbar_bottom,
                                                               colorbar_left=colorbar_data_one_slice['colorbar_left'],\
                                                               colorbar_right=colorbar_data_one_slice['colorbar_right'],\
                                                               palette_r=palette_r))
    colorbar_data_all_slices_bokeh = ColumnDataSource(data=dict(colorbar_data_all_left=colorbar_data_all_left,\
                                                                colorbar_data_all_right=colorbar_data_all_right))
    # data for dispersion curves
    curve_default_index = style_parameter['curve_default_index']
    selected_dot_on_map_bokeh = ColumnDataSource(data=dict(lat=[map_lat_list[curve_default_index]],\
                                                     lon=[map_lon_list[curve_default_index]],\
                                                     color=[map_data_one_slice[curve_default_index]],\
                                                     index=[curve_default_index]))
    selected_curve_data = curve_data_array[curve_default_index]
    selected_curve_data_bokeh = ColumnDataSource(data=dict(curve_period=selected_curve_data['period'],\
                                                          curve_velocity=selected_curve_data['velocity']))

    period_all = []
    velocity_all = []
    for acurve in curve_data_array:
        period_all.append(acurve['period'])
        velocity_all.append(acurve['velocity'])
    curve_data_all_bokeh = ColumnDataSource(data=dict(period_all=period_all, velocity_all=velocity_all))
    
    selected_curve_lat_label_bokeh = ColumnDataSource(data=dict(x=[style_parameter['curve_lat_label_x']], \
                                                                y=[style_parameter['curve_lat_label_y']],\
                                                                lat_label=[map_lat_label_list[curve_default_index]]))
    selected_curve_lon_label_bokeh = ColumnDataSource(data=dict(x=[style_parameter['curve_lon_label_x']], \
                                                                y=[style_parameter['curve_lon_label_y']],\
                                                                lon_label=[map_lon_label_list[curve_default_index]]))
    all_curve_lat_label_bokeh = ColumnDataSource(data=dict(map_lat_label_list=map_lat_label_list))
    all_curve_lon_label_bokeh = ColumnDataSource(data=dict(map_lon_label_list=map_lon_label_list))
    # ==============================
    map_view = Figure(plot_width=style_parameter['map_view_plot_width'], \
                      plot_height=style_parameter['map_view_plot_height'], \
                      y_range=[style_parameter['map_view_lat_min'],\
                    style_parameter['map_view_lat_max']], x_range=[style_parameter['map_view_lon_min'],\
                    style_parameter['map_view_lon_max']], tools=style_parameter['map_view_tools'],\
                    title=style_parameter['map_view_title'])
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
    # add period label
    map_view.rect(style_parameter['map_view_period_box_lon'], style_parameter['map_view_period_box_lat'], \
                  width=style_parameter['map_view_period_box_width'], height=style_parameter['map_view_period_box_height'], \
                  width_units='screen',height_units='screen', color='#FFFFFF', line_width=1., line_color='black', level='underlay')
    map_view.text('lon', 'lat', 'map_period', source=map_data_one_slice_period_bokeh,\
                  text_font_size=style_parameter['annotating_text_font_size'],text_align='left',level='underlay')
    # ------------------------------
    # plot dots
    map_view.circle('map_lon_list', 'map_lat_list', color='map_data_one_slice', \
                    source=map_data_one_slice_bokeh, size=style_parameter['marker_size'],\
                    line_width=0.2, line_color='black', alpha=1.0,\
                    selection_color='map_data_one_slice', selection_line_color='black',\
                    selection_fill_alpha=1.0,\
                    nonselection_fill_alpha=1.0, nonselection_fill_color='map_data_one_slice',\
                    nonselection_line_color='black', nonselection_line_alpha=1.0)
    map_view.circle('lon', 'lat', color='color', source=selected_dot_on_map_bokeh, \
                    line_color='#00ff00', line_width=4.0, alpha=1.0, \
                    size=style_parameter['selected_marker_size'])
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
                      fill_color='palette_r',source=colorbar_data_one_slice_bokeh)
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
    curve_fig = Figure(plot_width=style_parameter['curve_plot_width'], plot_height=style_parameter['curve_plot_height'], \
                       y_range=(style_parameter['curve_y_min'],style_parameter['curve_y_max']), \
                       x_range=(style_parameter['curve_x_min'],style_parameter['curve_x_max']),x_axis_type='log',\
                        tools=['save','box_zoom','wheel_zoom','reset','crosshair','pan'],
                        title=style_parameter['curve_title'])
    # ------------------------------
    curve_fig.rect([style_parameter['curve_label_box_x']], [style_parameter['curve_label_box_y']], \
                   width=style_parameter['curve_label_box_width'], height=style_parameter['curve_label_box_height'], \
                   width_units='screen', height_units='screen', color='#FFFFFF', line_width=1., line_color='black', level='underlay')
    curve_fig.text('x', 'y', \
                   'lat_label', source=selected_curve_lat_label_bokeh)
    curve_fig.text('x', 'y', \
                   'lon_label', source=selected_curve_lon_label_bokeh)
    # ------------------------------
    curve_fig.line('curve_period', 'curve_velocity', source=selected_curve_data_bokeh, color='black')
    curve_fig.circle('curve_period', 'curve_velocity', source=selected_curve_data_bokeh, size=5, color='black')
    # ------------------------------
    curve_fig.title.text_font_size = style_parameter['title_font_size']
    curve_fig.title.align = 'center'
    curve_fig.title.text_font_style = 'normal'
    curve_fig.xaxis.axis_label = style_parameter['curve_xlabel']
    curve_fig.xaxis.axis_label_text_font_style = 'normal'
    curve_fig.xaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig.xaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig.yaxis.axis_label = style_parameter['curve_ylabel']
    curve_fig.yaxis.axis_label_text_font_style = 'normal'
    curve_fig.yaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig.yaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig.xgrid.grid_line_dash = [4, 2]
    curve_fig.ygrid.grid_line_dash = [4, 2]
    curve_fig.xaxis[0].formatter = PrintfTickFormatter(format="%4.0f")
    curve_fig.toolbar.logo = None
    curve_fig.toolbar_location = 'above'
    curve_fig.toolbar_sticky = False
    # ==============================
    map_data_one_slice_bokeh.callback = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                                          map_data_one_slice_bokeh=map_data_one_slice_bokeh,\
                                                          selected_curve_data_bokeh=selected_curve_data_bokeh,\
                                                          curve_data_all_bokeh=curve_data_all_bokeh,\
                                                          selected_curve_lat_label_bokeh=selected_curve_lat_label_bokeh,\
                                                          selected_curve_lon_label_bokeh=selected_curve_lon_label_bokeh,\
                                                          all_curve_lat_label_bokeh=all_curve_lat_label_bokeh,\
                                                          all_curve_lon_label_bokeh=all_curve_lon_label_bokeh), code="""
    
    var inds = Math.round(cb_obj.selected['1d'].indices)
    
    selected_dot_on_map_bokeh.data['index'] = [inds]
    
    var new_slice = map_data_one_slice_bokeh.data
    
    selected_dot_on_map_bokeh.data['lat'] = [new_slice['map_lat_list'][inds]]
    selected_dot_on_map_bokeh.data['lon'] = [new_slice['map_lon_list'][inds]]
    selected_dot_on_map_bokeh.data['color'] = [new_slice['map_data_one_slice'][inds]]
    
    selected_dot_on_map_bokeh.change.emit()
    
    selected_curve_data_bokeh.data['curve_period'] = curve_data_all_bokeh.data['period_all'][inds]
    selected_curve_data_bokeh.data['curve_velocity'] = curve_data_all_bokeh.data['velocity_all'][inds]
    
    selected_curve_data_bokeh.change.emit()
    
    var all_lat_labels = all_curve_lat_label_bokeh.data['map_lat_label_list']
    var all_lon_labels = all_curve_lon_label_bokeh.data['map_lon_label_list']
    
    selected_curve_lat_label_bokeh.data['lat_label'] = [all_lat_labels[inds]]
    selected_curve_lon_label_bokeh.data['lon_label'] = [all_lon_labels[inds]]
    
    selected_curve_lat_label_bokeh.change.emit()
    selected_curve_lon_label_bokeh.change.emit()
    """)
    # ==============================
    period_slider_callback = CustomJS(args=dict(map_data_all_slices_bokeh=map_data_all_slices_bokeh,\
                                  map_data_one_slice_bokeh=map_data_one_slice_bokeh,\
                                  colorbar_data_all_slices_bokeh=colorbar_data_all_slices_bokeh, \
                                  colorbar_data_one_slice_bokeh=colorbar_data_one_slice_bokeh,\
                                  selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                  map_data_one_slice_period_bokeh=map_data_one_slice_period_bokeh),\
                       code="""
    var p_index = Math.round(cb_obj.value)
    var map_data_all_slices = map_data_all_slices_bokeh.data
    
    
    var map_data_new_slice = map_data_all_slices['map_data_all_slices_color'][p_index]
    map_data_one_slice_bokeh.data['map_data_one_slice'] = map_data_new_slice
    map_data_one_slice_bokeh.change.emit()
    
    var color_data_all_slices = colorbar_data_all_slices_bokeh.data
    colorbar_data_one_slice_bokeh.data['colorbar_left'] = color_data_all_slices['colorbar_data_all_left'][p_index]
    colorbar_data_one_slice_bokeh.data['colorbar_right'] = color_data_all_slices['colorbar_data_all_right'][p_index]
    colorbar_data_one_slice_bokeh.change.emit()
    
    var selected_index = selected_dot_on_map_bokeh.data['index']
    selected_dot_on_map_bokeh.data['color'] = [map_data_new_slice[selected_index]]
    selected_dot_on_map_bokeh.change.emit()
    
    map_data_one_slice_period_bokeh.data['map_period'] = [map_data_all_slices['map_data_all_slices_period'][p_index]]
    map_data_one_slice_period_bokeh.change.emit()
    """)
    period_slider = Slider(start=0, end=nperiod-1, value=style_parameter['map_view_default_index'], \
                           step=1, title=style_parameter['period_slider_title'], \
                           width=style_parameter['period_slider_plot_width'],\
                           height=50, callback=period_slider_callback)
    
    # ==============================
    curve_slider_callback = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                              map_data_one_slice_bokeh=map_data_one_slice_bokeh,\
                                              selected_curve_data_bokeh=selected_curve_data_bokeh,\
                                              curve_data_all_bokeh=curve_data_all_bokeh,\
                                              selected_curve_lat_label_bokeh=selected_curve_lat_label_bokeh,\
                                              selected_curve_lon_label_bokeh=selected_curve_lon_label_bokeh,\
                                              all_curve_lat_label_bokeh=all_curve_lat_label_bokeh,\
                                              all_curve_lon_label_bokeh=all_curve_lon_label_bokeh),\
                                    code="""
    var c_index = Math.round(cb_obj.value)
    
    var one_slice = map_data_one_slice_bokeh.data
    
    selected_dot_on_map_bokeh.data['index'] = [c_index]
    selected_dot_on_map_bokeh.data['lat'] = [one_slice['map_lat_list'][c_index]]
    selected_dot_on_map_bokeh.data['lon'] = [one_slice['map_lon_list'][c_index]]
    selected_dot_on_map_bokeh.data['color'] = [one_slice['map_data_one_slice'][c_index]]
    
    selected_dot_on_map_bokeh.change.emit()
    
    selected_curve_data_bokeh.data['curve_period'] = curve_data_all_bokeh.data['period_all'][c_index]
    selected_curve_data_bokeh.data['curve_velocity'] = curve_data_all_bokeh.data['velocity_all'][c_index]
    
    selected_curve_data_bokeh.change.emit()
    
    var all_lat_labels = all_curve_lat_label_bokeh.data['map_lat_label_list']
    var all_lon_labels = all_curve_lon_label_bokeh.data['map_lon_label_list']
    
    selected_curve_lat_label_bokeh.data['lat_label'] = [all_lat_labels[c_index]]
    selected_curve_lon_label_bokeh.data['lon_label'] = [all_lon_labels[c_index]]
    
    selected_curve_lat_label_bokeh.change.emit()
    selected_curve_lon_label_bokeh.change.emit()
    """)
    curve_slider = Slider(start=0, end=ncurve-1, value=style_parameter['curve_default_index'], \
                          step=1, title=style_parameter['curve_slider_title'], width=style_parameter['curve_plot_width'],\
                          height=50, callback=curve_slider_callback)
    
    # ==============================
    # annotating text
    annotating_fig01 = Div(text=style_parameter['annotating_html01'], \
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    annotating_fig02 = Div(text=style_parameter['annotating_html02'],\
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    # ==============================
    output_file(filename,title=style_parameter['html_title'],mode=style_parameter['library_source'])
    left_fig = Column(period_slider, map_view, colorbar_fig, annotating_fig01,\
                    width=style_parameter['left_column_width'] )
    right_fig = Column(curve_slider, curve_fig, annotating_fig02, \
                    width=style_parameter['right_column_width'] )
    layout = Row(left_fig, right_fig)
    save(layout)
# ========================================================
if __name__ == '__main__':
    # parameters used to customize figures 
    style_parameter = {}
    style_parameter['html_title'] = 'Dipsersion Viewer'
    style_parameter['xlabel_fontsize'] = '12pt'
    style_parameter['xtick_label_fontsize'] = '12pt'
    style_parameter['title_font_size'] = '14pt'
    style_parameter['annotating_text_font_size'] = '12pt'
    style_parameter['marker_size'] = 10
    style_parameter['selected_marker_size'] = 20
    style_parameter['map_view_lat_min'] = 23.5
    style_parameter['map_view_lat_max'] = 58.5
    style_parameter['map_view_lon_min'] = -129.5
    style_parameter['map_view_lon_max'] = -94.5
    style_parameter['map_view_plot_width'] = 500
    style_parameter['map_view_plot_height'] = 550
    style_parameter['map_view_title'] = 'Dispersion Map'
    style_parameter['map_view_tools'] = ['tap','save','crosshair']
    style_parameter['map_view_xlabel'] = 'Longitude (degree)'
    style_parameter['map_view_ylabel'] = 'Latitude (degree)'
    style_parameter['map_view_default_index'] = 5
    style_parameter['map_view_period_label_lon'] = -117
    style_parameter['map_view_period_label_lat'] = 56.5
    style_parameter['map_view_period_box_lon'] = -112.5
    style_parameter['map_view_period_box_lat'] = 57.2
    style_parameter['map_view_period_box_width'] = 150
    style_parameter['map_view_period_box_height'] = 30
    style_parameter['nan_value'] = np.nan
    style_parameter['nan_color'] = '#808080'
    style_parameter['spread_factor'] = 3
    #
    style_parameter['colorbar_title'] = 'Rayleigh-Wave Group Velocity (km/s)'
    style_parameter['colorbar_plot_height'] = 70
    #
    style_parameter['period_slider_plot_width'] = 500
    style_parameter['period_slider_title'] = 'Period Index (drag to change the period)'
    #
    style_parameter['curve_default_index'] = 214
    style_parameter['curve_title'] = 'Dispersion Curve'
    style_parameter['curve_xlabel'] = 'Period (s)'
    style_parameter['curve_ylabel'] = 'Rayleigh-Wave Group Velocity (km/s)'
    style_parameter['curve_plot_width'] = 500
    style_parameter['curve_plot_height'] = 550
    style_parameter['curve_x_min'] = 6
    style_parameter['curve_x_max'] = 89
    style_parameter['curve_y_min'] = 2
    style_parameter['curve_y_max'] = 5
    style_parameter['curve_lat_label_x'] = 7
    style_parameter['curve_lat_label_y'] = 4.5
    style_parameter['curve_lon_label_x'] = 7
    style_parameter['curve_lon_label_y'] = 4.35
    style_parameter['curve_label_box_x'] = 10.
    style_parameter['curve_label_box_y'] = 4.48
    style_parameter['curve_label_box_width'] = 140
    style_parameter['curve_label_box_height'] = 50
    #
    style_parameter['curve_slider_title'] = 'Curve Index (drag to change the location)'
    #
    style_parameter['annotation_plot_width'] = 550
    style_parameter['annotation_plot_height'] = 150
    style_parameter['annotation_tools'] = []
    #
    style_parameter['annotating_html01'] = """<p style="font-size:16px">
        <b> References:</b> <br>
        Herrmann et al. (2013, <a href="http://www.eas.slu.edu/eqc/eqc_research/NATOMO/">Online</a>)</p>"""
    #
    style_parameter['annotating_html02'] = """<p style="font-size:16px">
        <b> Tips:</b> <br>
        Drag the sliders to change the period or the curve location. <br>
        Click a dot in the map to show a dispersion curve at the location."""
    #
    style_parameter['left_column_width'] = 650
    style_parameter['right_column_width'] = 650
    # inline for embeded libaries; CDN for online libaries
    style_parameter['library_source'] = 'inline' # 'CDN'  
    style_parameter['dispersion_folder'] = './RAYLU/'
    style_parameter['html_filename'] = 'dispersion_viewer.html'
    #
    # read dispersion data from text files, the data are saved for map view
    period_array, period_map_point_array, period_map_data_array = read_all_period_data(style_parameter['dispersion_folder']+'/per.uniq')
    # get unique locations from the data
    unique_point_list = get_unique_locations(period_map_point_array)
    # sort unique locations
    unique_point_list.sort(key=lambda tup: tup[0], reverse=True)
    # convert map-view data into curve-view data
    curve_data_array = map_to_curve(unique_point_list, period_array, period_map_point_array, period_map_data_array)
    # read boundary data
    boundary_data = read_boundary_data()
    # 
    
    # plot dispersion data using bokeh
    plot_dispersion_bokeh(style_parameter['html_filename'], period_array, curve_data_array, boundary_data, style_parameter)