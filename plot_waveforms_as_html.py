# -*- coding: utf-8 -*-
# Waveform Interactive Viewer
# 
# by Chengping Chai, University of Tennessee, October 4, 2017
# 
# Version 1.4
#
# Updates:
#       V1.0, Chengping Chai, University of Tennessee, October 4, 2017
#       V1.1, Chengping Chai, University of Tennessee, October 5, 2017
#         some changes for bokeh 0.12.9
#       V1.2, Chengping Chai, University of Tennessee, October 6, 2017
#         minor changes
#       V1.3, Chengping Chai, University of Tennessee, December 2, 2017
#         added taper and filter
#       V1.4, Chengping Chai, Oak Ridge National Laboratory, December 31, 2018
#         minor changes to work with latest libraries.
#       V1.5, Chengping Chai, Oak Ridge National Laboratory, August 31, 2021
#         minor changes to work with latest libraries.
#
# This script is prepared for a paper named as Interactive Visualization of
#  Complex Seismic Data and Models Using Bokeh submitted to SRL.
#
# Requirement:
#       numpy 1.21.1
#       bokeh 2.3.2
#       obspy 1.2.2
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
from obspy import read
import glob
from utility import *
# ========================================================
def read_waveform_from_sac(folder, style_parameter, lat_min=-90,lat_max=90,\
                          lon_min=-180,lon_max=180):
    '''
    Read waveform data for one event from SAC files in a folder.
    
    Input:
        folder is the path to the folder
        lat_min (default is -90) is the minimum latitude
        lat_max (90) is the maximum latitude
        lon_min (-180) is the minimum longitude
        lon_max (180) is the maximum longitude
        
    Output:
        lat_list is a list of latitudes from SAC headers
        lon_list is a list of longitudes from SAC headers
        waveform_list is a list of wavefoms with waveforms from same location group together
    '''
    file_list = glob.glob(folder)
    sta_lat_list = []
    sta_lon_list = []
    event_id_list = []
    station_id_list = []
    data_list = []
    info_list = []
    freqmin = style_parameter['curve_filter_freqmin']
    freqmax = style_parameter['curve_filter_freqmax']
    taper_percentage = style_parameter['curve_taper_percentage']
    for a_file in file_list:
        tr = read(a_file)[0]
        tr.taper(taper_percentage, type='hann')
        tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        sta_lat = tr.stats.sac['stla']
        sta_lon = tr.stats.sac['stlo']
        sta_code = tr.stats.station
        if sta_lat >= lat_min and sta_lat <= lat_max \
           and sta_lon >= lon_min and sta_lon <= lon_max:
            sta_lat_list.append(sta_lat)
            sta_lon_list.append(sta_lon)
            station_id = '{0:.4f}_{1:.4f}_{2}'.format(sta_lat,sta_lon,sta_code)
            station_id_list.append(station_id)
            event_lat = tr.stats.sac['evla']
            event_lon = tr.stats.sac['evlo']
            event_id = '{0:.4f}_{1:.4f}'.format(event_lat,event_lon)
            event_id_list.append(event_id)
            data_list.append(tr.data)
            info = {}
            info['station'] = tr.stats.station
            info['network'] = tr.stats.network
            info['location'] = tr.stats.location
            info['channel'] = tr.stats.channel
            info['starttime'] = str(tr.stats.starttime)
            info['endtime'] = str(tr.stats.endtime)
            info['sampling_rate'] = tr.stats.sampling_rate
            info['delta'] = tr.stats.delta
            info['npts'] = tr.stats.npts
            info_list.append(info)
    #
    unique_event_id_list = list(set(event_id_list))
    if len(unique_event_id_list) > 1:
        print('More than one event is found in the folder, plotting the event with most data.')
    else:
        event_id = unique_event_id_list[0]
    #
    words = event_id.split('_')
    event_lat = float(words[0])
    event_lon = float(words[1])
    unique_station_id_list = list(set(station_id_list))
    waveform_list = [[None]]*len(unique_station_id_list)
    metadata_list = [[None]]*len(unique_station_id_list)
    station_lat_list = [None]*len(unique_station_id_list)
    station_lon_list = [None]*len(unique_station_id_list)
    for i in range(len(station_id_list)):
        station_id = station_id_list[i]
        index = unique_station_id_list.index(station_id)
        if station_lat_list[index] is None:
            waveform_list[index] = [list(data_list[i])]
            metadata_list[index] = [info_list[i]]
            station_lat_list[index] = sta_lat_list[i]
            station_lon_list[index] = sta_lon_list[i]
        else:
            waveform_list[index].append(list(data_list[i]))
            metadata_list[index].append(info_list[i])
    return station_lat_list, station_lon_list, event_lat, event_lon, \
            waveform_list, metadata_list
# ========================================================
def plot_waveform_bokeh(filename,waveform_list,metadata_list,station_lat_list,\
                       station_lon_list, event_lat, event_lon, boundary_data, style_parameter):
    xlabel_fontsize = style_parameter['xlabel_fontsize']
    #
    map_station_location_bokeh = ColumnDataSource(data=dict(map_lat_list=station_lat_list,\
                                                            map_lon_list=station_lon_list))
    dot_default_index = 0
    selected_dot_on_map_bokeh = ColumnDataSource(data=dict(lat=[station_lat_list[dot_default_index]],\
                                                           lon=[station_lon_list[dot_default_index]],\
                                                           index=[dot_default_index]))
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
                        boundary_data['country']['latitude'],color='gray',\
                        line_width=2, level='underlay', nonselection_line_alpha=1.0,\
                        nonselection_line_color='gray')
    # marine boundaries
    map_view.multi_line(boundary_data['marine']['longitude'],\
                        boundary_data['marine']['latitude'],color='gray',\
                        level='underlay', nonselection_line_alpha=1.0,\
                        nonselection_line_color='gray')
    # shoreline boundaries
    map_view.multi_line(boundary_data['shoreline']['longitude'],\
                        boundary_data['shoreline']['latitude'],color='gray',\
                        line_width=2, nonselection_line_alpha=1.0, level='underlay',
                        nonselection_line_color='gray')
    # state boundaries
    map_view.multi_line(boundary_data['state']['longitude'],\
                        boundary_data['state']['latitude'],color='gray',\
                        level='underlay', nonselection_line_alpha=1.0,\
                        nonselection_line_color='gray')
    #
    map_view.triangle('map_lon_list', 'map_lat_list', source=map_station_location_bokeh, \
                      line_color='gray', size=style_parameter['marker_size'], fill_color='black',\
                      selection_color='black', selection_line_color='gray',\
                      selection_fill_alpha=1.0,\
                      nonselection_fill_alpha=1.0, nonselection_fill_color='black',\
                      nonselection_line_color='gray', nonselection_line_alpha=1.0)
    map_view.triangle('lon','lat', source=selected_dot_on_map_bokeh,\
                      size=style_parameter['selected_marker_size'], line_color='black',fill_color='red')
    map_view.asterisk([event_lon], [event_lat], size=style_parameter['event_marker_size'], line_width=3, line_color='red', \
                      fill_color='red')
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
    # --------------------------------------------------------
    max_waveform_length = 0
    max_waveform_amp = 0
    ncurve = len(waveform_list)
    for a_sta in waveform_list:
        for a_trace in a_sta:
            if len(a_trace) > max_waveform_length:
                max_waveform_length = len(a_trace)
            if np.max(np.abs(a_trace)) > max_waveform_amp:
                max_waveform_amp = np.max(np.abs(a_trace))
    #
    plotting_list = []
    for a_sta in waveform_list:
        temp = []
        for a_trace in a_sta:
            if len(a_trace) < max_waveform_length:
                a_trace = np.append(a_trace,np.zeros([(max_waveform_length-len(a_trace)),1]))
            temp.append(list(a_trace))
        plotting_list.append(temp)
    #
    time_list = []
    for ista in range(len(plotting_list)):
        a_sta = plotting_list[ista]
        temp = []
        for itr in range(len(a_sta)):
            a_trace = a_sta[itr]
            delta = metadata_list[ista][itr]['delta']
            time = list(np.arange(len(a_trace))*delta)
            temp.append(time)
        #
        time_list.append(temp)
    #
    reftime_label_list = []
    channel_label_list = []
    for ista in range(len(metadata_list)):
        temp_ref = []
        temp_channel = []
        a_sta = metadata_list[ista]
        for a_trace in a_sta:
            temp_ref.append('Starting from '+a_trace['starttime'])
            temp_channel.append(a_trace['network']+'_'+a_trace['station']+'_'+a_trace['channel'])
        reftime_label_list.append(temp_ref)
        channel_label_list.append(temp_channel)
    # --------------------------------------------------------
    curve_fig01 = Figure(plot_width=style_parameter['curve_plot_width'], plot_height=style_parameter['curve_plot_height'], \
                       y_range=(-max_waveform_amp*1.05,max_waveform_amp*1.05), \
                       x_range=(0,max_waveform_length),\
                    tools=['save','box_zoom','ywheel_zoom','xwheel_zoom','reset','crosshair','pan']) 
    #
    curve_index = 0
    select_curve_data = plotting_list[dot_default_index][curve_index]
    select_curve_time = time_list[dot_default_index][curve_index]
    
    selected_curve_data_bokeh01 = ColumnDataSource(data=dict(time=select_curve_time,amp=select_curve_data))
    select_reftime_label = reftime_label_list[dot_default_index][curve_index]
    selected_reftime_label_bokeh01 = ColumnDataSource(data=dict(x=[style_parameter['curve_reftime_label_x']],\
                                                                y=[style_parameter['curve_reftime_label_y']],\
                                                                label=[select_reftime_label]))
    select_channel_label = channel_label_list[dot_default_index][curve_index]
    selected_channel_label_bokeh01 = ColumnDataSource(data=dict(x=[style_parameter['curve_channel_label_x']],\
                                                                y=[style_parameter['curve_channel_label_y']],\
                                                                label=[select_channel_label]))
    all_curve_data_bokeh = ColumnDataSource(data=dict(t=time_list, amp=plotting_list))
    all_reftime_label_bokeh = ColumnDataSource(data=dict(label=reftime_label_list))
    all_channel_label_bokeh = ColumnDataSource(data=dict(label=channel_label_list))
    # plot waveform
    curve_fig01.line('time','amp', source=selected_curve_data_bokeh01,\
                   line_color='black')
    # add refference time as a label
    curve_fig01.text('x', 'y', 'label', source=selected_reftime_label_bokeh01)
    # add channel label
    curve_fig01.text('x', 'y', 'label', source=selected_channel_label_bokeh01)
    # change style
    curve_fig01.title.text_font_size = style_parameter['title_font_size']
    curve_fig01.title.align = 'center'
    curve_fig01.title.text_font_style = 'normal'
    curve_fig01.xaxis.axis_label = style_parameter['curve_xlabel']
    curve_fig01.xaxis.axis_label_text_font_style = 'normal'
    curve_fig01.xaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig01.xaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig01.yaxis.axis_label = style_parameter['curve_ylabel']
    curve_fig01.yaxis.axis_label_text_font_style = 'normal'
    curve_fig01.yaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig01.yaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig01.toolbar.logo = None
    curve_fig01.toolbar_location = 'above'
    curve_fig01.toolbar_sticky = False
    # --------------------------------------------------------
    curve_fig02 = Figure(plot_width=style_parameter['curve_plot_width'], plot_height=style_parameter['curve_plot_height'], \
                       y_range=(-max_waveform_amp*1.05,max_waveform_amp*1.05), \
                       x_range=(0,max_waveform_length),\
                    tools=['save','box_zoom','ywheel_zoom','xwheel_zoom','reset','crosshair','pan']) 
    #
    curve_index = 1
    select_curve_data = plotting_list[dot_default_index][curve_index]
    select_curve_time = time_list[dot_default_index][curve_index]
    selected_curve_data_bokeh02 = ColumnDataSource(data=dict(time=select_curve_time,amp=select_curve_data))
    select_channel_label = channel_label_list[dot_default_index][curve_index]
    selected_channel_label_bokeh02 = ColumnDataSource(data=dict(x=[style_parameter['curve_channel_label_x']],\
                                                                y=[style_parameter['curve_channel_label_y']],\
                                                                label=[select_channel_label]))
    # plot waveform
    curve_fig02.line('time','amp', source=selected_curve_data_bokeh02,\
                   line_color='black')
    # add channel label
    curve_fig02.text('x', 'y', 'label', source=selected_channel_label_bokeh02)
    # change style
    curve_fig02.title.text_font_size = style_parameter['title_font_size']
    curve_fig02.title.align = 'center'
    curve_fig02.title.text_font_style = 'normal'
    curve_fig02.xaxis.axis_label = style_parameter['curve_xlabel']
    curve_fig02.xaxis.axis_label_text_font_style = 'normal'
    curve_fig02.xaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig02.xaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig02.yaxis.axis_label = style_parameter['curve_ylabel']
    curve_fig02.yaxis.axis_label_text_font_style = 'normal'
    curve_fig02.yaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig02.yaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig02.toolbar.logo = None
    curve_fig02.toolbar_location = 'above'
    curve_fig02.toolbar_sticky = False
    # --------------------------------------------------------
    curve_fig03 = Figure(plot_width=style_parameter['curve_plot_width'], plot_height=style_parameter['curve_plot_height'], \
                       y_range=(-max_waveform_amp*1.05,max_waveform_amp*1.05), \
                       x_range=(0,max_waveform_length),\
                    tools=['save','box_zoom','ywheel_zoom','xwheel_zoom','reset','crosshair','pan']) 
    #
    curve_index = 2
    select_curve_data = plotting_list[dot_default_index][curve_index]
    select_curve_time = time_list[dot_default_index][curve_index]
    selected_curve_data_bokeh03 = ColumnDataSource(data=dict(time=select_curve_time,amp=select_curve_data))
    select_channel_label = channel_label_list[dot_default_index][curve_index]
    selected_channel_label_bokeh03 = ColumnDataSource(data=dict(x=[style_parameter['curve_channel_label_x']],\
                                                                y=[style_parameter['curve_channel_label_y']],\
                                                                label=[select_channel_label]))
    # plot waveform
    curve_fig03.line('time','amp', source=selected_curve_data_bokeh03,\
                   line_color='black')
    # add channel label
    curve_fig03.text('x', 'y', 'label', source=selected_channel_label_bokeh03)
    # change style
    curve_fig03.title.text_font_size = style_parameter['title_font_size']
    curve_fig03.title.align = 'center'
    curve_fig03.title.text_font_style = 'normal'
    curve_fig03.xaxis.axis_label = style_parameter['curve_xlabel']
    curve_fig03.xaxis.axis_label_text_font_style = 'normal'
    curve_fig03.xaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig03.xaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig03.yaxis.axis_label = style_parameter['curve_ylabel']
    curve_fig03.yaxis.axis_label_text_font_style = 'normal'
    curve_fig03.yaxis.axis_label_text_font_size = xlabel_fontsize
    curve_fig03.yaxis.major_label_text_font_size = xlabel_fontsize
    curve_fig03.toolbar.logo = None
    curve_fig03.toolbar_location = 'above'
    curve_fig03.toolbar_sticky = False
    # --------------------------------------------------------
    map_station_location_js = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                                            map_station_location_bokeh=map_station_location_bokeh,\
                                                            selected_curve_data_bokeh01=selected_curve_data_bokeh01,\
                                                            selected_curve_data_bokeh02=selected_curve_data_bokeh02,\
                                                            selected_curve_data_bokeh03=selected_curve_data_bokeh03,\
                                                            selected_channel_label_bokeh01=selected_channel_label_bokeh01,\
                                                            selected_channel_label_bokeh02=selected_channel_label_bokeh02,\
                                                            selected_channel_label_bokeh03=selected_channel_label_bokeh03,\
                                                            selected_reftime_label_bokeh01=selected_reftime_label_bokeh01,\
                                                            all_reftime_label_bokeh=all_reftime_label_bokeh,\
                                                            all_channel_label_bokeh=all_channel_label_bokeh,\
                                                            all_curve_data_bokeh=all_curve_data_bokeh), code="""
    var inds = cb_obj.indices
    
    selected_dot_on_map_bokeh.data['index'] = [inds]
    var new_loc = map_station_location_bokeh.data
    
    selected_dot_on_map_bokeh.data['lat'] = [new_loc['map_lat_list'][inds]]
    selected_dot_on_map_bokeh.data['lon'] = [new_loc['map_lon_list'][inds]]
    
    selected_dot_on_map_bokeh.change.emit()
    
    selected_curve_data_bokeh01.data['t'] = all_curve_data_bokeh.data['t'][inds][0]
    selected_curve_data_bokeh01.data['amp'] = all_curve_data_bokeh.data['amp'][inds][0]

    selected_curve_data_bokeh01.change.emit()
    
    selected_curve_data_bokeh02.data['t'] = all_curve_data_bokeh.data['t'][inds][1]
    selected_curve_data_bokeh02.data['amp'] = all_curve_data_bokeh.data['amp'][inds][1]

    selected_curve_data_bokeh02.change.emit()
    
    selected_curve_data_bokeh03.data['t'] = all_curve_data_bokeh.data['t'][inds][2]
    selected_curve_data_bokeh03.data['amp'] = all_curve_data_bokeh.data['amp'][inds][2]

    selected_curve_data_bokeh03.change.emit()
    
    selected_reftime_label_bokeh01.data['label'] = [all_reftime_label_bokeh.data['label'][inds][0]]
    
    selected_reftime_label_bokeh01.change.emit()
    
    selected_channel_label_bokeh01.data['label'] = [all_channel_label_bokeh.data['label'][inds][0]]
    
    selected_channel_label_bokeh01.change.emit()
    
    selected_channel_label_bokeh02.data['label'] = [all_channel_label_bokeh.data['label'][inds][1]]
    
    selected_channel_label_bokeh02.change.emit()
    
    selected_channel_label_bokeh03.data['label'] = [all_channel_label_bokeh.data['label'][inds][2]]
    
    selected_channel_label_bokeh03.change.emit()
    """)
    #
    map_station_location_bokeh.selected.js_on_change('indices', map_station_location_js)
    #
    curve_slider_callback = CustomJS(args=dict(selected_dot_on_map_bokeh=selected_dot_on_map_bokeh,\
                                                map_station_location_bokeh=map_station_location_bokeh,\
                                                selected_curve_data_bokeh01=selected_curve_data_bokeh01,\
                                                selected_curve_data_bokeh02=selected_curve_data_bokeh02,\
                                                selected_curve_data_bokeh03=selected_curve_data_bokeh03,\
                                                selected_channel_label_bokeh01=selected_channel_label_bokeh01,\
                                                selected_channel_label_bokeh02=selected_channel_label_bokeh02,\
                                                selected_channel_label_bokeh03=selected_channel_label_bokeh03,\
                                                selected_reftime_label_bokeh01=selected_reftime_label_bokeh01,\
                                                all_reftime_label_bokeh=all_reftime_label_bokeh,\
                                                all_channel_label_bokeh=all_channel_label_bokeh,\
                                                all_curve_data_bokeh=all_curve_data_bokeh),code="""
    var inds = Math.round(cb_obj.value)
    
    selected_dot_on_map_bokeh.data['index'] = [inds]
    var new_loc = map_station_location_bokeh.data
    
    selected_dot_on_map_bokeh.data['lat'] = [new_loc['map_lat_list'][inds]]
    selected_dot_on_map_bokeh.data['lon'] = [new_loc['map_lon_list'][inds]]
    
    selected_dot_on_map_bokeh.change.emit()
    
    selected_curve_data_bokeh01.data['t'] = all_curve_data_bokeh.data['t'][inds][0]
    selected_curve_data_bokeh01.data['amp'] = all_curve_data_bokeh.data['amp'][inds][0]

    selected_curve_data_bokeh01.change.emit()
    
    selected_curve_data_bokeh02.data['t'] = all_curve_data_bokeh.data['t'][inds][1]
    selected_curve_data_bokeh02.data['amp'] = all_curve_data_bokeh.data['amp'][inds][1]

    selected_curve_data_bokeh02.change.emit()
    
    selected_curve_data_bokeh03.data['t'] = all_curve_data_bokeh.data['t'][inds][2]
    selected_curve_data_bokeh03.data['amp'] = all_curve_data_bokeh.data['amp'][inds][2]

    selected_curve_data_bokeh03.change.emit()
    
    selected_reftime_label_bokeh01.data['label'] = [all_reftime_label_bokeh.data['label'][inds][0]]
    
    selected_reftime_label_bokeh01.change.emit()
    
    selected_channel_label_bokeh01.data['label'] = [all_channel_label_bokeh.data['label'][inds][0]]
    
    selected_channel_label_bokeh01.change.emit()
    
    selected_channel_label_bokeh02.data['label'] = [all_channel_label_bokeh.data['label'][inds][1]]
    
    selected_channel_label_bokeh02.change.emit()
    
    selected_channel_label_bokeh03.data['label'] = [all_channel_label_bokeh.data['label'][inds][2]]
    
    selected_channel_label_bokeh03.change.emit()
    """)
    curve_slider = Slider(start=0, end=ncurve-1, value=style_parameter['curve_default_index'], \
                          step=1, title=style_parameter['curve_slider_title'], width=style_parameter['map_view_plot_width'],\
                          height=50)
    curve_slider.js_on_change('value', curve_slider_callback)
    curve_slider_callback.args['curve_index'] = curve_slider
    # ==============================
    # annotating text
    annotating_fig01 = Div(text=style_parameter['annotating_html01'], \
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    annotating_fig02 = Div(text=style_parameter['annotating_html02'],\
        width=style_parameter['annotation_plot_width'], height=style_parameter['annotation_plot_height'])
    # ==============================
    output_file(filename,title=style_parameter['html_title'],mode=style_parameter['library_source'])
    #
    left_fig = Column(curve_slider, map_view, annotating_fig01, width=style_parameter['left_column_width'] )
    
    right_fig = Column(curve_fig01, curve_fig02, curve_fig03, annotating_fig02, width=style_parameter['right_column_width'])
    layout = Row(left_fig, right_fig)
    save(layout)
# ========================================================
if __name__ == '__main__':
    # parameters used to customize figures 
    style_parameter = {}
    style_parameter['html_title'] = 'Waveform Viewer'
    style_parameter['xlabel_fontsize'] = '12pt'
    style_parameter['title_font_size'] = '14pt'
    style_parameter['marker_size'] = 15
    style_parameter['selected_marker_size'] = 20 
    style_parameter['event_marker_size'] = 20
    style_parameter['map_view_lat_min'] = 25.0
    style_parameter['map_view_lat_max'] = 45.0
    style_parameter['map_view_lon_min'] = -110.0
    style_parameter['map_view_lon_max'] = -86.0
    style_parameter['map_view_plot_width'] = (110-86)*25
    style_parameter['map_view_plot_height'] = (45-25)*30
    style_parameter['map_view_title'] = 'Station Map'
    style_parameter['map_view_tools'] = ['tap','save','crosshair']
    style_parameter['map_view_xlabel'] = 'Longitude (degree)'
    style_parameter['map_view_ylabel'] = 'Latitude (degree)'
    style_parameter['curve_title'] = 'Waveform'
    style_parameter['curve_xlabel'] = 'Time (s)'
    style_parameter['curve_ylabel'] = 'Displacement (m)'
    style_parameter['curve_plot_width'] = 500
    style_parameter['curve_plot_height'] = 250
    style_parameter['curve_default_index'] = 0
    style_parameter['curve_slider_title'] = 'Station Index (drag to change the location)'
    style_parameter['curve_reftime_label_x'] = 50
    style_parameter['curve_reftime_label_y'] = 0.0004
    style_parameter['curve_channel_label_x'] = 500
    style_parameter['curve_channel_label_y'] = 0.00005
    style_parameter['curve_filter_freqmin'] = 0.0125
    style_parameter['curve_filter_freqmax'] = 0.1
    style_parameter['curve_taper_percentage'] = 0.05 

    style_parameter['annotation_plot_width'] = 750
    style_parameter['annotation_plot_height'] = 150
    style_parameter['annotating_html01'] = """<p style="font-size:16px">
        <b> Reference:</b> <br>
        Chai, C., Ammon, C.J., Maceira, M., Herrmann, R.B., 2018. Interactive Visualization of Complex Seismic Data \
        and Models Using Bokeh. Seismol. Res. Lett. 89, 668–676. https://doi.org/10.1785/0220170132. <br>
        <b> Tips:</b> <br>
        Drag the slider to choose a station (triangle). <br>
        Click a triangle on the map to show waveforms at the location.
        </p>"""
    style_parameter['annotating_html02'] = """<p style="font-size:16px">
        </p>"""
    style_parameter['left_column_width'] = 750
    style_parameter['right_column_width'] = 680
    # inline for embeded libaries; CDN for online libaries
    style_parameter['library_source'] = 'inline' # 'CDN' 
    style_parameter['waveform_folder'] = './WaveformData/'
    style_parameter['html_filename'] =  'waveform_viewer.html'
    #
    boundary_data = read_boundary_data()
    #
    station_lat_list, station_lon_list, event_lat, event_lon, \
    waveform_list, metadata_list = read_waveform_from_sac(style_parameter['waveform_folder']+'/*.sac', \
        style_parameter)
    #
    plot_waveform_bokeh(style_parameter['html_filename'],waveform_list,metadata_list,station_lat_list,\
        station_lon_list, event_lat, event_lon, boundary_data, style_parameter)