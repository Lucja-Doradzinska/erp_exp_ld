# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:12:34 2021

@author: Lucja Doradzinska <l.doradzinska@nencki.edu.pl>
"""

from collections import OrderedDict 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import scipy.stats as stats

import mne

from .files_mng import _make_dir, _save_log



def _load_evokeds(subjects, path, evokeds_folder, evokeds_labels):
    """Load instances of mne.Evoked and return them as dict.

    Parameters
    ----------
    subjects : list of str
        The list of subject ids.
    path : str
        A path to experiments folder.
    evokeds_folder : str
        The folder with evokeds data.
    evokeds_labels : list of str
        Labels of evokeds by conditions.
        
    Returns
    -------
    evokeds : OrderedDict
        The dictionary containing mne.Evoked instances.
    """  
    
    evokeds = OrderedDict()
    for lab in evokeds_labels:
        evokeds[lab] = mne.read_evokeds(path + evokeds_folder + lab + '_ave.fif')
    return evokeds



def _combine_labels (lab1, lab2):
    """Combine two labels with _.

    Parameters
    ----------
    lab1 : str
        First label.
    lab2 : str
        Second label.
        
    Returns
    -------
    label : str
        The list of mne.Evoked instances reflecting specified ERP component.
    """ 
    if lab1 == '':
        label = lab2
    elif lab2 == '':
        label = lab1
    else:
        label = lab1 + '_' + lab2
    return label



def _plot_erp (erp_wave, plot_name, plot_folder, plot_time_wind, time_wind, time_wind_color, plot_ci, ci, colors, styles, legend, 
               legend_size, legend_loc, fig_size, ylim, y_ticks, xlim, x_ticks, save_format):
    """Plot ERP waveforms and save it.

    Parameters
    ----------
    erp_wave : DataFrame
        ERP waveforms for conditions to plot.
    plot_name : str
        Plot name.
    plot_folder : string
        A folder to save plots.
    plot_time_wind : Bool
        Whether to plot ERP time-window.
    time_wind : list of float
        The list containing start and end time of the ERP in seconds, relative to the time-locked event.
    time_wind_color : color
        Color of wime-window.
        See matplotlib colors.
    plot_ci : bool
        Whether to plot ERP confidence interval.
    ci : DataFrame
        Confidence intervals for ERP waveforms.
    colors : list of colors
        Colors of plotted ERP waveforms.
        See matplotlib colors.
    styles : list of linestyles
        Linestyles of ERP waveforms.
        See matplotlib linestyles.
    legend : list of str,
        A list of labels.
    legend_size: int | {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        Legend font-size
    legend_loc : 2-tuple
        Localization of legend box.
        See matplotlib.legend.bbox_to_anchor.
    fig_size : list of floats
        The list containing width and height matplotlib figure
    ylim : list of floats
        The list containing min and max of y-axis
    y_ticks : list of floats
        The list of y-axis ticks.
    xlim : list of floats
        The list containing min and max of x-axis
    x_ticks : list of floats
        The list of x-axis ticks.
    save_format : str
        A format in which plots should be saved.
    """
    
    fig, ax = plt.subplots() 
    erp_wave.plot(ax = ax, linewidth = 2.5, style = styles, color = colors)
    fig.set_figheight(fig_size[1])
    fig.set_figwidth(fig_size[0])
    plt.axhline(y=0, color = 'k', linestyle='-', linewidth = 1)
    plt.axvline(x=0, color = 'k', linestyle='--', linewidth = 0.5)
    ax.set_xlabel('Time [ms]', fontsize='xx-large')
    ax.set_ylabel('Potential [µV]', fontsize='xx-large')
    if legend:
        ax.legend(fontsize=legend_size, bbox_to_anchor = legend_loc, loc='upper left')
    else:
        ax.get_legend().remove()
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_yticks(y_ticks)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    if plot_time_wind:
        ax.axvspan(time_wind[0], time_wind[1], color= time_wind_color, zorder = 0)
    if plot_ci:
        c = 0
        for contr in erp_wave:
            ax.fill_between(erp_wave.index, (erp_wave[contr] - ci[contr]), (erp_wave[contr] + ci[contr]), color = colors[c],
                            alpha=.1)
            c += 1
    plt.savefig(plot_folder + plot_name + '.' + save_format , format = save_format ) 



class Erp_waveform:
    """Class to compute ERP amplitudes and plot waveforms.
    
    Parameters
    ----------
    subjects : list of str
        The list of subject ids.
    path : str
        A path to experiments folder.
    evokeds_folder : str
         A folder with evoked data.
    evokeds_labels : list of str
        The list of conditions of evoked data.
        
    Attributes
    ----------
    evokeds : dict
        The dictionary holding mne.Evoked instances.
    """
    
    def __init__(self, subjects, path, evokeds_folder, evokeds_labels):
        """Init it."""
        
        self.subjects = subjects
        self.path = path
        self.evokeds = _load_evokeds(subjects, path, evokeds_folder, evokeds_labels)



    def _get_avg_evoked(self, erp_labs):
        """Average evokeds across conditions, pool channels and crop to specified time-window.
    
        Parameters
        ----------
        erp_labs : list of str
            Labels of conditions to average.
            
        Returns
        -------
        avg_evoked : list of mne.Evoked
            The list of mne.Evoked instances reflecting specified ERP component.
        """ 
            
        avg_evoked = []
        evoked_holder = {}
        for lab in erp_labs:
            evoked_holder[lab] = self.evokeds[lab].copy()
        for i in range(len(self.subjects)):
            avg_evoked.append(mne.grand_average([evoked_holder[lab][i] for lab in erp_labs]))
    
        return avg_evoked

    

    def _get_erp_amp (self, erp_lab_lists, channels, time_wind):
        """Get the amplitude of ERP component.
    
        Parameters
        ----------
        erp_lab_list : list of lists
            The list of Labels of conditions to average.
        channels : list of str | list od lists
            Channels to pool.
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            
        Returns
        -------
        erp_amp : float
            Averaged ERP amplitude extracted from data.
        """ 
        
        rois = []
        for i, erp_labs in enumerate(erp_lab_lists):
            avg_evoked = self._get_avg_evoked(erp_labs)
            rois.append([evoked.copy().crop(time_wind[0]/1000, time_wind[1]/1000).pick_channels(channels[i]) 
                         for evoked in avg_evoked])
        if len(rois) == 1:
            arr_data = np.array([subj.data for subj in rois[0]])
        else:
            roi_list = []
            for roi in rois:
                roi_list.append([subj.data for subj in roi])
            arr_data = np.mean(np.array(roi_list), axis = 0)
        erp_amp = np.mean(np.mean(arr_data, axis = 1), axis = 1) * 1e6
        return erp_amp
    


    def calc_central_erp_amp (self, erp_folder, erp_name, erp_labels, channels = [], time_wind = [], save_xlsx = True, 
                              save_csv = True): 
        """Calculate mean ERP amplitude.

        Parameters
        ----------
        erp_folder : string
            A folder to save amplitudes.
        erp_name : str
            The name of ERP component.
        erp_labels : dict
            Conditions specified as dicts of contrasts.
        channels : list of str
            Channels to pool.
            Defaults to [].
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        save_xlsx : bool
            Whether to save ERP amplitudes in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save ERP amplitudes in csv format.
            Defaults to False.
            
        Returns
        -------
        erp_df  : pd.DataFrame
            Log with mean ERP amplitudes per condition.
        """
        
        _make_dir(self.path, erp_folder)    
        erp_amps = OrderedDict()
        erp_amps['subjects'] = self.subjects
        for cond in erp_labels:
            for contr in erp_labels[cond]:
                label = _combine_labels(cond, contr)
                erp_amps[label] = self._get_erp_amp([erp_labels[cond][contr]], [channels], time_wind)
        _save_log(erp_amps, self.path + erp_folder, erp_name + '_amp', save_xlsx, save_csv) 
        erp_df = pd.DataFrame(erp_amps, columns = list(erp_amps.keys())) 
        return erp_df   
    
    
    
    def calc_lateral_erp_amp (self, erp_folder, erp_name, erp_labels_left, erp_labels_right, channels_left = [], 
                              channels_right = [], time_wind = [], save_xlsx = True, save_csv = True): 
        """Calculate mean lateralized ERP amplitude.

        Parameters
        ----------
        erp_folder : string
            A folder to save amplitudes.
        erp_labels_left : dict
            Conditions specified as dicts of contrasts with left presentations.
        erp_labels_right : dict
            Conditions specified as dicts of contrasts with right presentations.
        channels_left : list of str
            Left channels to pool.
            Defaults to [].
        channels_right : list of str
            Right channels to pool.
            Defaults to [].
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        save_xlsx : bool
            Whether to save ERP amplitudes in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save ERP amplitudes in csv format.
            Defaults to False.
            
        Returns
        -------
        erp_df  : pd.DataFrame
            Log with mean ipsi and contra-lateral ERP amplitudes per condition.
        """
        
        _make_dir(self.path, erp_folder)    
        erp_amps = OrderedDict()
        erp_amps['subjects'] = self.subjects
        for cond in erp_labels_left:
            for contr in erp_labels_left[cond]:
                label = _combine_labels(cond, contr)
                lab_ipsi = _combine_labels(label, 'ipsi')   
                lab_contra = _combine_labels(label, 'contra')   
                erp_amps[lab_ipsi] = self._get_erp_amp([erp_labels_left[cond][contr], erp_labels_right[cond][contr]], 
                                                       [channels_left, channels_right], time_wind)
                erp_amps[lab_contra] = self._get_erp_amp([erp_labels_left[cond][contr], erp_labels_right[cond][contr]], 
                                                         [channels_right, channels_left], time_wind)
        _save_log(erp_amps, self.path + erp_folder, erp_name + '_amp', save_xlsx, save_csv)   
        erp_df = pd.DataFrame(erp_amps, columns = list(erp_amps.keys()))
        return erp_df 
    
    
    
    def mean_topo(self, plot_folder, plot_name, erp_labels, time_min = 0, time_max = 0.7, time_step = 0.05, volt_min = -10, 
                  volt_max = 10, cmap = 'RdBu_r', save_format = 'png'):
        """Plot ERP amplitudes averaged acrss conditions on a topographic map.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        plot_name : str
            The name of the plot.
        erp_labels : dict
            Conditions specified as dicts of contrasts to plot.
        time_min : float
            The first timepoint to plot in seconds.
            Default is to 0 
        time_max : float
            The last timepoint to plot in seconds.
            Default is to 0.7.
        time_step : float
            The step for subsequent timepoints to plot in seconds.
            Default is to 0.05. 
        volt_min : float
            Minimal values to plot in microvolts.
            Default is to -10.
        volt_max : float
            Maximal values to plot in microvolts.
            Default is to 10.
        cmap : str
            Colormap, see matplotlib
            Default is to 'RdBu_r'.
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)   
        for cond in erp_labels:
            evokeds_to_plot = OrderedDict()
            for contr in erp_labels[cond]:
                evokeds_to_plot[contr] = self._get_avg_evoked(erp_labels[cond][contr])
            mean = mne.grand_average([mne.grand_average(evokeds_to_plot[contr]) for contr in erp_labels[cond]])
            mean.plot_topomap(np.arange(time_min, time_max, time_step), ch_type='eeg',
                              vmin = volt_min, vmax = volt_max, cmap = cmap, time_unit='s',
                              title = cond + ' signal averaged over stimuli types')
            plt.savefig(plot_folder + plot_name + '_' + cond + '_mean_topo.' + save_format, format = save_format)
    
    
    
    def contr_topo(self, plot_folder, plot_name, erp_labels, time_min = 0, time_max = 0.7, time_step = 0.05, volt_min = -10, 
                   volt_max = 10, cmap = 'RdBu_r', save_format = 'png'):
        """Plot ERP amplitudes on a topographic map.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        plot_name : str
            The name of the plot.
        erp_labels : dict
            Conditions specified as dicts of contrasts.
        time_min : float
            The first timepoint to plot in seconds.
            Default is to 0 
        time_max : float
            The last timepoint to plot in seconds.
            Default is to 0.7.
        time_step : float
            The step for subsequent timepoints to plot in seconds.
            Default is to 0.05. 
        volt_min : float
            Minimal values to plot in microvolts.
            Default is to -10.
        volt_max : float
            Maximal values to plot in microvolts.
            Default is to 10.
        cmap : str
            Colormap, see matplotlib
            Default is to 'RdBu_r'.
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder) 
        for cond in erp_labels:
            for contr in erp_labels[cond]:
                stim = self._get_avg_evoked(erp_labels[cond][contr])
                stim.plot_topomap(np.arange(time_min, time_max, time_step), ch_type='eeg',
                                  vmin = volt_min, vmax = volt_max, cmap = cmap, time_unit='s',
                                  title = cond + ' ' + contr)
                plt.savefig(self.path + plot_folder + plot_name + '_' + contr + '_topo.' + save_format, 
                            format = save_format)



    def diff_topo(self, plot_folder, plot_name, erp_labels, time_min = 0, time_max = 0.7, time_step = 0.05, volt_min = -3, 
                  volt_max = 3, cmap = 'RdBu_r', save_format = 'png'):
        """Plot differential ERP amplitudes on a topographic map.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        plot_name : str
            The name of the plot.
        erp_labels : dict
            Conditions specified as dicts of contrasts to plot.
            Each contition shoud contain two keys.
        time_min : float
            The first timepoint to plot in seconds.
            Default is to 0 
        time_max : float
            The last timepoint to plot in seconds.
            Default is to 0.7.
        time_step : float
            The step for subsequent timepoints to plot in seconds.
            Default is to 0.05. 
        volt_min : float
            Minimal values to plot in microvolts.
            Default is to -3.
        volt_max : float
            Maximal values to plot in microvolts.
            Default is to 3.
        cmap : str
            Colormap, see matplotlib
            Default is to 'RdBu_r'.
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)  
        for cond in erp_labels:
            if len(erp_labels[cond]) > 2:
                print('Cannot compare more than 2 conditions')
            else:
                stim_array = []
                for contr in erp_labels[cond]:
                    stim_erp = self._get_avg_evoked(erp_labels[cond][contr])
                    stim_array.append(mne.grand_average(stim_erp))
                diff = mne.combine_evoked (stim_array, weights=[1,-1])
                diff.plot_topomap(np.arange(time_min, time_max, time_step), ch_type='eeg',
                              vmin = volt_min, vmax = volt_max, cmap = cmap, time_unit='s',
                              title = ' difference between ' + list(erp_labels[cond].keys())[0] + ' and ' + list(erp_labels[cond].keys())[1])
            plt.savefig(self.path + plot_folder + plot_name + '_' + cond + '_diff_topo.' + save_format, 
                        format = save_format) 
    
    
    
    def lat_diff_topo(self, plot_folder, plot_name, erp_labels_left, erp_labels_right, time_min = 0, time_max = 0.7, 
                      time_step = 0.05, volt_min = 0, volt_max = 3, cmap = 'Blues', save_format = 'png'):
        """Plot diferrential lateralized ERP amplitude on a topographic map.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        plot_name : str
            The name of the plot.
        erp_labels_left : dict
            Conditions specified as dict of contitions to subtract. 
            Each condition should contain 2 keys.
        erp_labels_right : dict
            Conditions specified as dict of contitions to subtract.
            Each condition should contain 2 keys.
        time_min : float
            The first timepoint to plot in seconds.
            Default is to 0 
        time_max : float
            The last timepoint to plot in seconds.
            Default is to 0.7.
        time_step : float
            The step for subsequent timepoints to plot in seconds.
            Default is to 0.05. 
        volt_min : float
            Minimal values to plot in microvolts.
            Default is to 0.
        volt_max : float
            Maximal values to plot in microvolts.
            Default is to 3.
        cmap : str
            Colormap, see matplotlib
            Default is to 'RdBu_r'.
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)  
        for cond in erp_labels_left:
            if len(erp_labels_left[cond]) > 1:
                print('Cannot compare more than 2 conditions')
            else:
                for contr in erp_labels_left[cond]:
                    stim_erp_left = self._get_avg_evoked(erp_labels_left[cond][contr])
                    stim_erp_right = self._get_avg_evoked(erp_labels_right[cond][contr])
                    diff = mne.combine_evoked ([mne.grand_average(stim_erp_left), mne.grand_average(stim_erp_right)], 
                                               weights=[1,-1])
                    diff = diff.apply_function(np.abs)
                    diff.plot_topomap(np.arange(time_min, time_max, time_step), ch_type='eeg',
                                  vmin = volt_min, vmax = volt_max, cmap = cmap, time_unit='s',
                                  title = 'absolute difference between ' + cond + ' left and ' + cond + ' right')
            plt.savefig(self.path + plot_folder + plot_name + '_' + cond + '_diff_lat_topo.' + save_format, 
                        format = save_format) 
    
    
    
    def _get_erp_waves (self, erp_lab_list, channels):
        """Get ERP waveforms for each participant.
    
        Parameters
        ----------
        erp_lab_list : list of lists
            The list of labels of conditions to average.
        channels : list of str | list od lists
            Channels to pool.
        
        Returns
        -------
        erp : DataFrame
            ERP waveform per each subject
        """ 
     
        erp = pd.DataFrame()
        if len(erp_lab_list) == 1:
            avg_evoked = self._get_avg_evoked(erp_lab_list[0])
            for i, subj in enumerate(self.subjects):
                erp[subj] = avg_evoked[i].to_data_frame(channels[0], 'time').mean(axis = 1)   
        else:
            avg_evokeds = []
            for erp_labs in erp_lab_list:
                avg_evokeds.append(self._get_avg_evoked(erp_labs))
            for i, subj in enumerate(self.subjects):
                erp_holder = pd.DataFrame()
                for j, avg_evoked in enumerate(avg_evokeds):
                    erp_holder[str(j)] = avg_evoked[i].to_data_frame(channels[j], 'time').mean(axis = 1) 
                erp[subj] = erp_holder.mean(axis = 1)
        return erp
    
    
    
    def _central_waveforms(self, erp_contrasts, channels):
        """Calculate central ERP waveforms averaged across participants.
    
        Parameters
        ----------
        erp_contrasts : dict
            Conditions specified as lists of labels.
        channels : list of str
            Channels to pool.
        
        Returns
        -------
        erp_wave : DataFrame
            ERP waveforms for specified conditions.
        ci : DataFrame
            Confidence intervals for ERP waveforms.
        """
        
        erp_wave = pd.DataFrame()
        ci = pd.DataFrame()
        for contr in erp_contrasts:
            erp = self._get_erp_waves([erp_contrasts[contr]], [channels])
            erp_wave[contr] = erp.mean(axis = 1)
            ci[contr] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp.std(axis = 1) / np.sqrt(len(self.subjects))
        return erp_wave, ci


    
    def comp_plot (self, plot_folder, comp_name, erp_labels, channels, plot_ci = False, plot_time_wind = False, 
                   time_wind = [], time_wind_color = '#e2e2e2ff', colors = [], styles = [], legend = [''], legend_size = 16, legend_loc = (0.7, 1.15), fig_size = [8, 5.3], ylim = [-7, 9], 
                   y_ticks = [-6, -3, -0, 3, 6, 9], xlim = [-50, 700], x_ticks = [0, 100, 200, 300, 400, 500, 600, 700], 
                   save_format = 'png'):
        """Compare ERP waveforms on a plot.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        comp_name : str
            The name of ERP component.
        erp_labels : dict
            Conditions specified as dicts of contrasts to plot on one plot.
        channels : list of str
            Channels to pool.
        plot_ci : bool
            Whether to plot ERP confidence interval.
            Defaults to False.
        plot_time_wind : Bool
            Whether to plot ERP time-window.
            Defaults to False.
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        time_wind_color : color
            Color of wime-window.
            See matplotlib colors.
            Defaults to '#e2e2e2ff'.
        colors : list of colors
            Colors of plotted ERP waveforms.
            See matplotlib colors.
            Defaults to [].
        styles : list
            Linestyles of ERP waveforms.
            See matplotlib linestyles.
            Defaults to [].
        legend : list of str,
            A list of label.
            Defaults to [''].
        legend_size: int | {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            Legend font-size
            Defaults to 16.
        legend_loc : 2-tuple
            Localization of legend box.
            See matplotlib.legend.bbox_to_anchor.
            Defaults to (0.7, 1.15)
        fig_size : list of floats
            The list containing width and height matplotlib figure
            Defaults to [8, 5.3].
        ylim : list of floats
            The list containing min and max of y-axis
            Defaults to [-7, 9].
        y_ticks : list of floats
            The list of y-axis ticks.
            Defaults to [-6, -3, -0, 3, 6, 9].
        xlim : list of floats
            The list containing min and max of x-axis
            Defaults to [-50, 700].
        x_ticks : list of floats
            The list of x-axis ticks.
            Defaults to [0, 100, 200, 300, 400, 500, 600, 700].
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)
        for cond in erp_labels:
            erp_wave_dict, ci_dict = self._central_waveforms(erp_labels[cond], channels)     
            _plot_erp (erp_wave_dict, cond + '_' + comp_name, self.path + plot_folder, plot_time_wind, time_wind, time_wind_color,
                       plot_ci, ci_dict, colors, styles, cond in legend, legend_size, legend_loc, fig_size, ylim, y_ticks, xlim, x_ticks, save_format)
    
    
    
    def _diff_centr_waveforms(self, erp_labels, channels):
        """Calculate differential ERP waveforms averaged across participants.
    
        Parameters
        ----------
        erp_labels : dict
            Conditions specified as dict of contitions to subtract.
            Each condition should contain 2 keys.
        channels : list of str
            Channels to pool.
        
        Returns
        -------
        diff_wave : DataFrame
            ERP differential waveforms for specified conditions.
        ci : DataFrame
            Confidence intervals for differential ERP waveforms.
        """
        
        diff_wave = pd.DataFrame()
        ci = pd.DataFrame()
        for cond in erp_labels:
            contrasts = list(erp_labels[cond].keys())
            erp1 = self._get_erp_waves ([erp_labels[cond][contrasts[1]]], [channels])
            erp2 = self._get_erp_waves ([erp_labels[cond][contrasts[0]]], [channels])
            erp = erp1 - erp2
            diff_wave[cond] = erp.mean(axis = 1)
            ci[cond] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp.std(axis = 1) / np.sqrt(len(self.subjects))
        return diff_wave, ci



    def diff_plot (self, plot_folder, comp_name, erp_labels, channels, plot_ci = False, plot_time_wind = False, time_wind = [], time_wind_color = '#e2e2e2ff', colors = [], 
                   styles = [], legend = [''], legend_size = 16, legend_loc = (0.7, 1.15), fig_size = [8, 5.3], ylim = [-7, 9], y_ticks = [-6, -3, -0, 3, 6, 9], xlim = [-50, 700], 
                   x_ticks = [0, 100, 200, 300, 400, 500, 600, 700], save_format = 'png'):
        """Plot differential ERP waveforms.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        comp_name : str
            The name of ERP component.
        erp_labels : dict
            Conditions specified as dicts of conditions to subtract.
            Each condition should contain 2 keys.
        channels : list of str
            Channels to pool.
        plot_ci : bool
            Whether to plot ERP confidence interval.
            Defaults to False.
        plot_time_wind : Bool
            Whether to plot ERP time-window.
            Defaults to False.
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        time_wind_color : color
            Color of wime-window.
            See matplotlib colors.
            Defaults to '#e2e2e2ff'.
        colors : list of colors
            Colors of plotted ERP waveforms.
            See matplotlib colors.
            Defaults to [].
        styles : list
            Linestyles of ERP waveforms.
            See matplotlib linestyles.
            Defaults to [].
        legend : list of str,
            A list of label.
            Defaults to [''].
        legend_size: int | {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            Legend font-size
            Defaults to 16.
        legend_loc : 2-tuple
            Localization of legend box.
            See matplotlib.legend.bbox_to_anchor.
            Defaults to (0.7, 1.15)
        fig_size : list of floats
            The list containing width and height matplotlib figure
            Defaults to [8, 5.3].
        ylim : list of floats
            The list containing min and max of y-axis
            Defaults to [-7, 9].
        y_ticks : list of floats
            The list of y-axis ticks.
            Defaults to [-6, -3, -0, 3, 6, 9].
        xlim : list of floats
            The list containing min and max of x-axis
            Defaults to [-50, 700].
        x_ticks : list of floats
            The list of x-axis ticks.
            Defaults to [0, 100, 200, 300, 400, 500, 600, 700].
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder) 
        diff_wave, ci = self._diff_centr_waveforms(erp_labels, channels)
        _plot_erp (diff_wave, comp_name + '_differential_plots', self.path + plot_folder, plot_time_wind, time_wind, time_wind_color,
                   plot_ci, ci, colors, styles,  True, legend_size, legend_loc, fig_size, ylim, y_ticks, xlim, x_ticks, save_format)
        
        
        
    def _lateral_waveforms(self, erp_contrasts_left, erp_contrasts_right, channels_left, channels_right):
        """Calculate lateralized ERP waveforms averaged across participants.
    
        Parameters
        ----------
        erp_contrasts_left : dict
            Conditions with left presentations specified as lists of labels.
        erp_contrasts_right : dict
            Conditions with right presentations specified as lists of labels.
        channels_left : list of str
            Left channels to pool.
        channels_right : list of str
            Right channels to pool.
        
        Returns
        -------
        erp_wave : DataFrame
            Ipsi- and contralateral ERP waveform for specified conditions.
        ci : DataFrame
            Confidence intervals for ERP waveforms.
        """
        
        erp_wave = pd.DataFrame()
        ci = pd.DataFrame()
        for contr in erp_contrasts_left:
            erp_ipsi = self._get_erp_waves([erp_contrasts_left[contr], erp_contrasts_right[contr]], 
                                           [channels_left, channels_right])
            erp_contra = self._get_erp_waves([erp_contrasts_left[contr], erp_contrasts_right[contr]], 
                                             [channels_right, channels_left])
            if len(erp_contrasts_left) == 1:
                lab_ipsi = 'ipsi'
                lab_contra = 'contra'
            else:
                lab_ipsi = _combine_labels(contr, 'ipsi')   
                lab_contra = _combine_labels(contr, 'contra')  
            erp_wave[lab_ipsi] = erp_ipsi.mean(axis = 1)
            erp_wave[lab_contra] = erp_contra.mean(axis = 1)
            ci[lab_ipsi] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp_ipsi.std(axis = 1) / np.sqrt(len(self.subjects))
            ci[lab_contra] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp_contra.std(axis = 1) / np.sqrt(len(self.subjects))
        return erp_wave, ci
    
        
        
    def lat_comp_plot (self, plot_folder, comp_name, erp_labels_left, erp_labels_right, channels_left, channels_right,
                       plot_ci = False, plot_time_wind = False, time_wind = [], time_wind_color = '#e2e2e2ff', colors = [], styles = [], legend = [''], legend_size = 16, 
                       legend_loc = (0.7, 1.15), fig_size = [8, 5.3], ylim = [-7, 9], y_ticks = [-6, -3, -0, 3, 6, 9], xlim = [-50, 700], 
                       x_ticks = [0, 100, 200, 300, 400, 500, 600, 700], save_format = 'png'):
        """Plot lateralized ERP waveforms.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        comp_name : str
            The name of ERP component.
        erp_labels_left : dict
            Conditions specified as dicts of contrasts with left presentations to plot on one plot.
        erp_labels_right : dict
            Conditions specified as dicts of contrasts with left presentations to plot on one plot
        channels_left : list of str
            Left channels to pool.
        channels_right : list of str
            Right channels to pool.
        plot_ci : bool
            Whether to plot ERP confidence interval.
            Defaults to False.
        plot_time_wind : Bool
            Whether to plot ERP time-window.
            Defaults to False.
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        time_wind_color : color
            Color of wime-window.
            See matplotlib colors.
            Defaults to '#e2e2e2ff'.
        colors : list of colors
            Colors of plotted ERP waveforms.
            See matplotlib colors.
            Defaults to [].
        styles : list
            Linestyles of ERP waveforms.
            See matplotlib linestyles.
            Defaults to [].
        legend : list of str,
            A list of label.
            Defaults to [''].
        legend_size: int | {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            Legend font-size
            Defaults to 16.
        legend_loc : 2-tuple
            Localization of legend box.
            See matplotlib.legend.bbox_to_anchor.
            Defaults to (0.7, 1.15)
        fig_size : list of floats
            The list containing width and height matplotlib figure
            Defaults to [8, 5.3].
        ylim : list of floats
            The list containing min and max of y-axis
            Defaults to [-7, 9].
        y_ticks : list of floats
            The list of y-axis ticks.
            Defaults to [-6, -3, -0, 3, 6, 9].
        xlim : list of floats
            The list containing min and max of x-axis
            Defaults to [-50, 700].
        x_ticks : list of floats
            The list of x-axis ticks.
            Defaults to [0, 100, 200, 300, 400, 500, 600, 700].
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)  
        for cond in erp_labels_left:
            erp_wave_dict, ci_dict = self._lateral_waveforms(erp_labels_left[cond], erp_labels_right[cond], channels_left, channels_right)  
            _plot_erp (erp_wave_dict, cond + '_' + comp_name, self.path + plot_folder, plot_time_wind, time_wind, time_wind_color,
                       plot_ci, ci_dict, colors, styles, cond in legend, legend_size, legend_loc, fig_size, ylim, y_ticks, xlim, x_ticks, save_format)
    
    
            
    def _diff_lat_waveforms(self, erp_labels_left, erp_labels_right, channels_left, channels_right):
        """Calculate differetial lateralized ERP waveforms averaged across participants.
    
        Parameters
        ----------
        erp_labels_left : dict
            Conditions specified as dict of contitions to subtract. 
            Each condition should contain 1 or 2 keys.
            If only one key, contra- and ipsilateral waveforms will be subtracted.
        erp_labels_right : dict
            Conditions specified as dict of contitions to subtract.
            Each condition should contain 1 or 2 keys.
            If only one key, contra- and ipsilateral waveforms will be subtracted.
        channels_left : list of str
            Left channels to pool.
        channels_right : list of str
            Right channels to pool.
        
        Returns
        -------
        diff_wave : DataFrame
            ERP differential waveforms for specified conditions.
        ci : DataFrame
            Confidence intervals for differential ERP waveforms.
        """
        
        diff_wave = pd.DataFrame()
        ci = pd.DataFrame()
        for cond in erp_labels_left:
            if len(erp_labels_left[cond]) == 1:
                for contr in erp_labels_left[cond]:
                    erp_ipsi = self._get_erp_waves([erp_labels_left[cond][contr], erp_labels_right[cond][contr]], 
                                                   [channels_left, channels_right])
                    erp_contra = self._get_erp_waves([erp_labels_left[cond][contr], erp_labels_right[cond][contr]], 
                                                     [channels_right, channels_left])
                    erp = erp_contra = erp_ipsi
                diff_wave[cond] = erp.mean(axis = 1)
                ci[cond] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp.std(axis = 1) / np.sqrt(len(self.subjects))
            else:
                contrasts = list(erp_labels_left[cond].keys())
                erp_ipsi1 = self._get_erp_waves([erp_labels_left[cond][contrasts[1]], erp_labels_right[cond][contrasts[1]]], 
                                                [channels_left, channels_right])
                erp_ipsi2 = self._get_erp_waves([erp_labels_left[cond][contrasts[0]], erp_labels_right[cond][contrasts[0]]], 
                                                [channels_left, channels_right])
                erp_contra1 = self._get_erp_waves([erp_labels_left[cond][contrasts[1]], erp_labels_right[cond][contrasts[1]]], 
                                                  [channels_right, channels_left])
                erp_contra2 = self._get_erp_waves([erp_labels_left[cond][contrasts[0]], erp_labels_right[cond][contrasts[0]]], 
                                                  [channels_right, channels_left])
                erp_ipsi = erp_ipsi1 - erp_ipsi2
                erp_contra = erp_contra1 - erp_contra2
                lab_ipsi = _combine_labels(cond, 'ipsi')   
                lab_contra = _combine_labels(cond, 'contra') 
                diff_wave[lab_ipsi] = erp_ipsi.mean(axis = 1)
                diff_wave[lab_contra] = erp_contra.mean(axis = 1)
                ci[lab_ipsi] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp_ipsi.std(axis = 1) / np.sqrt(len(self.subjects))
                ci[lab_contra] = stats.t.interval(0.95, len(self.subjects) - 1, 0, 1)[1] * erp_contra.std(axis = 1) / np.sqrt(len(self.subjects))
        return diff_wave, ci     
    
    
    
    def lat_diff_plot (self, plot_folder, comp_name, erp_labels_left, erp_labels_right, channels_left, channels_right, 
                       plot_ci = False, plot_time_wind = False, time_wind = [], time_wind_color = '#e2e2e2ff', colors = [], styles = [], legend = [''], legend_size = 16, 
                       legend_loc = (0.7, 1.15), fig_size = [8, 5.3], ylim = [-7, 9], y_ticks = [-6, -3, -0, 3, 6, 9], xlim = [-50, 700], 
                       x_ticks = [0, 100, 200, 300, 400, 500, 600, 700], save_format = 'png'):
        """Plot differetial lateralized ERP waveforms.

        Parameters
        ----------
        plot_folder : string
            A folder to save plots.
        comp_name : str
            The name of ERP component.
        erp_labels_left : dict
            Conditions specified as dict of contitions to subtract. 
            Each condition should contain 1 or 2 keys.
            If only one key, contra- and ipsilateral waveforms will be subtracted.
        erp_labels_right : dict
            Conditions specified as dict of contitions to subtract.
            Each condition should contain 1 or 2 keys.
            If only one key, contra- and ipsilateral waveforms will be subtracted.
        channels_left : list of str
            Left channels to pool.
        channels_right : list of str
            Right channels to pool.
        plot_ci : bool
            Whether to plot ERP confidence interval.
            Defaults to False.
        plot_time_wind : Bool
            Whether to plot ERP time-window.
            Defaults to False.
        time_wind : list of float
            The list containing start and end time of the ERP in seconds, relative to the time-locked event.
            Defaults to [].
        time_wind_color : color
            Color of wime-window.
            See matplotlib colors.
            Defaults to '#e2e2e2ff'.
        colors : list of colors
            Colors of plotted ERP waveforms.
            See matplotlib colors.
            Defaults to [].
        styles : list
            Linestyles of ERP waveforms.
            See matplotlib linestyles.
            Defaults to [].
        legend : list of str,
            A list of label.
            Defaults to [''].
        legend_size: int | {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
            Legend font-size
            Defaults to 16.
        legend_loc : 2-tuple
            Localization of legend box.
            See matplotlib.legend.bbox_to_anchor.
            Defaults to (0.7, 1.15)
        fig_size : list of floats
            The list containing width and height matplotlib figure
            Defaults to [8, 5.3].
        ylim : list of floats
            The list containing min and max of y-axis
            Defaults to [-7, 9].
        y_ticks : list of floats
            The list of y-axis ticks.
            Defaults to [-6, -3, -0, 3, 6, 9].
        xlim : list of floats
            The list containing min and max of x-axis
            Defaults to [-50, 700].
        x_ticks : list of floats
            The list of x-axis ticks.
            Defaults to [0, 100, 200, 300, 400, 500, 600, 700].
        save_format : str
            A format in which plots should be saved.
            Defaults to 'png'.
        """
        
        _make_dir(self.path, plot_folder)
        diff_wave, ci = self._diff_lat_waveforms(erp_labels_left, erp_labels_right, channels_left, channels_right)
        _plot_erp (diff_wave, comp_name + '_differential_lateralized_plots', self.path + plot_folder, plot_time_wind, time_wind, time_wind_color,
                   plot_ci, ci, colors, styles, True, legend_size, legend_loc, fig_size, ylim, y_ticks, xlim, x_ticks, save_format)

    
