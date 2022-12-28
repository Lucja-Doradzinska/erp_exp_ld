# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:53:25 2022

@author: Lucja Doradzinska <l.doradzinska@nencki.edu.pl>
"""

from collections import OrderedDict 
import pandas as pd 

import mne

from .files_mng import _make_dir, _save_log



class Trials_data:
    """Class to extract components trial by trial.
    
    Parameters
    ----------
    subjects : list of str
        The list of subject ids.
    path : str
        A path to EEG data.
    epochs_folder : str
         A folder with mne.Epochs.
    reduce markers : dict
        The dictionary containing labels to replace with new ones.
        Defaults to {}.
    evoked_labels : list of lists
        The list of conditions of evoked data.
        Defaults to [].
    equalize : bool
        Whether equalize number of epochs between conditions.
        Defaults to False.
    equalize_method : str
        Method to peak epochs for rejection.
        See mne.Epochs.equalize_event_counts().
        Defaults to 'mintime'.
        
    Attributes
    ----------
    epochs : dict
        The dictionary holding mne.Epochs instances.
    """
    
    def __init__(self, subjects, path, epochs_folder, reduce_markers = {}, evoked_labels = [], equalize = False, 
                 equalize_method = 'mintime'):
        """Init it."""
        
        self.subjects = subjects
        self.path = path
        self.epochs = OrderedDict()
        for subj in subjects:
            #loading epochs
            epochs = mne.read_epochs(path + epochs_folder + subj + '_epo.fif')
            #reducing markers
            for marker in reduce_markers:
                epochs = mne.epochs.combine_event_ids(epochs, marker['old_labels'], {marker['new_label']:marker['new_marker']})
            #equalizing epochs 
            if equalize:
                for cond in evoked_labels:
                    epochs = epochs.equalize_event_counts(cond, method = equalize_method)[0]
            self.epochs[subj] = epochs
            
    
    
    def get_central_comp (self, comp_long_folder, comp_name, comp_labels, channels, time_window, save_xlsx = True, 
                          save_csv = True, return_log = False):
        """Extract central components amplitudes trial by trial.

        Parameters
        ----------
        comp_long_folder : string
            A folder to save extracted components data.
        comp_name : str
            The name of ERP component.
        comp_labels : list of dicts
            A list of dicts, each containing a list of labels and a dict with conditions.
        channels : list of str
            Channels to pool.
        time_window : list of float
            The list containing start and end time of the ERP in miliseconds, relative to the time-locked event.
        save_xlsx : bool
            Whether save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether save log in csv format.
            Defaults to True.
        return_log : bool
            Whether to return pd.DataFrame with extracted data.
            Defaults to True.
            
        Returns
        -------
        comp_long_df  : pd.DataFrame
            If return_log is True returns log with components mean amplitudes extracted trial by trial.
        """
        
        _make_dir(self.path, comp_long_folder) 
        log = OrderedDict()
        log['subject'] = []
        for label in comp_labels:
            for factor in label['conds']:
                log[factor] = []
        log['erp_amp'] = []
        for subj in self.subjects:
            for indx in range(len(self.epochs[subj])):
                epoch = self.epochs[subj][indx]
                for label in comp_labels:
                    if list(epoch.event_id.keys())[0] in label['labels']:
                        log['subject'].append(subj)
                        for factor in label['conds']:
                            log[factor].append(label['conds'][factor])
                        log['erp_amp'].append(epoch.get_data(picks = channels, tmin = time_window[0]/1000, tmax = time_window[1]/1000).mean() * 1e6)
        _save_log(log, self.path + comp_long_folder, comp_name + '_amps_long_format', save_xlsx, save_csv)  
        if return_log:
            comp_long_df = pd.DataFrame(log, columns = list(log.keys()))
            return comp_long_df
        
        
        
    def get_lateral_comp (self, comp_long_folder, comp_name, comp_labels_left, comp_labels_right, channels_left, channels_right, 
                          time_window, save_xlsx = True, save_csv = True, return_log = False):
        """Extract lateralized components amplitudes trial by trial.

        Parameters
        ----------
        comp_long_folder : string
            A folder to save extracted components data.
        comp_name : str
            The name of ERP component.
        comp_labels_left : list of dicts
            A list of dicts, each containing a list of labels and a dict with conditions with left presentations.
        comp_labels_right : list of dicts
            A list of dicts, each containing a list of labels and a dict with conditions with right presentations.
        channels_left : list of str
            Left channels to pool.
        channels_right : list of str
            Right channels to pool.
        time_window : list of float
            The list containing start and end time of the ERP in miliseconds, relative to the time-locked event.
        save_xlsx : bool
            Whether save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether save log in csv format.
            Defaults to True.
        return_log : bool
            Whether to return pd.DataFrame with extracted data.
            Defaults to True.
            
        Returns
        -------
        comp_long_df  : pd.DataFrame
            If return_log is True returns log with components mean amplitudes extracted trial by trial.
        """
        
        _make_dir(self.path, comp_long_folder) 
        log = OrderedDict()
        log['subject'] = []
        for label in comp_labels_left:
            for factor in label['conds']:
                log[factor] = []
        log['side'] = []
        log['erp_amp'] = []
        for subj in self.subjects:
            for indx in range(len(self.epochs[subj])):
                epoch = self.epochs[subj][indx] 
                for label in comp_labels_left:
                    if list(epoch.event_id.keys())[0] in label['labels']:
                        log['subject'].append(subj)
                        for factor in label['conds']:
                            log[factor].append(label['conds'][factor])
                        log['side'].append('ipsi')
                        log['erp_amp'].append(epoch.get_data(picks = channels_left, tmin = time_window[0]/1000, tmax = time_window[1]/1000).mean() * 1e6)
                        log['subject'].append(subj)
                        for factor in label['conds']:
                            log[factor].append(label['conds'][factor])
                        log['side'].append('contra')
                        log['erp_amp'].append(epoch.get_data(picks = channels_right, tmin = time_window[0]/1000, tmax = time_window[1]/1000).mean() * 1e6)
                for label in comp_labels_right:
                    if list(epoch.event_id.keys())[0] in label['labels']:
                        log['subject'].append(subj)
                        for factor in label['conds']:
                            log[factor].append(label['conds'][factor])
                        log['side'].append('ipsi')
                        log['erp_amp'].append(epoch.get_data(picks = channels_right, tmin = time_window[0]/1000, tmax = time_window[1]/1000).mean() * 1e6)
                        log['subject'].append(subj)
                        for factor in label['conds']:
                            log[factor].append(label['conds'][factor])
                        log['side'].append('contra')
                        log['erp_amp'].append(epoch.get_data(picks = channels_left, tmin = time_window[0]/1000, tmax = time_window[1]/1000).mean() * 1e6)
        _save_log(log, self.path + comp_long_folder, comp_name + '_amps_long_format', save_xlsx, save_csv)  
        if return_log:
            comp_long_df = pd.DataFrame(log, columns = list(log.keys()))
            return comp_long_df
        
        