# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:14:54 2021

@author: Lucja Doradzinska <l.doradzinska@nencki.edu.pl>
"""

from collections import OrderedDict 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 

import mne
from autoreject import AutoReject

from .files_mng import _make_dir, _save_log, _get_filename



def _set_eog (raw, veog_channels, heog_channels):
    """Set VEOG and HEOG as single channels.

    Parameters
    ----------
    raw : mne.Raw
        The raw instance containing bipolar or aux eogs.
    veog_channels : str | list of str | None
        Names of veog channels.
    heog_channels : str | list of str | None
        Names of heog channels.
    
    Returns
    -------
    raw : mne.Raw
        The raw instance with sigle VEOG and HEOG channels.
    """
    
    if veog_channels == None or heog_channels == None:
        print("unable to set empty eog channel")
    
    elif type(veog_channels)!= type(heog_channels):
        print("eog channels must be of the same type")
    elif type(veog_channels) == list and len(veog_channels)>2 or len(heog_channels)>2:
        print("the list of bipolar eog channels must contain max two electrodes")
    elif type(veog_channels) == list and len(veog_channels) != len(heog_channels):
        print("the list of bipolar eogs must be of the same length")
    else:
        if type(veog_channels) == list:
            if len(veog_channels) == 2:
                # Set bipolar reference.
                raw = mne.set_bipolar_reference (raw, [veog_channels[0], heog_channels[0]], [veog_channels[1], heog_channels[1]], 
                                                 ['VEOG', 'HEOG'], copy=False)
            else:
                # Rename channels
                raw = raw.rename_channels({veog_channels[0]:'VEOG', heog_channels[0]:'HEOG'})
        else:
            # Rename channels
            raw = raw.rename_channels({veog_channels:'VEOG', heog_channels:'HEOG'})
        # Set channel type to eog
        raw = raw.set_channel_types({'VEOG':'eog', 'HEOG':'eog'})   
    return raw

                

def _check_markers(events, stim_markers, subj):
    """Check if all datasets contain all markers."""
    
    for stim in stim_markers:
        if stim_markers[stim] not in events[:,2]:
            del stim_markers[stim]
            print('There are no events ' + stim + ' in raw object ' + subj + '.')



def _find_too_early (events, stim_markers, quest_markers):
    """Find events with too early response.

    Parameters
    ----------
    events : array
        Events extracted from a mne.Raw object with mne.find_events().
    stim_markers : dict
        Dictionary of events labels and ids.
    quest_markers : list of int
        Ids of question events.
    
    Returns
    -------
    del_t_e : list
        List of events to delete from the mne.Raw instance.
    """
        
    del_t_e = []          
    for j in range(len(events) - 1):
        if events[j, 2] in stim_markers.values():
            if events[j + 1, 2] not in quest_markers:
                del_t_e.append(j)
    return del_t_e



def _find_miss (events, stim_markers, resp_markers, quest_markers):
    """Find events with no response.

    Parameters
    ----------
    events : array
        Events extracted from a mne.Raw instance with mne.find_events().
    stim_markers : dict
        Dictionary of events labels and ids.
    resp_markers :  list of int
        Ids of response events.
    quest_markers : list of int
        Ids of question events.
    
    Returns
    -------
    del_m : list
        List of events to delete from the mne.Raw object.
    """
        
    del_m = []
    if quest_markers == []:
        for j in range(len(events) - 1):
            if events[j, 2] in stim_markers.values():
                if events[j + 1, 2] not in resp_markers:
                    del_m.append(j)
        if events[-1, 2] in stim_markers.values():
            del_m.append(len(events) - 1)
    else:
        for j in range(len(events) - 2):
            if events[j, 2] in stim_markers.values():
                if events[j + 2, 2] not in resp_markers:
                    del_m.append(j)
        if events[-2, 2] in stim_markers.values():
            del_m.append(len(events) - 2)
        if events[-1, 2] in stim_markers.values():
            del_m.append(len(events) - 1)
    return del_m



def _find_eog_epochs (epochs, eog_chann, eog_window, eog_thresh): 
    """Find epochs with eog artifacts.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs instance.
    eog_chann : str
        The name of single eog channel.
    eog_window : list of float
        A list containing start and end time of a window in which eog artifacts shoud be checked.
        Values are in seconds, relative to the time-locked event.
    eog_thresh : float
        Peak-to-peak treshold in volts.
        
    Returns
    -------
    rej : list of int
        Indexes of epochs to remove.
    """        
    
    eog = epochs.copy().pick_channels([eog_chann])
    eog = eog.crop(tmin = eog_window[0], tmax = eog_window[1])
    eog = eog.drop_bad(reject={'eog':eog_thresh}) 
    rej = []
    counter = 0
    for j in range(len(eog.drop_log)):
        if 'IGNORED' not in epochs.drop_log[j] and 'USER' not in epochs.drop_log[j]:
            if eog_chann in eog.drop_log[j]:
                rej.append(counter)
            counter = counter + 1  
    
    return rej



def _perf_ica(epochs, n_components, ica_method, eog_channels, ica_thresh):
    """Find eog artifacts using ICA.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs instance.
    n_components : int
        Number of principal components .
    ica_method : ‘fastica’ | ‘infomax’ | ‘picard’
        The ICA method to use in the fit method (see mne.preprocessing.ICA).
    eog_channels : list of str
        A list of eog channels.
    ica_thresh : float
        Threshold on the z-score used in the iterative z-scoring method of finding bad components.
        
    Returns
    -------
    epochs : mne.Epochs
        Epochs instance with removed eog artifacts.
    rej_comps : int
        Number of rejected components.
    """   
     
    #getting eeg channels
    picks = mne.pick_types(epochs.info, eeg = True, eog = False)
    #describing ica
    ica = mne.preprocessing.ICA(n_components = n_components, method = ica_method).fit(epochs, picks)
    rej_comps = 0
    for chan in eog_channels:
        #finding eog components
        eog_inds, _ = ica.find_bads_eog(epochs, chan, ica_thresh)
        rej_comps += len(eog_inds)
        #rejecting eogs
        ica.exclude.extend(eog_inds)
    #applying ica
    epochs = ica.apply(epochs)
    #correcting baseline
    epochs = epochs.apply_baseline(baseline=(None, 0))
    return epochs, rej_comps



def _plot_psd_raw(raws, plot_name, save_psd_plots, path, psd_plot_folder, plot_save_format):
    """Compare psd plots for multiple mne.Raw instances on one figure.

    Parameters
    ----------
    raws : list of mne.Raw
        The list of raw objects to compare.
    save_psd_plots : bool
        Whether to save plots or not.
    path : str
        A path to experiments folder.
    psd_plot_folder : str
        A name of psd folder.
    plot_save_format : str
        A format in which plots should be saved.
    """
    fig, axs= plt.subplots(len(raws), 1, figsize=(5*len(raws), 9))
    
    for i, raw in enumerate(raws):
        raw.plot_psd(fmin = 0, fmax = 70, n_fft = 8192, ax = axs[i], show = False)
    fig.suptitle(plot_name)
    plt.show()
        
    if save_psd_plots:
        _make_dir(path, psd_plot_folder)  
                
        plt.savefig(path + psd_plot_folder + plot_name + '_psd.' + plot_save_format, format = plot_save_format)



def _plot_psd_epoch(epochs, plot_name, save_psd_plots, path, psd_plot_folder, plot_save_format):
    """Compare psd plots for multiple mne.Epochs instances on one figure.

    Parameters
    ----------
    epochs : list of mne.Epochs
        The list of epochs objects to compare.
    plot_name : str
        The name of the figure.
    save_psd_plots : bool
        Whether to save plots or not.
    path : str
        A path to experiments folder.
    psd_plot_folder : str
        A name of psd folder.
    plot_save_format : str
        A format in which plots should be saved.
    """
    
    fig, axs= plt.subplots(len(epochs), 1, figsize=(5*len(epochs), 9))
    for i, ep in enumerate(epochs):
        ep.plot_psd(fmin = 0, fmax = 70, ax = axs[i], show = False)
    fig.suptitle(plot_name)
    plt.show()   
    if save_psd_plots:
        _make_dir(path, psd_plot_folder)  
        plt.savefig(path + psd_plot_folder + plot_name + '_psd.' + plot_save_format, format = plot_save_format)



def raw_prep(subjects, path, raw_signal_folder, raw_prep_folder, stim_channel, montage = 'biosemi64',  
             set_eog = False, veog_channels = None, heog_channels = None, exclude_channels = [], set_reref = True, 
             reref_channels = 'average'):
    """Set montage, eogs and rereference raw data.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    raw_signal_folder : str
        A folder with raw EEG data.
    raw_prep_folder : str
        A folder to save preprocessed data.
    stim_channel : str
        Stimulus channel name.
    montage : str
        Standard mne montage.
        Defaults to 'biosemi64'.
    set_eog : bool
        Whether to set eog channels.
        Defaults to False.
    veog_channels : str | list of str | None
        Names of veog channels.
        Defaults to None.
    heog_channels : str | list of str | None
        Names of heog channels.
        Defaults to None.
    exclude_channels : list of str
        Channels to exclude from dataset.
        Defaults to [].
    set_eog : bool
        Whether to rereference data.
        Defaults to True.
    reref_channels : str | list of str
        The name(s) of channel(s) used to construct reference or predefined method.
        See mne.set_eeg_reference.
        Defaults to 'average'.
    """
    
    _make_dir(path, raw_prep_folder)  
    #creating standard montage
    montage = mne.channels.make_standard_montage(montage)
    #listing files in dir
    filenames = os.listdir(path + raw_signal_folder)   
    #preprocessing EEG data
    for subj in subjects:
        #searching for dataset
        filename = _get_filename(filenames, subj)
        #importing data and dropping empty channels
        raw = mne.io.read_raw_bdf(path + raw_signal_folder + filename, stim_channel = stim_channel, exclude = exclude_channels, preload = True)
        #setting bipolar eog channels
        if set_eog:
            raw = _set_eog(raw, veog_channels, heog_channels)
        #rereferencing
        if set_reref:
            raw , _ = mne.set_eeg_reference(raw, reref_channels, copy = False)     
        if type(reref_channels) == list:
            raw = raw.drop_channels(reref_channels)
        #setting electrodes locations
        raw = raw.set_montage(montage)
        #saving raw object    
        raw.save(path + raw_prep_folder + subj + '_raw.fif', overwrite = True)



def filter_raw(subjects, path, raw_prep_folder, filt_folder, set_filter = True, filt_method = 'fir', 
               low_cutoff = 0.1, high_cutoff = 40, set_notch = False, plot_psd = False, save_psd_plots = False, 
               plot_save_format = 'png'):
    """Filter raw data and plot psd before and after cleaning.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    raw_prep_folder : str
        A folder with uncleaned raw EEG data.
    filt_folder : str
        A folder to save filtered data.
    set_filter : bool
        Whether to filter data.
        Defaults to True.
    filt_method : str
        Filtering method.
        Defaults to 'fir'.
    low_cutoff : float  
        Lower cutoff frequency in dB.
        Defaults to 0.1.
    high_cutoff : float 
        Higher cutoff frequency in dB.
        Defaults to 40.
    set_notch : bool
        Whether to filter data with notch.
        Defaults to False.
    plot_psd : bool
        Whether to plot psd comparison.
        Defaults to False.
    save_psd_plots : bool
        Whether to save psd plots.
        Defaults to False.
    plot_save_format : str
        A format in which plots should be saved.
        Defaults to 'png'.
    """ 
    
    _make_dir(path, filt_folder) 
    if filt_method == 'fir':
        low_cutoff = 2*low_cutoff
        high_cutoff = (8*high_cutoff)/9
    #preprocessing EEG data
    for subj in subjects:
        #loading raw eeg data
        raw = mne.io.read_raw_fif(path + raw_prep_folder + subj + '_raw.fif', preload=True)  
        raw_filt = raw.copy()
        #filtering
        if set_filter:
            raw_filt = raw_filt.filter(low_cutoff, high_cutoff, 'eeg', method = filt_method)
            if 'eog' in raw_filt:
                raw_filt = raw_filt.filter(low_cutoff, high_cutoff, 'eog', method = filt_method)
        if set_notch:
            raw_filt = raw_filt.notch_filter(np.arange(50, 201, 50), 'eeg')
            if 'eog' in raw_filt:
                raw_filt = raw_filt.notch_filter(np.arange(50, 201, 50), 'eog')
        if plot_psd:   
            _plot_psd_raw([raw, raw_filt], subj, save_psd_plots, path, filt_folder + 'psd_plots\\', plot_save_format)
        #saving preprocessed eeg signal
        raw_filt.save(path + filt_folder + subj + '_filt.fif', overwrite = True)


        
def to_epochs(subjects, path, filt_folder, epochs_raw_folder, eeg_log_folder, stim_channel, stim_markers, 
              resp_markers = [], quest_markers = [], reject_miss = True, reject_too_early = True, epoch_tmin = -0.2, 
              epoch_tmax = 1.2, epoch_baseline = (None, 0), resample = False, resample_freq = 256, save_xlsx = True, 
              save_csv = False):
    """Transform raw data to epochs rejecting invalid trials.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    filt_folder : str
        A folder with filtered raw EEG data.
    epochs_raw_folder : str
        A folder to save epoched data.
    eeg_log_folder : str
        A folder to save rejection log.
    stim_channel : str
        Stimulus channel name.
    stim_markers : dict
        Dictionary of events labels and ids.
    resp_markers :  list of int
        Ids of response events.
        Defaults to [].
    quest_markers : list of int
        Ids of question events.    
        Defaults to [].
    reject_miss : bool
        Whether to reject trials with no response.
        Defaults to True.
    reject_too_early : bool
        Whether to reject trials with too early responses.
        Defaults to True.
    epoch_tmin : float
        Start time of the epochs in seconds, relative to the time-locked event.
        Defaults to -0.2.
    epoch_tmax : float
       End time of the epochs in seconds, relative to the time-locked event.
       Defaults to 1.2.
    epoch_baseline : None | tuple of length 2
        The time interval considered as baseline. If None no baseline correction will be applyed (see mne.Epoch()).
        Defaults to (None, 0).
    resample : bool
        Whether to resample EEG data.
        Defaults to False.  
    resample_freq : float
        Resampling frequency in Hz.
        Defaults to 256.
    save_xlsx : bool
        Whether to save log in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether to save log in csv format.
        Defaults to False.
    
    Returns
    -------
    invalid_resp_df : pd.DataFrame
        Log with the number of rejected epochs per subject.
    """  
    
    _make_dir(path, epochs_raw_folder)  
    #creating rejection log
    log = OrderedDict()
    if reject_miss or reject_too_early:
        log['subjects'] = subjects   
    if reject_too_early:
        log['too_early'] = []
    if reject_miss:
        log['miss'] = []
    for subj in subjects:
        #loading raw eeg data
        raw = mne.io.read_raw_fif(path + filt_folder + subj + '_filt.fif', preload=True)  
        #getting events array
        events = mne.find_events(raw, stim_channel, shortest_event = 1)
        _check_markers(events, stim_markers, subj)
        # rejecting too early trials
        if reject_too_early:
            del_t_e = _find_too_early(events, stim_markers, quest_markers)
            log['too_early'].append(len(del_t_e))
            events = np.delete(events, del_t_e, 0)  
        #rejecting miss trials
        if reject_miss:
            del_m = _find_miss(events, stim_markers, resp_markers, quest_markers)
            log['miss'].append(len(del_m))       
            events = np.delete(events, del_m, 0)
        #epoching
        epochs = mne.Epochs(raw, events, stim_markers, epoch_tmin, epoch_tmax, baseline = epoch_baseline,
                            preload = True, reject_by_annotation = False)
        #resampling
        if resample:
            epochs = epochs.resample(resample_freq)
        #saving epochs
        epochs.save(path + epochs_raw_folder + subj + '_epo.fif', overwrite=True)
    _save_log(log, path + eeg_log_folder, 'invalid_resp_log', save_xlsx, save_csv)
    invalid_resp_df = pd.DataFrame(log, columns = list(log.keys()))   
    return invalid_resp_df



def remove_eogs(subjects, path, epochs_raw_folder, epochs_noeog_folder, eeg_log_folder, reject_veog = False, 
                veog_thresh = 140e-6, veog_window = [-0.1, 0.1], reject_heog = False, heog_thresh = 80e-6, 
                heog_window = [0, 0.5], perform_ica = False, ica_thresh = 3, n_components = 64, ica_method = 'fastica',
                save_xlsx = True, save_csv = False):
    """Remove epochs with eye movements and components with remaining EOG artifacts.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    epochs_raw_folder : str
        A folder with raw, epoched EEG data.
    epochs_noeog_folder : str
        A folder to save epochs after EOG artifact removal.
    eeg_log_folder : str
        A folder to save rejection log.
    reject_veog : bool
        Whether to reject trials with vertical eye movement.
        Defaults to False.
    veog_thresh : float
        Peak-to-peak treshold in volts for veog rejection.
        Defaults to 140e-6.
    veog_window : list of float
        A list containing start and end time of a window in which veog artifacts shoud be checked.
        Values are in seconds, relative to the time-locked event.   
        Defaults to [-0.1, 0.1].
    reject_heog : bool
        Whether to reject trials with horizontal eye movement.
        Defaults to False.
    heog_thresh : float
        Peak-to-peak treshold in volts for heog rejection.
        Defaults to 80e-6.
    heog_window : list of float
        A list containing start and end time of a window in which heog artifacts shoud be checked.
        Values are in seconds, relative to the time-locked event.   
        Defaults to [0, 0.5].
    perform_ica : bool
        Whether perform ICA decomposition for eoug artifact rejection.
        Defaults to False.
    ica_thresh : float
        Threshold on the z-score used in the iterative z-scoring method of finding bad components.
        Defaults to 3.
    n_components : int
        Number of principal components.
        Defaults to 64.
    ica_method : ‘fastica’ | ‘infomax’ | ‘picard’
        The ICA method to use in the fit method (see mne.preprocessing.ICA).
        Defaults to 'fastica'.
    save_xlsx : bool
        Whether to save log in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether to save log in csv format.
        Defaults to False.
    
    Returns
    -------
    eog_artifacts_df : pd.DataFrame
        Log with the number of rejected epochs and components per subject.
    """  
    
    _make_dir(path, epochs_noeog_folder)          
    #creating rejection log
    log = OrderedDict()
    log['subjects'] = subjects
    if reject_veog:
        log['VEOG_epochs'] = []
    if reject_heog:
        log['HEOG_epochs'] = []    
    if perform_ica:
        log['rej_comps'] = []    
    for subj in subjects:
        epochs = mne.read_epochs(path + epochs_raw_folder + subj + '_epo.fif')
        #removing epochs with blinks at stimulus
        if reject_veog:
            #finding heog artifacts
            veog_rej = _find_eog_epochs(epochs, 'VEOG', veog_window, veog_thresh)
            log['VEOG_epochs'].append(len(veog_rej))
            #removing epochs with veog artifacts
            epochs.drop(veog_rej)
        #removing epochs with horizontal eye movments
        if reject_heog:
            #finding heog artifacts
            heog_rej = _find_eog_epochs(epochs, 'HEOG', heog_window, heog_thresh)
            log['HEOG_epochs'].append(len(heog_rej))
            #removing epochs with heog artifacts
            epochs.drop(heog_rej)
        #cleaning data from eog artifacts with ICA
        if perform_ica:
            epochs, rej_comps = _perf_ica(epochs, n_components, ica_method, ['VEOG', 'HEOG'], ica_thresh)
            log['rej_comps'].append(rej_comps)
        #dropping eog channels    
        epochs.drop_channels(['VEOG', 'HEOG']) 
        #saving cleaned epochs
        epochs.save(path + epochs_noeog_folder + subj + '_epo.fif', overwrite=True)
    _save_log(log, path + eeg_log_folder, 'eog_artifacts_log', save_xlsx, save_csv)
    eog_artifacts_df = pd.DataFrame(log, columns = list(log.keys()))  
    return eog_artifacts_df

    

def clean_epochs(subjects, path, epochs_noeog_folder, epochs_clean_folder, eeg_log_folder,
                 autoreject = False, autorej_method = 'random_search', rej_by_amplitude = False, 
                 eeg_thresh = 140e-6, rej_manually = False, save_xlsx = True, save_csv = False, 
                 plot_psd = False, save_psd_plots = False, plot_save_format = 'png'):
    """Clean epochs from EEG artifacts and plot psd before and after cleaning.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    epochs_noeog_folder : str
        A folder with epoched EEG data.
    epochs_clean_folder : str
        A folder to save epochs after EEG artifact removal.
    eeg_log_folder : str
        A folder to save rejection log.
    autoreject : bool
        Whether to perform rejection with AutoReject.
        Defaults to False.
    autorej_method : 'random_search' | 'bayesian_optimization'
        The method to establich rejection threshold (see autoreject.AutoReject()).
        Defaults to 'random_search'.
    rej_by_amplitude : bool
        Whether to reject epochs based on peak-to-peak amplitude.
        Defaults to False.
    eeg_thresh : float
        Peak-to-peak treshold in volts for eeg rejection.
        Defaults to 140e-6.
    rej_manually : bool
        Whether to display epochs for manual rejection.
        Defaults to False.
    save_xlsx : bool
        Whether to save log in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether to save log in csv format.
        Defaults to False.
    plot_psd : bool
        Whether to plot psd comparison.
        Defaults to False.
    save_psd_plots : bool
        Whether to save psd plots.
        Defaults to False.
    plot_save_format : str
        A format in which plots should be saved.
        Defaults to 'png'.    
    
    Returns
    -------
    epochs_cleaning_df  : pd.DataFrame
        Log with the number of rejected epochs per subject.
    """
    
    _make_dir(path, epochs_clean_folder)   
    #creating rejection log
    log = OrderedDict()
    log['subjects'] = subjects
    log['bad_epochs'] = []
    if not (autoreject or rej_by_amplitude):
        log['bad_channels'] = []
    for subj in subjects:
        epochs = mne.read_epochs(path + epochs_noeog_folder + subj + '_epo.fif')
        epochs_clean = epochs.copy()
        if autoreject:
            #creating AutoReject object
            ar = AutoReject(thresh_method = autorej_method, verbose = False)
            #fitting the data to AR object
            ar.fit(epochs_clean)
            #removing bad epochs and interpolating bad channels:
            epochs_clean = ar.transform(epochs_clean)
        #rejecting bad epochs by amplitude
        if rej_by_amplitude:
            #rejecting bad epochs
            epochs_clean = epochs_clean.drop_bad(reject={'eeg':eeg_thresh})
        #rejecting epochs manually        
        if rej_manually:
            epochs_clean.plot(title = "Select epochs for exclusion by clicking on the signal (red-drop, black-leave) and channels for interpolation by clicking on their name (grey-interpolate, black-leave). Press Enter to continue...")
            while True:
                if plt.waitforbuttonpress(): break
            plt.close()
            log['bad_channels'].append(len(epochs_clean.info['bads']))
            epochs_clean.interpolate_bads()
        log['bad_epochs'].append(len(epochs.events) - len(epochs_clean.events)) 
        if plot_psd:
            _plot_psd_epoch([epochs, epochs_clean], subj, save_psd_plots, path, epochs_clean_folder + 'psd_plots\\', 
                                   plot_save_format)
        #saving cleaned epochs
        epochs_clean.save(path + epochs_clean_folder + subj + '_epo.fif', overwrite=True)
    _save_log(log, path + eeg_log_folder, 'epochs_cleaning_log', save_xlsx, save_csv)
    epochs_cleaning_df = pd.DataFrame(log, columns = list(log.keys()))     
    return epochs_cleaning_df 
    


def calculate_evoked(subjects, path, epochs_clean_folder, evokeds_folder, eeg_log_folder, reduce_markers = {}, 
                     evoked_labels = [], equalize = True, equalize_method = 'mintime', save_xlsx = True, save_csv = False): 
    """Clean epochs from EEG artifacts and plot psd before and after cleaning.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    epochs_clean_folder : str
        A folder to save epochs after EEG artifact removal.
    epochs_noeog_folder : str
        A folder with epoched EEG data.
    evokeds_folder : str
        A folder to save evokeds.
    eeg_log_folder : str
        A folder to save rejection log.
    reduce markers : dict
        The dictionary containing labels to replace with new ones.
        Defaults to {}.
    evoked_labels : list of lists
        The list of conditions of evoked data.
        Defaults to [].
    equalize : bool
        Whether to equalize number of epochs between conditions.
        Defaults to True.
    equalize_method : str
        Method to peak epochs for rejection.
        See mne.Epochs.equalize_event_counts().
        Defaults to 'mintime'.
    save_xlsx : bool
        Whether to save log in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether to save log in csv format.
        Defaults to False.
        
    Returns
    -------
    evokeds_df  : pd.DataFrame
        Log with the number remaining epochs per condition.
    """
    
    _make_dir(path, evokeds_folder) 
    #creating evoked container
    evokeds = OrderedDict()
    log = OrderedDict()
    log['subjects'] = subjects
    for cond in evoked_labels:
        for lab in cond:
            log[lab] = []
            evokeds[lab] = []
    #calculating ERPs
    for subj in subjects:
        #loading epochs
        epochs = mne.read_epochs(path + epochs_clean_folder + subj + '_epo.fif')
        #reducing markers
        for marker in reduce_markers:
            epochs = mne.epochs.combine_event_ids(epochs, marker['old_labels'],
                                                  {marker['new_label']:marker['new_marker']})
        #equalizing and averaging ERP        
        for cond in evoked_labels:
            if equalize:
                epochs= epochs.equalize_event_counts(cond, method = equalize_method)
            for lab in cond:
                log[lab].append(len(epochs[lab]))
                evokeds[lab].append(epochs[lab].average())
    #saving evokeds 
    for cond in evoked_labels:
        for lab in cond:  
            mne.write_evokeds(path + evokeds_folder + lab + '_ave.fif',  evokeds[lab])
    _save_log(log, path + eeg_log_folder, 'evokeds_log', save_xlsx, save_csv)
    evokeds_df = pd.DataFrame(log, columns = list(log.keys())) 
    return evokeds_df


