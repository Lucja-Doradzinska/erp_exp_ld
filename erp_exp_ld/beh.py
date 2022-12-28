# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:51:22 2021

@author: Lucja Doradzinska <l.doradzinska@nencki.edu.pl>
"""

from collections import OrderedDict 
import numpy as np
import os
import pandas as pd 
from scipy import stats

from .files_mng import _make_dir, _save_log, _get_filename



def _open_logfile(filenames, subj, log_extention, path, logfile_folder):
    """Open presentation logfile and read events.

    Parameters
    ----------
    filenames : list of str
        List of filenames.
    subj : str
        Subjects id.
    log_extention : str
        Logfile extention.
    path : str
        A path to experiments folder.
    logfile_folder : str
        A folder with logfiles
        
    Returns
    -------
    events : list
        List of events read from the logfile.
    filename : str
        Filename for subject.
    """
    for name in filenames:
        if subj in name and log_extention in name:
            filename = name
    beh = open(path + logfile_folder + filename, 'r')
    events = beh.readlines()
    beh.close()
    del events[0:5]
    return events, filename



def _get_beh_log(events, factors, stim_conditions, quest_codes, trial_end_codes, valid_resps):
    """Extract trials to a dictionary.

    Parameters
    ----------
    events : list
        List of events read from the logfile.
    factors : list of str
        A list with factors.
    stim_conditions : dict
        A dictionary containing events as keys and a dict of conditions for each key.
    quest_codes : list of str
        A list with codes naming the question event.
    trial_end_codes : list of str
        A list with codes naming the end of the trial.
    valid_resps : list of str
        A list with codes naming valid responses.
        
    Returns
    -------
    log : dict
        The dictionary containing trials data.
    """
    
    log = OrderedDict()  
    for factor in factors:
        log[factor] = []
    log['response'] = []
    log['resp_code'] = []
    log['RT'] = []
    flag_trial = 0
    flag_quest = 0
    flag_resp = 0
    quest_time = 0
    for i in range(len(events)):
        try:
            events[i] = events[i].split("\t")
        except:
            pass
    for line in events:
        if len(line) > 3:
            code = line[3]
            time = float(line[4]) / 10
            if code in stim_conditions:
                for factor in stim_conditions[code]:
                    log[factor].append(stim_conditions[code][factor])
                flag_trial = 1
                flag_quest = 0
                flag_resp = 0
            elif code in quest_codes:
                quest_time = time
                flag_quest = 1
            elif code in valid_resps and flag_trial == 1 and flag_resp == 0:
                if flag_quest == 0:
                    log['response'].append('too_early')
                    log['resp_code'].append(None)
                    log['RT'].append(None)
                else:
                    log['response'].append(valid_resps[code])
                    log['resp_code'].append(code)
                    log['RT'].append(time - quest_time)
                flag_resp = 1
            elif code in trial_end_codes and flag_trial == 1 and flag_resp == 0:
                log['response'].append('miss')
                log['resp_code'].append(None)
                log['RT'].append(None)
                flag_trial = 0
    return log



def logfiles_to_array(subjects, path, logfile_folder, log_extention, raw_beh_df_folder, factors, stim_conditions, quest_codes, 
                      trial_end_codes, valid_resps, save_xlsx = True, save_csv = False):
    """Rewrite trials from presentation logfile to xlsx or csv array.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    logfile_folder : str
        A folder with logfiles.
    log_extention : str
        Logfile extention.
    raw_beh_df_folder : str
        A folder to write extracted arrays.
    factors : list of str
        A list with factors.
    stim_conditions : dict
        A dictionary containing events as keys and a dict of conditions for each key.
    quest_codes : list of str
        A list with codes naming the question event.
    trial_end_codes : list of str
        A list with codes naming the end of the trial.
    valid_resps : list of str
        A list with codes naming valid responses.
    save_xlsx : bool
        Whether save array in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether save array in csv format.
        Defaults to False.
    """
    
    _make_dir(path, raw_beh_df_folder) 
    filenames = os.listdir(path + logfile_folder) 
    for subj in subjects:
        events, filename = _open_logfile(filenames, subj, log_extention, path, logfile_folder)
        log = _get_beh_log(events, factors, stim_conditions, quest_codes, trial_end_codes, valid_resps)
        _save_log(log, path + raw_beh_df_folder, filename.split('.')[0], save_xlsx, save_csv)
                       


def _get_surv_log(events, quests):
    """Extract survey responses to a dictionary.

    Parameters
    ----------
    events : list
        List of events read from the logfile.
    quests : list of str
        A list with codes naming the question event.
        
    Returns
    -------
    log : dict
        The dictionary containing survey responses.
    """
    
    log = OrderedDict()
    log['quest'] = []
    log['quest_content'] = []
    log['resp_code'] = []
    log['resp_content'] = []
    for i in range(len(events)):
        try:
            events[i] = events[i].split("\t")
        except:
            pass
    current_quest = ''
    for line in events:
        if len(line) > 3:
            code = line[3]
            if code in quests:
                log['quest'].append(code)
                log['quest_content'].append(quests[code]['content'])
                current_quest = code
            elif current_quest in quests:
                if code in quests[current_quest]['resps']:
                    log['resp_code'].append(code)
                    log['resp_content'].append(quests[current_quest]['resps'][code])
                    current_quest = ''
            else:
                current_quest = ''
    for quest in quests:
        if quest not in log['quest']:
            log['quest'].append(quest)
            log['quest_content'].append(quests[quest]['content'])
            log['resp_code'].append(None)
            log['resp_content'].append(None)
    return log



def survey_to_df(subjects, path, logfile_folder, log_extention, raw_beh_df_folder, quests,
                  save_xlsx = True, save_csv = False):
    """Rewrite survey responses from presentation logfile to xlsx or csv array.

    Parameters
    ----------
    subjects : list of str
        List of subject ids.
    path : str
        A path to experiments folder.
    logfile_folder : str
        A folder with logfiles.
    log_extention : str
        Logfile extention.
    raw_beh_df_folder : str
        A folder to write extracted arrays.
    quests : list of str
        A list with codes naming the question event.
    save_xlsx : bool
        Whether to save array in xlsx format.
        Defaults to True.
    save_csv : bool
        Whether to save array in csv format.
        Defaults to False.
    """
    
    _make_dir(path, raw_beh_df_folder)  
    filenames = os.listdir(path + logfile_folder) 
    for subj in subjects:
        events, filename = _open_logfile(filenames, subj, log_extention, path, logfile_folder)
        log = _get_surv_log(events, quests)
        _save_log(log, path + raw_beh_df_folder, filename.split('.')[0], save_xlsx, save_csv)
                       


def _get_labels(beh_compare, pool_all, raw_beh_df, line):
    """Get labels describing a trial.

    Parameters
    ----------
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    line : int
        Index of a current trial.
    
    Returns
    -------
    labels : list of str
        The list of labels.
    """
    
    if pool_all:
        labels = ['all']
    else:
        labels = []
    for label in beh_compare:
        if all([beh_compare[label][factor] == str(raw_beh_df.at[line, factor]) for factor in beh_compare[label]]):
            labels.append(label)
    return labels



def _count_corr_resps (raw_beh_df, resps_log, beh_compare, pool_all):
    """Count trials with correct responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    resps_log : dict of lists
        A dictionary in which to write responses counts.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    
    Returns
    -------
    resps_log : dict of lists
        A log filled with counted correct responses.
    """
    
    for lab in resps_log:
        if lab != 'subjects':
            resps_log[lab].append(0)
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        for label in labels:
            resps_log[label + '_total'][-1] += 1
        if raw_beh_df.at[line, 'response'] == 'correct':
            for label in labels:
                resps_log[label + '_corr'][-1] += 1
        elif raw_beh_df.at[line, 'response'] == 'incorrect':
            for label in labels:
                resps_log[label + '_incorr'][-1] += 1
        else:
            for label in labels:
                resps_log[label + '_invalid'][-1] += 1
    return resps_log            
 


def _count_resps_types (raw_beh_df, resps_log, beh_compare, resps_types, pool_all):
    """Count trials with particular responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    resps_log : dict of lists
        A dictionary in which to write responses counts.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    resps_types : list of str
        List of responses to count.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    
    Returns
    -------
    resps_log : dict of lists
        A log filled with counted responses.
    """
    
    for lab in resps_log:
        if lab != 'subjects':
            resps_log[lab].append(0)
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        for label in labels:
            resps_log[label + '_total'][-1] += 1
        if raw_beh_df.at[line, 'response'] in resps_types:
            resp = raw_beh_df.at[line, 'response']
            for label in labels:
                resps_log[label + '_' + resp][-1] += 1
        else:
            for label in labels:
                resps_log[label + '_invalid'][-1] += 1
    return resps_log                          



def _count_too_early(raw_beh_df, time_trim_log, beh_compare, RT_min, RT_max, pool_all):
    """Count trials with too early or too late responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    time_trim_log : dict of lists
        A dictionary in which to write responses counts.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    RT_min : float
        Minimal RT in miliseconds.
    RT_max : float
        Maximal RT in miliseconds.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    
    Returns
    -------
    time_trim_log : dict of lists
        A log filled with counted too early or too late responses.
    """
    
    for lab in time_trim_log:
        if lab != 'subjects':
            time_trim_log[lab].append(0)
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        for label in labels:
            time_trim_log[label + '_total'][-1] += 1
        if raw_beh_df.at[line, 'response'] == 'correct' or raw_beh_df.at[line, 'response'] == 'incorrect':
            if raw_beh_df.at[line, 'RT'] < RT_min:
                for label in labels:
                    time_trim_log[label + '_too_early'][-1] += 1
            elif raw_beh_df.at[line, 'RT'] > RT_max:
                for label in labels:
                    time_trim_log[label + '_too_late'][-1] += 1
            else:
                for label in labels:
                    time_trim_log[label + '_valid'][-1] += 1
    return time_trim_log
 
    

def _get_valid_rates(raw_beh_df, valid_log, beh_compare, pool_all, RT_trim_min, RT_trim_max):
    """Extract rates of valid responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    valid_log : dict of lists
        A dictionary in which to write responses rates.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    RT_trim_min : float
        Minimal RT in miliseconds.
    RT_trim_max : float
        Maximal RT in miliseconds.
        
    Returns
    -------
    valid_log : dict of lists
        A log filled with valid responses rates.
    """
    
    value_container = OrderedDict()
    for lab in valid_log:
        if lab != 'subjects':
            value_container[lab + '_valid'] = 0
            value_container[lab + '_total'] = 0
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        if (raw_beh_df.at[line, 'response'] == 'correct' or raw_beh_df.at[line, 'response'] == 'incorrect') and raw_beh_df.at[line, 'RT'] >= RT_trim_min and raw_beh_df.at[line, 'RT'] <= RT_trim_max:
            for label in labels:
                value_container[label + '_valid'] += 1
                value_container[label + '_total'] += 1
        else:
            for label in labels:
                    value_container[label + '_total'] += 1
    for lab in valid_log:
        if lab != 'subjects':
            if value_container[lab + '_total'] == 0:
                valid_log[lab].append(0)
            else:
                valid_log[lab].append(value_container[lab + '_valid']/value_container[lab + '_total'])
    return valid_log



def _get_correct_rates(raw_beh_df, acc_log, beh_compare, pool_all, exclude_invalid, RT_trim_min, RT_trim_max):
    """Extract rates of correct responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    acc_log : dict of lists
        A dictionary in which to write responses rates.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    exclude_invalid : bool
        Whether to exclude invalid responses.
    RT_trim_min : float
        Minimal RT in miliseconds.
    RT_trim_max : float
        Maximal RT in miliseconds.
        
    Returns
    -------
    acc_log : dict of lists
        Log filled with correct responses rates.
    """
    
    value_container = OrderedDict()
    for lab in acc_log:
        if lab != 'subjects':
            value_container[lab + '_corr'] = 0
            value_container[lab + '_total'] = 0
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        if raw_beh_df.at[line, 'response'] == 'correct' and raw_beh_df.at[line, 'RT'] >= RT_trim_min and raw_beh_df.at[line, 'RT'] <= RT_trim_max:
            for label in labels:
                value_container[label + '_corr'] += 1
                value_container[label + '_total'] += 1
        elif exclude_invalid:           
            if raw_beh_df.at[line, 'response'] == 'incorrect' and raw_beh_df.at[line, 'RT'] >= RT_trim_min and raw_beh_df.at[line, 'RT'] <= RT_trim_max:
                for label in labels:
                    value_container[label + '_total'] += 1
        else:
            for label in labels:
                value_container[label + '_total'] += 1
    for lab in acc_log:
        if lab != 'subjects':
            if value_container[lab + '_total'] == 0:
                acc_log[lab].append(None)
            else:
                acc_log[lab].append(value_container[lab + '_corr']/value_container[lab + '_total'])
    return acc_log



def _get_rates(raw_beh_df, rates_log, beh_compare, resps_types, pool_all, exclude_invalid, RT_trim_min, RT_trim_max):
    """Extract rates of particular responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    rates_log : dict of lists
        A dictionary in which to write responses rates.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    resps_types : list of str
        List of responses to count.
    pool_all : bool
        Whether to extract information pooled across all conditions.
    exclude_invalid : bool
        Whether to exclude invalid responses.
    RT_trim_min : float
        Minimal RT in miliseconds.
    RT_trim_max : float
        Maximal RT in miliseconds.
        
    Returns
    -------
    rates_log : dict of lists
        Log filled with responses rates.
    """
    
    value_container = OrderedDict()
    for lab in beh_compare:
        value_container[lab + '_total'] = 0
        for resp in resps_types:
            value_container[lab + '_' + resp] = 0
    labels = [] 
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        if raw_beh_df.at[line, 'response'] in resps_types and raw_beh_df.at[line, 'RT'] >= RT_trim_min and raw_beh_df.at[line, 'RT'] <= RT_trim_max:
            for label in labels:
                value_container[label + '_' + raw_beh_df.at[line, 'response']] += 1
                value_container[label + '_total'] += 1
        elif exclude_invalid: 
            if raw_beh_df.at[line, 'response'] not in resps_types and raw_beh_df.at[line, 'RT'] >= RT_trim_min and raw_beh_df.at[line, 'RT'] <= RT_trim_max:
                for label in labels:
                    value_container[label + '_total'] += 1
        else:
            for label in labels:
                value_container[label + '_total'] += 1
    for lab in beh_compare:
        if lab != 'subjects':
            if value_container[lab + '_total'] == 0:
                for resp in resps_types:
                    rates_log[lab + '_' + resp].append(None)
            else:
                for resp in resps_types:
                    rates_log[lab + '_' + resp].append(value_container[lab + '_' + resp]/value_container[lab + '_total'])
    return rates_log



def _get_hit_fa_rates(raw_beh_df, hit_fa_log, beh_compare, signal, noise, pool_all):
    """Calculate rates of hits and false alarms.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    hit_fa_log : dict of lists
        A dictionary in which to write responses rates.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    signal : dict
        A dict of factors with conditions specifying signal category.
    noise : dict
        A dict of factors with conditions specifying noise category.    
    pool_all : bool
        Whether to extract information pooled across all conditions.
        
    Returns
    -------
    hit_fa_log : dict of lists
        Log filled with responses rates.
    """
    
    value_container = OrderedDict()
    for lab in beh_compare:
        value_container[lab + '_hits'] = 0
        value_container[lab + '_fas'] = 0
        hit_fa_log[lab + '_signal'].append(0)
        hit_fa_log[lab + '_noise'].append(0)
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        if all([raw_beh_df.at[line, factor] in signal[factor] for factor in signal]):
            if raw_beh_df.at[line, 'response'] =='correct':
                for label in labels:
                    hit_fa_log[label + '_signal'][-1] += 1
                    value_container[label + '_hits'] += 1
            elif raw_beh_df.at[line, 'response'] =='incorrect':
                for label in labels:
                    hit_fa_log[label + '_signal'][-1] += 1
        elif all([raw_beh_df.at[line, factor] in noise[factor] for factor in noise]):
            if raw_beh_df.at[line, 'response'] =='correct':
                for label in labels:
                    hit_fa_log[label + '_noise'][-1] += 1
            elif raw_beh_df.at[line, 'response'] =='incorrect':
                for label in labels:
                    hit_fa_log[label + '_noise'][-1] += 1
                    value_container[label + '_fas'] += 1
    for lab in beh_compare:
        hit_fa_log[lab + '_hit_rate'].append(value_container[lab + '_hits'] / hit_fa_log[lab + '_signal'][-1])
        hit_fa_log[lab + '_fa_rate'].append(value_container[lab + '_fas'] / hit_fa_log[lab + '_noise'][-1])
    return hit_fa_log



def _corr_hit_fa_rates(rate, total):
    """Correct rates to avoid 0 and 1 values.

    Parameters
    ----------
    rate : float
        Rate.
    total : int
        Total nuber of valid trials
        
    Returns
    -------
    rate : float
        Corrected rate.
    """
    
    if rate == 0:
        rate += 1/(2 * total)
    elif rate == 1:
        rate -= 1/(2 * total)
    return rate



def _get_raw_RTs(raw_beh_df, beh_compare, resps_types, RT_min, RT_max, pool_all):
    """Extract the lists of RT data.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    beh_compare : dict
        A dictionary containing labels as keys and a dict of conditions for each key.
    RT_min : float
        Minimal RT in miliseconds.
    RT_max : float
        Maximal RT in miliseconds.
    pool_all : bool
        Whether to extract information pooled across all conditions.
       
    Returns
    -------
    raw_RT_container : dict of lists
        Dictionary filled with RT data.
    """
    
    raw_RT_container = OrderedDict()
    if pool_all:
        raw_RT_container['all'] = []
    for lab in beh_compare:
        raw_RT_container[lab] = []
    for line in raw_beh_df.index:
        labels = _get_labels(beh_compare, pool_all, raw_beh_df, line)
        if raw_beh_df.at[line, 'response'] in resps_types and raw_beh_df.at[line, 'RT'] >= RT_min and raw_beh_df.at[line, 'RT'] <= RT_max :
            for label in labels:
                raw_RT_container[label].append(raw_beh_df.at[line, 'RT'])
    return raw_RT_container



class Beh_analizer:
    """Class to analize behavioral data.
    
    Parameters
    ----------
    subjects : list of str
        The list of subject ids.
    path : str
        A path to experiments folder.
    raw_beh_df_folder : str
         A folder with raw behavioral data in xlsx.
    beh_log_folder : str
        A folder to write logs.
        
    Attributes
    ----------
    raw_data : dict
        The dictionary containing raw data as pd.DataFrame. 
    log_folder : str
        A path to beh_log_folder.
    """
    
    def __init__(self, subjects, path, raw_beh_df_folder, beh_log_folder):
        """Init it and create a folder to write logs."""
        
        self.subjects = subjects
        raw_data = OrderedDict()
        filenames = os.listdir(path + raw_beh_df_folder)
        for subj in subjects:
            filename = _get_filename(filenames, subj)
            raw_data[subj] = pd.read_excel(path + raw_beh_df_folder + filename, dtype = object)
        self.raw_data = raw_data
        _make_dir(path, beh_log_folder) 
        self.log_folder = path + beh_log_folder
        
    
    
    def summarize_corr_resps(self, log_name, beh_compare, pool_all = False, save_xlsx = True, save_csv = False):
        """Create a log with summarized correct responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults to False
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        resp_count_df : pd.DataFrame
            Dataframe containing counts of correct responses.
        """
        
        resps_log = OrderedDict()
        resps_log['subjects'] = self.subjects
        if pool_all:
            resps_log['all_total'] = []
            resps_log['all_corr'] = []
            resps_log['all_incorr'] = []
            resps_log['all_invalid'] = []
        for lab in beh_compare:
            resps_log[lab + '_total'] = []
            resps_log[lab + '_corr'] = []
            resps_log[lab + '_incorr'] = []
            resps_log[lab + '_invalid'] = []
        for subj in self.subjects:
            resps_log = _count_corr_resps(self.raw_data[subj], resps_log, beh_compare, pool_all)
        _save_log(resps_log, self.log_folder, log_name, save_xlsx, save_csv)        
        resp_count_df = pd.DataFrame(resps_log, columns = list(resps_log.keys())) 
        return resp_count_df


    
    def summarize_resps_types(self, log_name, beh_compare, resps_types, pool_all = False, save_xlsx = True, save_csv = False):
        """Create a log with summarized counts of particular responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        resps_types : list of str
            List of responses to count.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults is to False
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        resp_count_df : pd.DataFrame
            Dataframe containing counts of responses.
        """
        
        resps_log = OrderedDict()
        resps_log['subjects'] = self.subjects
        if pool_all:
            resps_log['all_total'] = []
            for resp in resps_types:
                resps_log['all_' + resp] = []
            resps_log['all_invalid'] = []
        for lab in beh_compare:
            resps_log[lab + '_total'] = []
            for resp in resps_types:
                resps_log[lab + '_' + resp] = []
            resps_log[lab + '_invalid'] = []
        for subj in self.subjects:
            resps_log = _count_resps_types(self.raw_data[subj], resps_log, beh_compare, resps_types, pool_all)
        _save_log(resps_log, self.log_folder, log_name, save_xlsx, save_csv)      
        resp_count_df = pd.DataFrame(resps_log, columns = list(resps_log.keys())) 
        return resp_count_df

    

    def count_time_trim_resps (self, log_name, beh_compare, RT_min, RT_max, pool_all = False, save_xlsx = True, save_csv = False):
        """Create a log with summarized counts of trials with too early or too late responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        RT_min : float
            Minimal RT in miliseconds.
        RT_max : float
            Maximal RT in miliseconds.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults is to False
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        time_trim_df : pd.DataFrame
            Dataframe containing counts of too early or too late responses.
        """
        
        time_trim_log = OrderedDict()
        time_trim_log['subjects'] = self.subjects
        if pool_all:
            time_trim_log['all_total'] = []
            time_trim_log['all_valid'] = []
            time_trim_log['all_too_early'] = []
            time_trim_log['all_too_late'] = []
        for lab in beh_compare:
            time_trim_log[lab + '_total'] = []
            time_trim_log[lab + '_valid'] = []
            time_trim_log[lab + '_too_early'] = []
            time_trim_log[lab + '_too_late'] = []
        for subj in self.subjects:
            time_trim_log = _count_too_early(self.raw_data[subj], time_trim_log, beh_compare, RT_min, RT_max, pool_all)
        _save_log(time_trim_log, self.log_folder, log_name, save_xlsx, save_csv)   
        time_trim_df = pd.DataFrame(time_trim_log, columns = list(time_trim_log.keys())) 
        return time_trim_df 



    def calculate_valid_rates(self, log_name, beh_compare, pool_all = False, RT_trim_min = 0, RT_trim_max = 10e10, save_xlsx = True, save_csv = False):
        """Create a log with rates of valid responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults to False.
        RT_trim_min : float
            Minimal RT in miliseconds.
            Defaults to 0.
        RT_trim_max : float
            Maximal RT in miliseconds.
            Defaults to 10e10.
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        valid_df : pd.DataFrame
            Dataframe containing rates of valid responses.
        """
        
        valid_log = OrderedDict()
        valid_log['subjects'] = self.subjects
        if pool_all:
            valid_log['all'] = []
        for lab in beh_compare:
            valid_log[lab] = []
        for subj in self.subjects:
            valid_log = _get_valid_rates(self.raw_data[subj], valid_log, beh_compare, pool_all, RT_trim_min, RT_trim_max)  
        _save_log(valid_log, self.log_folder, log_name, save_xlsx, save_csv) 
        valid_df = pd.DataFrame(valid_log, columns = list(valid_log.keys())) 
        return valid_df



    def calculate_accuracy(self, log_name, beh_compare, pool_all = False, exclude_invalid = True, RT_trim_min = 0, RT_trim_max = 10e10, 
                           save_xlsx = True, save_csv = False):
        """Create a log with rates of correct responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults to False.
        exclude_invalid : bool
            Whether to exclude invalid responses.
            Defaults to True.
        RT_trim_min : float
            Minimal RT in miliseconds.
            Defaults to 0.
        RT_trim_max : float
            Maximal RT in miliseconds.
            Defaults to 10e10.
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        accuracy_df : pd.DataFrame
            Dataframe containing rates of correct responses.
        """
        
        acc_log = OrderedDict()
        acc_log['subjects'] = self.subjects
        if pool_all:
            acc_log['all'] = []
        for lab in beh_compare:
            acc_log[lab] = []
        for subj in self.subjects:
            acc_log = _get_correct_rates(self.raw_data[subj], acc_log, beh_compare, pool_all, exclude_invalid, RT_trim_min, 
                                         RT_trim_max) 
        _save_log(acc_log, self.log_folder, log_name, save_xlsx, save_csv) 
        accuracy_df = pd.DataFrame(acc_log, columns = list(acc_log.keys()))
        return accuracy_df



    def calculate_rates(self, log_name, beh_compare, resps_types, pool_all = False, exclude_invalid = True, RT_trim_min = 0, RT_trim_max = 10e10, 
                           save_xlsx = True, save_csv = False):
        """Create a log with rates of particular responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        resps_types : list of str
            List of responses to count.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults to False.
        exclude_invalid : bool
            Whether to exclude invalid responses.
            Defaults to True.
        RT_trim_min : float
            Minimal RT in miliseconds.
            Defaults to 0.
        RT_trim_max : float
            Maximal RT in miliseconds.
            Defaults to 10e10.
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        rates_df : pd.DataFrame
            Dataframe containing rates of responses.
        """
        
        rates_log = OrderedDict()
        rates_log['subjects'] = self.subjects
        if pool_all:
            for resp in resps_types:
                rates_log['all' + '_' + resp] = []
        for lab in beh_compare:
            for resp in resps_types:
                rates_log[lab + '_' + resp] = []
        for subj in self.subjects:
            rates_log = _get_rates(self.raw_data[subj], rates_log, beh_compare, resps_types, pool_all, exclude_invalid, 
                                   RT_trim_min, RT_trim_max)
        _save_log(rates_log, self.log_folder, log_name, save_xlsx, save_csv) 
        rates_df = pd.DataFrame(rates_log, columns = list(rates_log.keys())) 
        return rates_df

    

    def calculate_sdt_params(self, log_name, beh_compare, signal, noise, pool_all = False, d_corr = False, save_hit_fa_log = False, 
                             save_xlsx = True, save_csv = False):
        """Create a log with d' and criterion.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        signal : dict
            A dict of factors with conditions specifying signal category.
        noise : dict
            A dict of factors with conditions specifying noise category.    
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults to False.
        d_corr : bool
            Whether to correct for extreme rates.
            Defaults to False.
        save_hit_fa_log : bool
            Whether save the log containing hits and false alarms rates.
            Defaults to False.
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        d_df : pd.DataFrame
            Dataframe containing d' indexes.
        c_df : pd.DataFrame
            Dataframe containing criterion.
        hit_fa_df : pd.DataFrame
            Dataframe containing rates of hits and false alarms.
        """
        
        hit_fa_log = OrderedDict()
        d_log = OrderedDict()
        c_log = OrderedDict()
        hit_fa_log['subjects'] = self.subjects
        d_log['subjects'] = self.subjects
        c_log['subjects'] = self.subjects
        if pool_all:
            hit_fa_log['all_hit_rate'] = []
            hit_fa_log['all_signal'] = []
            hit_fa_log['all_fa_rate'] = []
            hit_fa_log['all_noise'] = []
            d_log['all'] = []
            c_log['all'] = []
        for lab in beh_compare:
            hit_fa_log[lab + '_hit_rate'] = []
            hit_fa_log[lab + '_signal'] = []
            hit_fa_log[lab + '_fa_rate'] = []
            hit_fa_log[lab + '_noise'] = []
            d_log[lab] = []
            c_log[lab] = []
        for subj in self.subjects:
            hit_fa_log = _get_hit_fa_rates(self.raw_data[subj], hit_fa_log, beh_compare, signal, noise, pool_all)
            for lab in beh_compare:
                hit_rate = hit_fa_log[lab + '_hit_rate'][-1]
                fa_rate = hit_fa_log[lab + '_fa_rate'][-1]
                if d_corr:
                    hit_rate = _corr_hit_fa_rates(hit_rate, hit_fa_log[lab + '_signal'][-1])
                    fa_rate = _corr_hit_fa_rates(fa_rate, hit_fa_log[lab + '_noise'][-1])
                d_log[lab].append(stats.norm.ppf(1 - fa_rate) - stats.norm.ppf(1 - hit_rate))
                c_log[lab].append(stats.norm.ppf(1 - fa_rate) - d_log[lab][-1]/2)
        _save_log(d_log, self.log_folder, 'd_' + log_name, save_xlsx, save_csv)
        _save_log(c_log, self.log_folder, 'c_' + log_name, save_xlsx, save_csv)
        if save_hit_fa_log:
            _save_log(hit_fa_log, self.log_folder, 'hit_fa_' + log_name, save_xlsx, save_csv)
        d_df = pd.DataFrame(d_log, columns = list(d_log.keys())) 
        c_df = pd.DataFrame(c_log, columns = list(c_log.keys())) 
        hit_fa_df = pd.DataFrame(hit_fa_log, columns = list(hit_fa_log.keys()))
        return d_df, c_df, hit_fa_df



    def aggregate_RTs (self, log_name, beh_compare, RT_min, RT_max, resps_types = ['correct'], method = 'median', pool_all = False, save_xlsx = True, save_csv = False):
        """Create a log with aggregated reaction times.

        Parameters
        ----------
        log_name : str
            The name of the log.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        beh_compare : dict
            A dictionary containing labels as keys and a dict of conditions for each key.
        RT_min : float
            Minimal RT in miliseconds.
        RT_max : float
            Maximal RT in miliseconds.
        resps_types : list of str
            List of responses to count.
            Defaults to ['correct'].
        method : str
            Aggregation method.
            possible values are 'mean', 'median', 'ex_gauss'.
            Defaults to 'median'.
        pool_all : bool
            Whether to extract information pooled across all conditions.
            Defaults is to False
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        agg_RT_df : pd.DataFrame
            Dataframe containing aggregated RTs.
        """
        
        agg_RT_log = OrderedDict()
        agg_RT_log['subjects'] = self.subjects
        if method == 'ex_gauss':
            if pool_all:
                agg_RT_log['all_miu'] = [] 
                agg_RT_log['all_sigma'] = [] 
                agg_RT_log['all_tau'] = [] 
            for lab in beh_compare:
                agg_RT_log[lab + '_miu'] = [] 
                agg_RT_log[lab + '_sigma'] = [] 
                agg_RT_log[lab + '_tau'] = [] 
        else:
            if pool_all:
                agg_RT_log['all'] = []
            for lab in beh_compare:
                agg_RT_log[lab] = []
        for subj in self.subjects:
            raw_RT_container = _get_raw_RTs(self.raw_data[subj], beh_compare, resps_types, RT_min, RT_max, pool_all)
            for lab in raw_RT_container:
                if method == 'mean':     
                    agg_RT_log[lab].append(np.mean(raw_RT_container[lab]))
                elif method == 'median':     
                    agg_RT_log[lab].append(np.median(raw_RT_container[lab]))
                elif  method == 'ex_gauss':
                    K, loc, scale = stats.exponnorm.fit(raw_RT_container[lab], 5, loc = 50, scale = 100)
                    agg_RT_log[lab + '_miu'].append(loc)
                    agg_RT_log[lab + '_sigma'].append(scale)
                    agg_RT_log[lab + '_tau'].append(K * scale)
        _save_log(agg_RT_log, self.log_folder, method + '_' + log_name, save_xlsx, save_csv)    
        agg_RT_df = pd.DataFrame(agg_RT_log, columns = list(agg_RT_log.keys())) 
        return agg_RT_df



def _sum_surv_resps (raw_beh_df, resps_log):
    """Extract survey responses.

    Parameters
    ----------
    raw_beh_df : pd.DataFrame
        DataFrame containing behavioral log.
    resps_log : dict of lists
        A dictionary in which to write responses.
    
    Returns
    -------
    resps_log : dict of lists
        A log filled with responses.
    """
    
    for line in raw_beh_df.index:
        resps_log[raw_beh_df.at[line, 'quest_content']].append(raw_beh_df.at[line, 'resp_content'])
    return resps_log    



class Surv_analizer:
    """Class to analize survey data.
    
    Parameters
    ----------
    subjects : list of str
        The list of subject ids.
    path : str
        A path to experiments folder.
    raw_beh_df_folder : str
         A folder with raw behavioral data in xlsx.
    beh_log_folder : str
        A folder to write logs.
        
    Attributes
    ----------
    raw_data : dict
        The dictionary containing raw data as pd.DataFrame. 
    log_folder : str
        A path to beh_log_folder.
    """
    
    def __init__(self, subjects, path, raw_beh_df_folder, beh_log_folder):
        """Init it and create a folder to write logs."""
        self.subjects = subjects
        raw_data = OrderedDict()
        filenames = os.listdir(path + raw_beh_df_folder)
        for subj in subjects:
            filename = _get_filename(filenames, subj)
            raw_data[subj] = pd.read_excel(path + raw_beh_df_folder + filename, dtype = object)
        self.raw_data = raw_data
        _make_dir(path, beh_log_folder) 
        self.log_folder = path + beh_log_folder


    
    def summarize_resps(self, log_name, quests, save_xlsx = True, save_csv = False):
        """Create a log with summarized survey responses.

        Parameters
        ----------
        log_name : str
            The name of the log.
        quests : dict
            A dictionary survey questions labels and content.
        save_xlsx : bool
            Whether to save log in xlsx format.
            Defaults to True.
        save_csv : bool
            Whether to save log in csv format.
            Defaults to False.
            
        Returns
        -------
        resp_sum_df : pd.DataFrame
            Dataframe containing survey responses.
        """
        resps_log = OrderedDict()
        resps_log['subjects'] = self.subjects
        for quest in quests:
            resps_log[quests[quest]['content']] = []
        for subj in self.subjects:
            resps_log = _sum_surv_resps(self.raw_data[subj], resps_log)
        _save_log(resps_log, self.log_folder, log_name, save_xlsx, save_csv)  
        resp_sum_df = pd.DataFrame(resps_log, columns = list(resps_log.keys()))
        return resp_sum_df

