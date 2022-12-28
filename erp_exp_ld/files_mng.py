# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:57:14 2021

@author: Lucja Doradzinska <l.doradzinska@nencki.edu.pl>
"""

import os
import csv
import pandas as pd
from collections import OrderedDict 



def _make_dir(path, new_folder):
    """Create a directory if not existing.

    Parameters
    ----------
    path : str
        A path to experiments folder.
    new_folder : str
        The folder to create.
    """  
    os.chdir(path)
    try:
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
    except OSError:
        print ('Error: Creating directory. ' + new_folder) 



def _save_log(log, path, savename, save_xlsx, save_csv):
    """Save log.

    Parameters
    ----------
    log : dict
        A dictionary containing column labels as keys and lists of values to write in a column.
        Lists of values must be the same length.
    path : str
        A path to experiments folder.
    save_xlsx : bool
        Whether to save log in xlsx format.
    save_csv : bool
        Whether to save log in csv format.
    """  
    
    if save_xlsx:
        df = pd.DataFrame(log, columns = list(log.keys())) 
        df.to_excel (path + savename + '.xlsx', index = False, header=True) 
    if save_csv:
        os.chdir(path)
        csv_columns = log.keys()
        try:
            with open(savename + '.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
                writer.writeheader()
                for i in range(len(df)):
                    row = OrderedDict()
                    for c in csv_columns:
                        row[c] = log[c][i]
                    writer.writerow(row)
        except IOError:
            print("I/O error") 



def _get_filename(filenames, subj):
    """Get filename for particular subject.

    Parameters
    ----------
    filenames : list of str
        A list of filenames.
    subj : str
        Subjects id.
        
    Returns
    -------
    filename : str
        Filename for subject.
    """  
    
    for name in filenames:
        if subj in name:
            filename = name
    return filename
