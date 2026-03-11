9#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:17:00 2025

@author: nclark
"""

# the purpose of this code is to convert a .ipac file for the old spline continuum code, to a 
# .txt file that when loaded in using np.loadtxt (or equivalent), will be compatible with the new code

#  your spline after doing this should look very similar (if not identical) to something from the old code



# my version methods
# 0
# ap_y is whatever the index of the ap_x corresponds to

# 1 
# ap_y is a median of indices ap_x - ext to ap_x + ext + 1, i.e. 2*ext + 1 points centred on ap_x index

# 2 
# ap_y is median hard-coded for ext=10, as a convinient shortcut for 'noisy data' or fringes.

# 3
# ap_y is the value of ap_x on a linear function, determined using the lower and upper bounds with y vals a median using ext.

# 4
# ap_y is a median of the lower and upper bounds using ext (case of 3 where slope is 0)

# not converted ipac things

# on plateau = True:
# groups anchor points in 2. First is lb, second is ub, straight line between them.
# makes global k=3 spline, replaces any plateau=True bits with the k=1 determined independantly

# make 2 versions depending on if bumps true and plat, to spit out a local and global verison? yes

# need a version with all points (local) and a version without any bumps == True



'''
IMPORTING MODULES
'''



# standard stuff
import numpy as np

# warning supression
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings(action='ignore', message='Mean of empty slice')



'''
CODE
'''



def ipac_to_txt(file_loc, ipac_name, txt_name):
    
    with open(file_loc + ipac_name) as f:
        # excludes header lines, and \n from each line
        # ipac_lines = [' '.join(line[:-1].split()) for line in f if '|' not in line]
        ipac_lines = [line[:-1] for line in f if '|' not in line]
    
    N = len(ipac_lines)
    
    # isolating each column
    x0 = [float(line.split()[0]) for line in ipac_lines]
    moment = [int(line.split()[1]) for line in ipac_lines]
    on_plateau = [line.split()[2] for line in ipac_lines]
    x0_min = [float(line.split()[3]) for line in ipac_lines]
    x0_max = [float(line.split()[4]) for line in ipac_lines]
    bumps = [line.split()[5] for line in ipac_lines]
    
    # for now, ignore on_plateau and bumps
    
    # x0 -> ap_x (identical formatting) (col 0 -> col 0)
    # moment -> method (need to convert int meaning, some will involve ext) (col 1 -> col 1)
    # on plateau (not used atm)
    # x0_min -> ap_lb (can be interpreted the same) (col 3 -> col 3)
    # x0_max -> ap_ub (can be interpreted the same) (col 4 -> col 4)
    
    # new columns
    ap_x = x0
    method = np.zeros(N).astype(np.int64)
    ext = np.zeros(N).astype(np.int64)
    ap_lb = x0_min
    ap_ub = x0_max
    
    # 0, bumps = True
    # y val is min in range of x0_min and x0_max. 
    # no current equivalent in my version
    
    for i in range(N):
        if moment[i] == 1:
            # # ap_y is mean of 3 data points. 
            # method[i], ext[i] = 1, 1
            # my version makes this moment method 0 with no mean
            method[i] = 0
        elif moment[i] == 2:
            # ap_y is mean of 5 data points
            method[i], ext[i] = 1, 2
        elif moment[i] == 3:
            # moment 1 applied. Then applied again +0.3 microns, then the 2 are averaged, wavelength ap_x + 0.15
            ap_lb[i], ap_ub[i] = x0[i], x0[i] + 0.3
            ap_x[i] = x0[i] + 0.15
            method[i], ext[i] = 4, 1
        elif moment[i] == 4:
            # mean of all points between min and max, x0 in middle. most similar to method 2
            ap_x[i] = (x0_min[i] + x0_max[i])/2
            method[i] = 2
        else:
            # any moments that dont have a direct translation, and moment 0
            method[i] = 0
            
    # saving to file
    with open(file_loc + txt_name, 'w') as txt:
        # writing the header line
        txt.write(' ap_x | method | ext | ap_lb | ap_ub\n')
    
        # writing data lines
        for i, x in enumerate(ap_x):
            # variable whitespace for 1 or 10
            ws1, ws2 = '', ''
            if x < 10.0:
                ws1 = ' '
            if ap_lb[i] < 10.0:
                ws2 = ' '
            line_seg1 = '{:.4f}'.format(x) + f'{ws1}   {method[i]}       {ext[i]}   '
            line_seg2 = '{:.4f}'.format(ap_lb[i]) + f'{ws2}  ' + '{:.4f}'.format(ap_ub[i]) + '\n'
            txt.write(line_seg1 + line_seg2)



# example usage



# file_loc = path
# ipac_name = spline.ipac
# txt_name = spline.txt

# ipac_to_txt(file_loc', ipac_name',txt_name)

# .txt file will be in the file_loc directory, same as the .ipac file