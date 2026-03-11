# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 12:21:06 2026

@author: nickjc
"""

import numpy as np
from scipy.interpolate import make_splrep

# input:
# need a col with the anchor point wavelengths, the anchor point method, and lower/upper wave for specific method
# what each method does: 
# 0: y val for the x val
# 1: a small median, 2 points on either side (5 tot)
# 2: a big median, 10 points on either side (21 tot)
# 3: fit to line, by finding slope of lower and upper wave line and taking y from this
# 4: fit to flat line, by finding mean of lower and upper wave as y

'''
general approach:
have x, y be the spline anchor points, with s=0 for interpolation and k=3
this is the same approach as used in bethany's version.

another possibility is x, y represent the snippets of the data meant to be
the continuum to be fit. if this approach is taken, the knots are determined
as opposed to input, the numer of which is controlled by s. big s = less knots
and smoother fit, small s = more knots and less smooth fit. for this mode it 
would be prudent to include a weights array w, which would be 1/stdev(y)
or 1/err (check jans error presentation to verify this)
'''

'''
UnivariateSpline vs make_splrep differences

class UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=None, ext=0, check_finite=False)
make_splrep(x, y, *, w=None, xb=None, xe=None, k=3, s=0, t=None, nest=None, bc_type=None)

bbox is the same as xb, xe
s=None means it is controlled by the len of w. new doesnt have none, other options the same.
float means the weighted least squares fit must be less than s, so bigger s = less good fit, 
less anchor points needed, smoother.
ext is for the extrapolation mode, can return a zero, bound or error besides value.
check_finite: verify no infs or nans to prevent crashes
t is a list of spline knots, must be between 2(k+1) and m+k+1, m=len(y)
nest is the length of the knot vector, which should be betwen 2(k+1) amd m+k+1
periodic makes the spline fit assume the data is periodic with a period of the 
data length, resulting in the spline continuing through the endpoints without discontinuities or cusps
bc_type goes with periodic

US is itself a class, while MS is a function that returns a BSpline instance. 
class BSpline(t, c, k, extrapolate=True, axis=0)[source]
c are spline coeffs ascociated with the 

'''



'''
HELPER FUNCTIONS
'''



# short function to convert wavelengths to indices
def _wave_to_ind(wavelengths, x):
    ind = np.zeros(x.shape).astype(np.int64)
    for i, wave in enumerate(x):
        ind[i] = np.argmin(abs(wavelengths - wave))
    return ind



def _anchor_point(
        data, 
        ap_ind, 
        ap_method=None, 
        ext=None, 
        ap_lb_ind=None, 
        ap_ub_ind=None
        ):
    """
    Calculates the y value of a single anchor point.

    Parameters
    ----------
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_ind : int
        anchor point x index corresponding to the y val to be calculated.
    ap_method : int, optional
        approach to be used for determining the y val.
    ext : int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb_ind : int, optional
        index of lower bound used for some methods.
    ap_ub_ind : int, optional
        index of upper bound used for some methods.
        
    Returns
    -------
    ap_y : int
        y val corresponding to ap_x, together making an anchor point used for spline calculation.
    """    
    # convinience variable
    N = data.shape[0]

    # determining logic of all extra conditions now to avoid excessive nesting
    # trading efficiency for readability of code
    
    # ap_ind cannot be too close to edges
    ec_2 = ap_ind > 9 and ap_ind < N-10
    
    if ext is not None:
        # ap_ind cannot be too close to edges
        ec_ext = ap_ind > ext-1 and ap_ind < N-ext
    else:
        ec_ext = False
        
    if ap_lb_ind is not None and ap_ub_ind is not None:
        # bounds need to be in correct order and not too close to edges
        ec_b = ap_ub_ind > ap_lb_ind and ap_lb_ind > ext-1 and ap_ub_ind < N-ext
    else:
        ec_b = False

    # calculating ap_y
    if ap_method == 1 and ec_ext == True:
        # median using ext
        ap_y = np.nanmedian(data[ap_ind - ext : ap_ind + ext + 1], axis=0)

    elif ap_method == 2 and ec_2 == True:
        # median with 10 on either side of ap_ind, 21 total
        ap_y = np.nanmedian(data[ap_ind - 10 : ap_ind + 11], axis=0)
    
    elif ap_method == 3 and ec_b == True:
        # use bounds to calculate linear function, anchor point on this function.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        # define line in terms of indices
        m = (d2 - d1)/(ap_ub_ind - ap_lb_ind)
        ap_y = m*(ap_ind - ap_lb_ind) + d1
        
    elif ap_method == 4 and ec_b == True: 
        # use bounds to calculate a flat line, anchor point on this line.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        ap_y = (d1 + d2)/2
        
    else:
        # use data[ind] only, unless the conditions are met for something more complex.
        # this is the expected output if ap_method=0, or if the intended args are incorrect.
        ap_y = data[ap_ind]
            
    return ap_y



'''
MAIN FUNCTION
'''



def spline_from_anchor_points(
        wavelengths, 
        data, 
        ap_x, 
        ap_method=None, 
        ext=None, 
        ap_lb=None, 
        ap_ub=None
        ):
    """
    Turns a list of wavelengths into anchor points, and fits a spline to them.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelengths array. Units should match ap_x, ap_lb, ap_ub.
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_x : numpy.ndarray
        anchor point wavelengths. All ap are assumed to have the same length.
    ap_method : list of int, optional
        approach to be used for determining the anchor point y val.
    ext : list of int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb : list of int, optional
        wavelengths of lower bounds used for some methods.
    ap_ub : list of int, optional
        wavelengths of upper bounds used for some methods.
        
    Returns
    -------
    spl : numpy.ndarray
        spline fits applied to wavelengths, same shape as data.
    """   
    # shape of data
    shape_y, shape_x = data[0].shape
    M = ap_x.shape[0]
    
    # data needs to have no nans, leave data intact 
    data_nonan = np.copy(data)
    data_nonan[np.isnan(data_nonan)] = 0
    
    # convert ap_x to ap_ind
    ap_ind = _wave_to_ind(wavelengths, ap_x)
    
    # need both ap_lb and ap_ub to not be None for their routines to function
    if ap_lb is not None and ap_ub is not None:
        ap_lb_ind = _wave_to_ind(wavelengths, ap_lb)
        ap_ub_ind = _wave_to_ind(wavelengths, ap_ub)
    else:
        ap_lb_ind = [None]*M
        ap_ub_ind = [None]*M
    
    # make ap_method iterable if it is None
    if ap_method is None:
        ap_method = [None]*M
        
    # make ext iterable if it is None
    if ext is None:
        ext = [None]*M
    
    # calculating the anchor point y vals
    ap_y = np.zeros((M, shape_y, shape_x))
    # each input is a list/ndarray of the same length
    for w, ind in enumerate(ap_ind):
        ap_y[w] = _anchor_point(
            data_nonan, 
            ind, 
            ap_method=ap_method[w], 
            ext=ext[w], 
            ap_lb_ind=ap_lb_ind[w], 
            ap_ub_ind=ap_ub_ind[w]
            )
    
    # calculate BSpline instance
    spl_func = make_splrep(ap_x, ap_y)
    spl = spl_func(wavelengths)
    return spl, spl_func
