# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:18:31 2026

@author: nickjc
"""

import numpy as np
import matplotlib.pyplot as plt
import spline_continuum as spc

from scipy.interpolate import BSpline

# Example 1. 

x = np.arange(3, 28, 0.01)
y = ((x**2)/50)*(np.sin(x/12))**2 + 3*np.exp(-1*((x-10)/2)**2)

ap1 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14, 15])
spl1 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap1)
spline1 = spl1(x)[:,0,0]

# want to extract the basis elements
# important: interpolate spline makes first/2nd and last/2nd last polynomials 
# the same on interval, so knot touching each endpoint is discarded to achieve this.
knot1 = spl1.t
coeff1 = spl1.c[:, 0, 0]
k1 = spl1.k

bs1 = np.zeros((len(coeff1), len(x)))
for i, c in enumerate(coeff1):
    temp = c*spl1.basis_element(knot1[i:i+5])(x)
    temp[x<knot1[i]] = 0
    temp[x>knot1[i+4]] = 0
    bs1[i] = temp

# confirming sum of bsplines is the same as spline
plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.xlim(3, 15)
plt.ylim(0, 6)
plt.plot(x, np.sum(bs1, axis=0), linestyle='dotted', color='black')
plt.savefig('Figures/ex1.png', bbox_inches='tight') 

#%%

# Example 2

# adding an anchor point near the end

ap2 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14, 14.5, 15])
spl2 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap2)
spline2 = spl2(x)[:,0,0]

# want to extract the basis elements
knot2 = spl2.t

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot([5.5, 5.5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)
plt.savefig('Figures/ex2.png', bbox_inches='tight') 

# expect differences to emerge at 5.5, but dont start to see them until around 8, why?
coeff2 = spl2.c[:, 0, 0]
k2 = spl2.k

bs2 = np.zeros((len(coeff2), len(x)))
for i, c in enumerate(coeff2):
    temp = c*spl2.basis_element(knot2[i:i+5])(x)
    temp[x<knot2[i]] = 0
    temp[x>knot2[i+4]] = 0
    bs2[i] = temp

#%%
plt.figure()
plt.plot(x, bs1[7])
plt.plot(x, bs2[7])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([5.5, 5.5], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')
plt.savefig('Figures/ex2_1.png', bbox_inches='tight') 

#%%
plt.figure()
plt.plot(x, bs1[8])
plt.plot(x, bs2[8])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([6, 6], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')
plt.savefig('Figures/ex2_2.png', bbox_inches='tight') 

#%%
plt.figure()
plt.plot(x, bs1[9])
plt.plot(x, bs2[9])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([13.0, 13.0], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')
plt.savefig('Figures/ex2_3.png', bbox_inches='tight') 
#%%
# Example 3

# instead of adding, changing anchor point 14 to 14.5
ap3 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14.5, 15])
spl3 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap3)
spline3 = spl3(x)[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot([5, 5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)
plt.savefig('Figures/ex3.png', bbox_inches='tight') 
#%%
# Example 4

# adding an anchor point in the middle
ap4 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.2, 13.5, 14, 15])
spl4 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap4)
spline4 = spl4(x)[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot(x, spline4)
plt.plot([5, 5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)
plt.savefig('Figures/ex4.png', bbox_inches='tight') 
#%%
# Example 5

# adding an anchor point on the feature (is it behaving as expected?)
ap5 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 10, 13, 13.5, 14, 15])
spl5 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap5)
spline5 = spl5(x)[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot(x, spline4)
plt.plot(x, spline5)
plt.plot([5, 5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)
plt.savefig('Figures/ex5.png', bbox_inches='tight') 
