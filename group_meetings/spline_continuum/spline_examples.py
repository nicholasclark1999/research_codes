# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:18:31 2026

@author: nickjc
"""

import numpy as np
import matplotlib.pyplot as plt
import spline_continuum as spc

# IMPORTANT NOTE

# as one of its conditions used by scipy's Bspline, first and second/ second last and last B splines
# are the same polynomial, so theres no knot between these internally.
# but, these points are still used in the interp with regards to the spline through those points.
# so, for ex1 there wont be a knot at 3.5 or 14, but the resulting spline will go through these
# points as though there was

# play data to fit splines to
x = np.arange(3, 28, 0.01)
y = ((x**2)/50)*(np.sin(x/12))**2 + 3*np.exp(-1*((x-10)/2)**2)

# Example 1
# Default Case 

ap1 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14, 15])
spline1, spl_func1 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap1)
# note the first thing returned by the function is equal to spl_func1(x)
spline1 = spline1[:,0,0] 

# want to extract the B splines and coeffs to check if spline1 is what we think it is
knot1 = spl_func1.t
coeff1 = spl_func1.c[:, 0, 0]
k1 = spl_func1.k

# applying knowledge of where the B spline is 0
bs1 = np.zeros((len(coeff1), len(x)))
for i, c in enumerate(coeff1):
    temp = c*spl_func1.basis_element(knot1[i:i+5])(x)
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



# Example 2
# adding an anchor point near the end of the list (14.5)

ap2 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14, 14.5, 15])
spline2, spl_func2 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap2)
spline2 = spline2[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot([5.5, 5.5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)

# expect differences to emerge at 5.5, but dont start to see them until around 8, why? examine B splines
knot2 = spl_func2.t
coeff2 = spl_func2.c[:, 0, 0]
k2 = spl_func2.k

bs2 = np.zeros((len(coeff2), len(x)))
for i, c in enumerate(coeff2):
    temp = c*spl_func2.basis_element(knot2[i:i+5])(x)
    temp[x<knot2[i]] = 0
    temp[x>knot2[i+4]] = 0
    bs2[i] = temp

# B_{5.5,3}(x)
plt.figure()
plt.plot(x, bs1[7])
plt.plot(x, bs2[7])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([5.5, 5.5], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')

# B_{6.0,3}(x)
plt.figure()
plt.plot(x, bs1[8])
plt.plot(x, bs2[8])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([6, 6], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')

# B_{13.0,3}(x)
plt.figure()
plt.plot(x, bs1[9])
plt.plot(x, bs2[9])
plt.ylim(-0.1, 2.5)
plt.xlim(4.5, 15.5)
plt.plot([13.0, 13.0], [0, 6], color='black', linestyle='dashed')
plt.plot([15, 15], [0, 6], color='black', linestyle='dashed')



# Example 3
# instead of adding, changing anchor point 14 to 14.5
# note that anchor points 3.5 amd 14.5 discarded, so this will have an identical knots vector example 1 
ap3 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.5, 14.5, 15])
spline3, spl_func3 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap3)
spline3 = spline3[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot([5, 5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)



# Example 4
# adding an anchor point in the middle, closer to the feature
ap4 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 13, 13.2, 13.5, 14, 15])
spline4, spl_func4 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap4)
spline4 = spline4[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot(x, spline4)
plt.plot([5, 5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)



# Example 5
# adding an anchor point on the feature (ask yourself if what you get is what you expected or not)

ap5 = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 10, 13, 13.5, 14, 15])
spline5, spl_func5 = spc.spline_from_anchor_points(x, y[:, np.newaxis, np.newaxis], ap5)
spline5 = spline5[:,0,0]

plt.figure()
plt.plot(x, y)
plt.plot(x, spline1)
plt.plot(x, spline2)
plt.plot(x, spline3)
plt.plot(x, spline4)
plt.plot(x, spline5)
plt.plot([4.5, 4.5], [0, 6], color='black', linestyle='dashed')
plt.xlim(3, 15)
plt.ylim(0, 6)
