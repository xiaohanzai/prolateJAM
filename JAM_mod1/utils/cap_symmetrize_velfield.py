#######################################################################
#
# Copyright (C) 2004-2014, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#######################################################################

import numpy as np
from scipy import interpolate

#----------------------------------------------------------------------

def _rotate_points(x, y, ang):
    """
    Rotates points conter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013
    
    """
    theta = np.radians(ang - 90.)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)
    return xNew, yNew
    
#----------------------------------------------------------------------
    
def symmetrize_velfield(xbin, ybin, velBin, sym=2, pa=90.):
    """
    This routine generates a bi-symmetric ('axisymmetric') 
    version of a given set of kinematical measurements.
    PA: is the angle in degrees, measured counter-clockwise,
      from the vertical axis (Y axis) to the galaxy major axis.
    SYM: by-simmetry: is 1 for (V,h3,h5) and 2 for (sigma,h4,h6)

    HISTORY:

    V1.0: Michele Cappellari, Vicenza, 21 May 2004
    V1.01: Added MISSING keyword to TRIGRID call. Flipped velocity sign.
      Written basic documentation. MC, Leiden, 25 May 2004
    V1.1: Included point-symmetric case. Remco van den Bosch, Leiden, 18 January 2005
    V1.11: Minor code revisions. MC, Leiden, 23 May 2005
    V1.12: Important: changed definition of PA to be measured counterclockwise
      with respect to the positive Y axis, as in astronomical convention and
      consistently with my FIND_GALAXY routine. MC, Leiden, 1 June 2005
    V1.13: Added optional keyword TRIANG. Corrected rare situation with w=-1.
      MC, Leiden, 2 June 2005
    V1.14: Added prefix SYMM_ to internal functions to prevent conflicts
      with external functions with the same name. MC, Oxford, 11 May 2007
    V2.0 : Completely rewritten without any loop. MC, Oxford, 8 October 2013  
    V2.01: Uses TOLERANCE keyword of TRIANGULATE to try to avoid IDL error
      "TRIANGULATE: Points are co-linear, no solution." MC, Oxford, 2 December 2013
    V3.0 : Translated from IDL into Python. MC, Oxford, 14 February 2014
    
    """        
    xbin = np.asarray(xbin)
    ybin = np.asarray(ybin)
    velBin = np.asarray(velBin)
    x, y = _rotate_points(xbin, ybin, -pa)  # Negative PA for counter-clockwise
    
    xyIn = np.column_stack([x, y])
    xout = np.hstack([x,-x, x,-x])
    yout = np.hstack([y, y,-y,-y])
    xyOut = np.column_stack([xout, yout])
    velOut = interpolate.griddata(xyIn, velBin, xyOut)
    velOut = velOut.reshape(4, xbin.size)
    
    if sym == 1:
        velOut[[1,3],:] *= -1.
    velSym = np.nanmean(velOut, axis=0)
    return velSym.copy()

#----------------------------------------------------------------------
