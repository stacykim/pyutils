# sfh.py
# created 2020.03.20 by stacy kim

from numpy import *


def sfr_pre(vmax,method='fiducial'):
    if   method == 'fiducial':   return 10**(6.95*log10(vmax)-11.6)  # using EDGE originals only w/my SFR and vmax (via NFW fit + time avg + smoothing) calculations
    elif method == 'maxfilter':  return 10**(5.23*log10(vmax)-10.2)  # using EDGE orig + GMOs w/maxfilt vmax, SFR from 0,t(zre)
    else:
        print('Do not recognize sfr_pre method',method,'.  Aborting...')
        exit()


def sfr_post(vmax):
    return 7.06 * (vmax/182.4)**3.07 * exp(-182.4/vmax)  # schechter fxn fit                                                                                        


def sfh(t, z, vmax, vthres=26., zre=4.,binning='3bins',sfr_pre_method='fiducial'):
    """
    Assumes we are given a halo's entire vmax trajectory.  Data must be given s.t. time t increases and starts at t=0.

    binning options:
       'all' sim points
       '2bins' pre/post reion, with pre-SFR from <vmax(z>zre)>, post-SFR from vmax(z=0)
       '3bins' which adds SFR = 0 phase after reion while vmax < vthres
    """
    dt = concatenate([[t[0]],t[1:]-t[:-1]])
    ire = where(z>=zre)[0][-1]
    vavg_pre = sum(vmax[:ire+1]*dt[:ire+1])/t[ire] #mean([ vv for vv,zz in zip(vmax,z) if zz > zre ])

    if   binning == 'all':
        return array([ sfr_pre(vv,method=sfr_pre_method) if zz >= zre else sfr_post(vv) for vv,zz in zip(vmax,z) ] )
    elif binning == '2bins':
        return array([ sfr_pre(vavg_pre,method=sfr_pre_method) if zz >= zre else sfr_post(vmax[-1]) for vv,zz in zip(vmax,z) ])
    elif binning == '3bins':
        return array([ sfr_pre(vavg_pre,method=sfr_pre_method) if zz >= zre else (sfr_post(vmax[-1]) if vv > vthres else 0) for vv,zz in zip(vmax,z) ])
