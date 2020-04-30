# sem.py
# created 2020.03.20 by stacy kim

from numpy import *
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt

from tangos.examples.mergers import *

from colossus.cosmology import cosmology
from colossus.utils.constants import G
cosmo = cosmology.setCosmology('planck18')



##################################################
# SFH ROUTINES

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



##################################################
# ACCRETED STARS

def accreted_stars(halo, vthres=26., zre=4., binning='3bins', plot_mergers=False):
    """
    Returns redshift, major/minor mass ratio, halo objects, and stellar mass accreted 
    for each of the given halo's mergers.  Does not compute the stellar contribution
    of mergers of mergers.
    """

    t,z,rbins,menc = halo.calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
    ire = where(z>=zre)[0][0]
    
    if plot_mergers:
        vmax = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(mencDM_sub,rbins_sub) ])
        fig, ax = plt.subplots()
        plt.plot(t,vmax,color='k')        
  
    zmerge, qmerge, hmerge = get_mergers_of_major_progenitor(halo)
    msmerge = zeros(len(zmerge))
  
    # record main branch components
    halos = {}
    depth = -1
    h = halo
    while h != None:
        depth += 1
        halos[ h.path ] = [ '0.'+str(depth) ]
        h = h.previous
    
    for ii,im in enumerate(range(len(zmerge))):
        t_sub,z_sub,rbins_sub,mencDM_sub = hmerge[im][1].calculate_for_progenitors('t()','z()','rbins_profile','dm_mass_profile')
        vmax_sub = array([ max(sqrt(G*mm/rr)) for mm,rr in zip(mencDM_sub,rbins_sub) ])
        
        if len(t_sub)==0: continue  # skip if no mass profile data

        # catch when merger tree loops back on itself --> double-counting
        h = hmerge[im][1]
        depth = -1
        isRepeat = False
        while h != None:
            depth += 1
            if h.path not in halos.keys():
                halos[h.path] = [ str(im)+'.'+str(depth) ]
            else:
                print('--> Found repeat!!',h.path,'while tracing merger',str(int(im))+'.'+str(depth),'(also in merger(s)',halos[h.path],')')
                halos[h.path] += [ str(im)+'.'+str(depth) ]
                isRepeat = True
                break
            h = h.previous
        if isRepeat: continue  # found a repeat! skip this halo.
                
        # went through all fail conditions, now calculate vmax trajectory, SFH --> M*
        if z_sub[0] < zre:
            zz_sub = concatenate([z_sub[:ire],[zre],z_sub[ire:]])[::-1]
            tt_sub = concatenate([t_sub[:ire],[cosmo.age(zre)],t_sub[ire:]])[::-1]
            vv_sub = concatenate([vmax_sub[:ire],[interp(zre,z_sub,vmax_sub)],vmax_sub[ire:]])[::-1] # interp in z, which approx vmax evol better
        else:
            zz_sub,tt_sub,vv_sub = z_sub[::-1],t_sub[::-1],vmax_sub[::-1]
        dt_sub = concatenate([[tt_sub[0]],tt_sub[1:]-tt_sub[:-1]])
        
        sfh_binned_sub = sfh(tt_sub,zz_sub,vv_sub,vthres=vthres,zre=zre,binning=binning)
        mstar_binned_sub = array( [ sum(sfh_binned_sub[:i+1] * 1e9*dt_sub[:i+1]) for i in range(len(sfh_binned_sub)) ] )
        msmerge[im] = mstar_binned_sub[-1]
        #mstar_main = interp(zmerge[im],zz[::-1],mstar_binned[::-1])
        #print('merger',im,'at z = {0:4.2f}'.format(zmerge[im]),'with {0:5.1e}'.format(mstar_binned_merge[im]),'msun stars vs {0:5.1e}'.format(mstar_main),'msun MAIN =',int(100.*mstar_binned_merge[im]/mstar_main),'%')

        if plot_mergers:
            plt.plot(t_sub,vmax_sub,color='0.7',linewidth=1,alpha=0.5)
            plt.plot( interp(zmerge[im],z,t), interp(zmerge[im],z,vmax) ,marker='.',color='0.7',linewidth=0)

    if plot_mergers:
        plt.xlabel('t (Gyr)')
        plt.ylabel(r'v$_{\rm max}$ (km/s)')
        figfn = 'mergers.pdf'
        plt.savefig(figfn)
        print('wrote',figfn)
        plt.clf()

    return zmerge, qmerge, hmerge, msmerge
