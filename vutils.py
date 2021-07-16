# vutils.py
# created 2018.05.14 by stacy kim from map_vf2mf.py

from sys import *
from numpy import *
from numpy.random import normal
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.size'] = 15
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root, brentq

#from vcorrect import *
from genutils import *  # NFW definitions imported from here


# includes tidal stripping using Penarrubia+ 2010's method. does not include stellar mass (mstar used to calculate Re if needed).
def menc(renc,m200,profile,mleft=1,mleft100=None,zin=0.,smhm='m13',mstar=None,reff='r17',Re0=None,nostripRe=False,cNFW_method='d15',c200=None,
         mcore_thres=None,tSF=None, wdm=False,mWDM=5., sigmaSI=None,fudge=1.3,stretch=0):
    """
    Some of the input parameters:
    renc     = radius in KPC w/in which to calculate enclosed mass
    m200     = infall mass of subhalo in MSUN (overdensity w/Delta = 200 * critical density)
    profile  = 'nfw' (or 'NFW'), 'cored', 'coreNFW', and 'sidm' (or 'SIDM')
    mleft    = fraction of original bound mass left, in M200 units (not used if mleft100 != None)
    mleft100 = fraction of original bound mass left, in M100 units
    zin      = redshift at which subhalo entered MW's virial volume
               if set to None, then computed from tin
    sigmaSI   = SIDM cross section
    tSF       = how long star formation lasted, in GYR
    wdm       = treat as WDM? only changes mass-concentration parameter
    mWDM      = mass of WDM particle, in keV; need to specify wdm=True if want wdm
    """
    
    h0 = h(0,method=cNFW_method)

    # assume everything given in M200c units, and switch to M100c, which is system P10 worked in
    if (not hasattr(c200,'__iter__')) and c200==None:
        c200 = cNFW(m200,z=zin,virial=False,method=cNFW_method,wdm=wdm,mWDM=mWDM)

    rs,r200 = nfw_r(m200,c200,z=zin,cNFW_method=cNFW_method)  # rs doesn't change with halo def

    m100_div_h, r100_div_h, c100 = changeMassDefinition(m200/h0, c200, zin, '200c', '100c')
    m100,r100 = m100_div_h * h0, r100_div_h * h0

    # switch from mleft200 to mleft100
    while (not hasattr(mleft100,'__iter__')) and mleft100 == None:

        # if no tidal stripping, then set mleft100 = 1 or ones(N)
        if hasattr(mleft,'__iter__'):
            if all(mleft==ones(len(mleft))):
                mleft100 = ones(len(mleft))
                break
        elif mleft==1:
            mleft100 = 1.
            break

        # else solve for corresponding mleft100
        f = lambda ml100,mm,rr,cc: menc(rr,mm,profile,mleft100=ml100,zin=zin,sigmaSI=sigmaSI,fudge=fudge,stretch=stretch,mcore_thres=mcore_thres,cNFW_method=cNFW_method,c200=cc,wdm=wdm,mWDM=mWDM)/mm - mleft
        if (hasattr(mleft,'__iter__') and len(mleft) > 1) or (hasattr(m200,'__iter__') and len(m200) > 1):  # solve for multiple subs at once
            x0 = mleft if (hasattr(mleft,'__iter__') and len(mleft) > 1) else mleft*ones(len(m200))
            mleft100 = array([ root(f,x0=xx0,args=(mm200,rr200,cc200)).x[0] for rr200,mm200,cc200,xx0 in zip(r200,m200,c200,x0) ])
            mleft100[mleft100 > 0.9] = 1.
        else:
            mleft100 = root(f,x0=mleft,args=(m200,r200)).x[0]
            if mleft100 > 0.9: mleft100 = 1.

    onesub = not hasattr(mleft100,'__iter__')

    tin = age(zin,method=cNFW_method)  # convert zin into time since infall, in Gyr


    # now calculate mass enclosed!
    if profile=='nfw' or profile=='NFW':

        #print 'mleft',mleft100,'m100',m100,'c100',c100

        fNFW = lambda xx: log(1+xx) - xx/(1+xx)
        fNFWstrip = lambda xx: 1./6 * xx**2 * (xx+3) / (xx+1)**3
        
        if (not onesub and all(mleft100==ones(len(mleft100)))) or (onesub and mleft100==1):  return m100*fNFW(renc/rs)/fNFW(c100)

        R_MU,R_ETA, V_MU,V_ETA = -0.3,0.4 , 0.4,0.3
        rmax     = 2.16258 * rs * KPC  # in cm
        vmax     = sqrt(G * m100*MSUN*fNFW(rmax/(rs*KPC))/fNFW(c100) / rmax) # in cm/s
        rmax_new = 2**R_MU * mleft100**R_ETA / (1+mleft100)**R_MU * rmax # in cm
        vmax_new = 2**V_MU * mleft100**V_ETA / (1+mleft100)**V_MU * vmax # in cm/s
        rs_new   = rmax_new/(sqrt(7.)-2.)  # in cm

        return rmax_new * vmax_new**2 / G * fNFWstrip(renc*KPC/rs_new)/fNFWstrip(rmax_new/rs_new) / MSUN # in MSUN


    elif profile=='coreNFW':

        ETA,KAPPA = 3.,0.04
        fCORENFW = lambda x: (exp(x)-exp(-x)) / (exp(x)+exp(-x))  # x = r/rc

        if  tSF==None: tSF = tin
        tSF *= GYR
        tDYN = 2*pi*sqrt((rs*KPC)**3/G/(menc(rs,m200,'nfw',mleft=1,zin=zin,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)*MSUN))
        q = KAPPA * tSF / tDYN
        n = fCORENFW(q)

        if (not hasattr(Re0,'__iter__')) and Re0==None:
            Re0 = Reff(m200,'coreNFW',mleft=1.,zin=zin,mcore_thres=mcore_thres,nostripRe=nostripRe,smhm=smhm,mstar=mstar,reff=reff,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # inital 2D half-light radius, in kpc
        Rc = ETA * Re0  # coreNFW core radius, in kpc

        if not hasattr(m200,'__iter__'):
            suppression = 1 if (mcore_thres != None and m200 < mcore_thres) else fCORENFW(renc/Rc)**n
        elif mcore_thres == None:
            suppression = fCORENFW(renc/Rc)**n
        else:
            suppression = array([1. if mm < mcore_thres else fCORENFW(rrenc/rrc)**nn for mm,rrenc,rrc,nn in zip(m200,renc,Rc,n)])

        return menc(renc,m200,'nfw',mleft100=mleft100,zin=zin,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM) * suppression


    elif profile=='cored':
        
        fCORE = lambda xx: log(xx+1) - 0.5*xx*(2+3*xx)/(1+xx)**2
        fCOREstrip = lambda xx: xx**3*(xx+4) / 12./(xx+1)**4
        
        if (not onesub and all(mleft100==ones(len(mleft100)))) or (onesub and mleft100==1):  return m100*fCORE(renc/rs)/fCORE(c100)
        
        R_MU,R_ETA, V_MU,V_ETA = -1.3,0.05 , 0.4,0.37
        rmax     = 4.4247 *rs * KPC  # in cm
        vmax     = sqrt(G * m100*MSUN*fCORE(rmax/(rs*KPC))/fCORE(c100) / rmax) # in cm/s
        rmax_new = 2**R_MU * mleft100**R_ETA / (1+mleft100)**R_MU * rmax  # in cm
        vmax_new = 2**V_MU * mleft100**V_ETA / (1+mleft100)**V_MU * vmax  # in cm/s
        rs_new   = rmax_new * 2/(sqrt(57)-5)  # in cm
        
        return rmax_new * vmax_new**2 / G * fCOREstrip(renc*KPC/rs_new)/fCOREstrip(rmax_new/rs_new) / MSUN # in MSUN


    elif profile=='sidm' or profile=='SIDM':

        if sigmaSI==None:
            print('given profile SIDM but not sigmaSI! Aborting...')
            exit()

        tin = age(0)  # assume it's had the entire age of universe to form core

        fNFW = lambda xx: log(1+xx) - xx/(1+xx)
        fBURK = lambda xx: (1./4.)*( log( 1+xx**2) + 2*log( 1+xx) - 2*arctan(xx))
        fNFWstrip = lambda xx: 1./6 * xx**2 * (xx+3) / (xx+1)**3
        fNFWdens = lambda xx: (xx*(1+xx)**2)**(-1)
        fBURKdens = lambda xx: ( (1+xx) * ( 1+xx**2))**(-1.)
        fNFWstripdens = lambda xx: 1./(xx*(1+xx)**4)
    
        fudgeVmax = 1.5  # Fiducial value is 2.5 from Rocha+2013.  Needed an adjustment going to dwarf scales,
                         # I think because the concentration change for dwarfs relative to clusters shifts
                         # the relationship between vrms and vmax.
        rmax     = 2.16258 * rs * KPC  # in cm
        vmax     = sqrt(G * m100*MSUN*fNFW(2.16258)/fNFW(c100) / rmax) # in cm/s
    
        # r1/rs according to the prescription in Sec. 7 of Rocha+ 2013
        if (hasattr(c100,'__iter__') and len(c100) > 1):
            r1_rs = array([ brentq( lambda x: fudgeVmax*vm**3/(G*rm**2) *fNFWdens(x) * tin*GYR * sigmaSI -1. , 1e-8, cc) for vm,rm,cc in zip(vmax,rmax,c100) ])
        else:
            r1_rs = brentq( lambda x: fudgeVmax*vmax**3/(G*rmax**2) *fNFWdens(x) * tin*GYR * sigmaSI -1. , 1e-8, c100 )
        rb = fudge * r1_rs  # rb/rs, currently in original form.

        if (not onesub and all(mleft100==ones(len(mleft100)))) or (onesub and mleft100==1):

            fSIDM_inner = lambda xx: m100*(fNFWdens(r1_rs)/fBURKdens(r1_rs/rb))*rb**3 * fBURK(xx/rb)/fNFW(c100)
            fSIDM_outer = lambda xx: m100*(fNFW(xx) - fNFW(r1_rs))/fNFW(c100) + fSIDM_inner(r1_rs)
            fSIDM = lambda xx: (xx > r1_rs)*fSIDM_outer(xx) + (xx <= r1_rs)*fSIDM_inner(xx)

            return fSIDM( renc/rs )
        
        else:

            R_MU,R_ETA, V_MU,V_ETA = -0.3,0.4 , 0.4,0.3
            rmax     = 2.16258 * rs * KPC  # in cm
            vmax     = sqrt(G * m100*MSUN*fNFW(rmax/(rs*KPC))/fNFW(c100) / rmax) # in cm/s
            rmax_new = 2**R_MU * mleft100**R_ETA / (1+mleft100)**R_MU * rmax # in cm
            vmax_new = 2**V_MU * mleft100**V_ETA / (1+mleft100)**V_MU * vmax # in cm/s
            rs_new   = rmax_new/(sqrt(7.)-2.)  # in cm
            mmax     = rmax_new * vmax_new**2 / G / MSUN # mass enclosed at rmax_new
            
            if stretch==0:  # Case if rb is a fixed physical scale, instead of with respect to rs.
                rb = rb/(rs_new/rs/KPC)
                r1_rs = r1_rs/(rs_new/rs/KPC)
            
            fSIDM_inner = lambda xx: mmax*(fNFWstripdens(r1_rs)/fBURKdens(r1_rs/rb))*rb**3 * fBURK(xx/rb)/fNFWstrip(rmax_new/rs_new)
            fSIDM_outer = lambda xx: mmax*(fNFWstrip(xx) - fNFWstrip(r1_rs))/fNFWstrip(rmax_new/rs_new) + fSIDM_inner(r1_rs)
            fSIDM = lambda xx: (xx > r1_rs)*fSIDM_outer(xx) + (xx <= r1_rs)*fSIDM_inner(xx)
            
            return fSIDM( renc / (rs_new/KPC) )


    else:

        print('menc for density profile',profile,'not implemented... aborting.')
        exit()


# ===========================================================================
# FOR THE GALAXIES

# Penarrubia+ 2008 change in reff
aSB0,bSB0 = 2.7,2.0
aRc ,bRc  = 1.5,0.65
acK ,bcK  = 1.75,0.0
def g(x,a,b): return 2**a * x**b / (1+x)**a

def Reff_in_Rcore(cK):
    f = lambda x: (log(1+x)-4*(sqrt(1+x)-1)/sqrt(1+cK**2) + x/(1+cK**2)) / (log(1+cK**2) - (3*sqrt(1+cK**2)-1)*(sqrt(1+cK**2)-1)/(1+cK**2)) - 0.5
    x0 = 1 if not (hasattr(cK,'__iter__') and len(cK) > 1) else ones(len(cK))
    return sqrt(root(f,x0=x0).x)  # half light radius in units of Rc (2D King core radius)

def delta_Reff(mleft,cK0=5.):
    Re2Rc0 = Reff_in_Rcore(cK0)
    Re2Rc  = Reff_in_Rcore(cK0*g(mleft,acK,bcK))
    delta_Rc = g(mleft,aRc,bRc)
    return Re2Rc / Re2Rc0 * delta_Rc


def Reff(m200,profile,cK=5.,mleft=1,zin=0.,sigmaSI=None,mcore_thres=None,reff='r17',Re0=None,nostripRe=False,smhm='m13',mstar=None,cNFW_method='d15',c200=None,wdm=False,mWDM=5.):
    """
    Assuming a King profile, returns the 2D half-light radius of a galaxy in KPC.
    Takes into account change in reff due to tidal stripping.
    If mstar not given, calculated from the chosen SMHM relation.
    All based on a Delta = 200c definition.
    """

    if (not hasattr(c200,'__iter__')) and c200==None:
        c200 = cNFW(m200,z=zin,virial=False,method=cNFW_method,wdm=wdm,mWDM=mWDM)

    rs,rvir = nfw_r(m200,c200,z=zin,cNFW_method=cNFW_method)

    if (not hasattr(mstar,'__iter__')) and mstar==None:
        if   smhm=='m13':
            mstar = moster13(m200,z=zin) #exp(mstarM13(log(m200)))
        if smhm=='m13+1sig':
            mstar = moster13(m200,z=zin) * 10**0.15
        if smhm=='m13-1sig':
            mstar = moster13(m200,z=zin) * 10**-0.15
        elif smhm=='m13+1sigGK17':  # 2.730e11 MSUN pivot mass from GK17's Mvir 1e11.5 to our M200
            mstar = moster13(m200,z=zin) * 10**array([0.2 if mm > 2.730e11 else (0.2-0.2*(log10(mm)-log10(2.730e11))) for mm in m200])
        elif smhm=='m13-1sigGK17':
            mstar = moster13(m200,z=zin) * 10**-array([0.2 if mm > 2.730e11 else (0.2-0.2*(log10(mm)-log10(2.730e11))) for mm in m200])
        elif smhm=='b13':
            mstar = exp(mstarB13(log(m200)))
        elif smhm=='b14':
            mstar = exp(mstarB14(log(m200)))
        elif smhm=='d17':
            mstar = exp(mstarD17(log(m200)))
        elif smhm=='d17+1sig':
            mstar = exp(mstarD17(log(m200))) * 10**0.4
        elif smhm=='d17-1sig':
            mstar = exp(mstarD17(log(m200))) * 10**-0.4
        elif smhm=='m21':
            mstar = exp(mstarM21(log(m200)))
        elif smhm=='m21+1sig':
            mstar = exp(mstarM21(log(m200))) * 10**array([0.3 if mm > 1e10 else (0.3-0.43*(log10(mm)-10)) for mm in m200])
        elif smhm=='m21-1sig':
            mstar = exp(mstarM21(log(m200))) * 10**-array([0.3 if mm > 1e10 else (0.3-0.43*(log10(mm)-10)) for mm in m200])

    if (not hasattr(Re0,'__iter__')) and Re0==None:
        if   reff=='r17': # fit to isolated dwarfs from Read+2017 and McConnachie+ 2012, taking repeats out from M12, and no Leo T
            Re0 = 10**(0.268*log10(mstar)-2.11)
        elif reff=='r17+1s':
            Re0 = 10**(0.268*log10(mstar)-2.11 + 0.234)
        elif reff=='r17-1s':
            Re0 = 10**(0.268*log10(mstar)-2.11 - 0.234)
        elif reff=='r17scatter':
            Re0 = 10**(0.268*log10(mstar)-2.11 + normal(loc=0,scale=0.234,size=len(mstar)))
        elif reff=='d18':
            Re0 = 10**(0.23*log10(mstar)-1.93)  # 2D half-light radius from shany's 2018 paper (assumes V-band mass-to-light ratio = 2.0)
        elif reff=='j18':
            re = 0.02 * (c200/10.)**-0.7 * rvir  # Jiang+ 2018's galaxy re (3D half-light radius) to DM rvir scaling relation
            Re0 = 0.75*re # 2D half-light radius; conversion factor from Wolf+ 2010


    if mleft==1 or nostripRe:  return Re0

    # for stripped halo, calculate change in reff via Penarrubia+ 2008 methods
    # (which only applies to ABG profiles --> base profiles for non-ABGs)
    base_profile = 'nfw' if (profile=='coreNFW' or profile=='sidm') else profile
    Rcore  = Re0/Reff_in_Rcore(cK)  # 2D core radius, in kpc
    mcore0 = menc(Rcore,m200,base_profile,mleft=1    ,zin=zin,Re0=Re0,sigmaSI=sigmaSI,mcore_thres=mcore_thres,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # 3D integral in mass, in MSUN
    mcore  = menc(Rcore,m200,base_profile,mleft=mleft,zin=zin,Re0=Re0,sigmaSI=sigmaSI,mcore_thres=mcore_thres,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # 3D integral in mass, in MSUN
    mleft_core = mcore / mcore0
    Re     = Re0 * delta_Reff(mleft_core)
    
    return Re



# ===========================================================================
# TRANSLATING TO VELOCITIES

def mvir2sigLOS(mvir,profile,mleft=1.,cK=5.,mstar=None,zin=1.,estimator='wolf2010',reff_method='r17',Re0=None,nostripRe=False,smhm='m13',
                cNFW_method='d15',c200=None,mcore_thres=None,tSF=None, wdm=False,mWDM=5., sigmaSI=None, verbose=False):
    """
    Returns sigLOS in (km/s)**2.
    """

    if verbose:
        print()
        print('PARAMETER VALUES')
        print('profile',profile)
        print('mleft',mleft)
        print('cK',cK)
        print('mstar',mstar)
        print('zin',zin)
        print('sigmaSI',sigmaSI)
        print('mcore_thres',mcore_thres)
        print('mass estimator',estimator)
        print('nostripRe',nostripRe)
        print('smhm',smhm)
        print('reff_method',reff_method)
        print('cNFW_method',cNFW_method)
        print('WDM?',wdm,'' if not wdm else 'mWDM',mWDM,'keV')
    

    #if not hasattr(Re0,'__iter__') and Re0==None:
    Re = Reff(mvir,profile,mleft=mleft,zin=zin,sigmaSI=sigmaSI,mcore_thres=mcore_thres,nostripRe=nostripRe,smhm=smhm,mstar=mstar,Re0=Re0,reff=reff_method,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # the 2D half-light radius

    if estimator=='wolf2010':

        re = Re/0.75  # the 3D half-light radius; conversion factor from Wolf+ 2010
        mh = menc(re,mvir,profile,mleft=mleft,zin=zin,sigmaSI=sigmaSI,Re0=Re0,mcore_thres=mcore_thres,tSF=tSF,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # mass w/in the 3D half-light radius
        if hasattr(mstar,'__iter__') or mstar != None:  mh += mstar/2.  # assume half stellar mass w/in Reff. Should adjust amount added w/tidal stripping?
        sigLOS2 = G/4*(mh*MSUN)/(Re*KPC) / KMS**2

    if estimator == 'errani2018':

        m_est = menc(1.8*Re,mvir,profile,mleft=mleft,zin=zin,sigmaSI=sigmaSI,Re0=Re0,mcore_thres=mcore_thres,tSF=tSF,cNFW_method=cNFW_method,c200=c200,wdm=wdm,mWDM=mWDM)  # mass w/in 1.8 * 2D half-light radius
        if hasattr(mstar,'__iter__') or mstar != None:  m_est += mstar/2.  # assume half stellar mass w/in Reff. Should adjust amount added w/tidal stripping?
        sigLOS2 = G/3.5*(m_est*MSUN)/(1.8*Re*KPC) / KMS**2
    
    return sqrt(sigLOS2)


# =============================================================================================
# MAIN

if __name__ == '__main__':

    from vcorrect import *

    if len(argv) < 2:
        print('usage: {0} [menc1](,mod1) .. [mencN](,modN)'.format(argv[0]))
        print("   mencX and modX can be nfw, sis, hern, or point to file [mencX]-menc.dat or [modX]-mod.dat")
        exit()

    profiles = argv[1:]


    # =======================================================================================
    # PLOT SATELLITES IN MINFALL-VMAX PLANE

    plotm = logspace(6.5,11)  # infall masses

    # ---------------------------------------------------------------------------------------
    # theoretical predictions

    plt.plot(mvir2sigLOS(plotm,'nfw',mleft=1   ),plotm,'k-' ,label='NFW')
    plt.plot(mvir2sigLOS(plotm,'nfw',mleft=0.1 ),plotm,'k--',label='NFW, mleft 1/10')
    plt.plot(mvir2sigLOS(plotm,'nfw',mleft=0.01),plotm,'k:',label='NFW, mleft 1/100')
    plt.plot(mvir2sigLOS(plotm,'cored',mleft=1   ),plotm,'r-' ,label='cored')
    plt.plot(mvir2sigLOS(plotm,'cored',mleft=0.1 ),plotm,'r--')#,label='NFW, mleft 1/10')
    plt.plot(mvir2sigLOS(plotm,'cored',mleft=0.01),plotm,'r:')#,label='NFW, mleft 1/100')


    # ---------------------------------------------------------------------------------------
    # from ELVIS sims

    CATDIR = '/Users/hgzxbprn/Documents/osu/research/msp/halo_catalogs/PairedTrees/'

    for halo_name in ['Zeus&Hera']:
        x     = loadtxt(CATDIR + halo_name + '/X.txt'    )
        y     = loadtxt(CATDIR + halo_name + '/Y.txt'    )
        z     = loadtxt(CATDIR + halo_name + '/Z.txt'    )
        vmax  = loadtxt(CATDIR + halo_name + '/Vmax.txt' )
        mvir  = loadtxt(CATDIR + halo_name + '/Mvir.txt' )

        ipeak = argmax(mvir,axis=1)
        mpeak = array([mvir[i,ip]      for i,ip in enumerate(ipeak)])

        imw,im31 = (0,1) if mvir[0,0] < mvir[1,0] else (1,0)
        print('Found 2 halos with mass > 1e11 MSUN, paired sim ( mass of first 3 halos: MW',mvir[imw,0],'M31',mvir[im31,0],'3rd',mvir[2,0],').')
        m31_pos = array([x[im31,0],y[im31,0],z[im31,0]])
        isubs0 = 2

        nsubhalos = len(x[isubs0:])
        print('Found a total of',nsubhalos,'subhalos,',end='')

        # center present day positions on the Milky Way (2nd most massive halo)
        x0 = (x[:,0] - x[imw,0])*1000 # convert to kpc
        y0 = (y[:,0] - y[imw,0])*1000
        z0 = (z[:,0] - z[imw,0])*1000
        gcdist = sqrt( x0**2 + y0**2 + z0**2 ) # dist. from most massive halo
        inMW = gcdist < RVIR
        print(sum(gcdist < RVIR),'of which are in the MW virial volume')
        print(len(vmax[inMW][::10]),'points should be plotted')
        plt.plot(vmax[inMW][::10]/sqrt(3),mpeak[inMW][::10],'bo',alpha=0.1,markersize=3,markeredgewidth=0,label='ELVIS')


    # ---------------------------------------------------------------------------------------
    # from "abundance matching" lum MF to completeness-corrected VF

    mf_cdm = lambda m: 1.88e-3 * m**-1.87 * mMW  # m in MSUN, from Dooley+ 2017a (infall mass)

    # set up fraction luminous
    fnflum = '../ntot/d17-flum93.dat'
    print('\nreading flum data from',fnflum)
    minfall,flum_dat = loadtxt(fnflum,unpack=True)
    flum = lambda m: interp(log(m),log(minfall),flum_dat,left=0,right=1)

    # integrate to calculate luminous MF
    ngal  = [ quad(lambda m: mf_cdm(m)*flum(m),mm,mMW)[0] for mm in plotm ]  # integrate number of galaxies
    ngaltot = max(ngal)

    # calculate VF
    sigmas, rdist_names, nboot = vcorrect(profiles,bootstrap=True)
    plotv = logspace(0,2)
    obssigmas = array([ dwarf['sigma'] for dwarf_name,dwarf in dwarfs.items() if dwarf['sigma'] != None ])
    vfxn   = array([ array([median    ([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)]   ) for ip in range(len(profiles)) ]) for sigma in plotv ])
    vfxn10 = array([ array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],10) for ip in range(len(profiles)) ]) for sigma in plotv ])
    vfxn90 = array([ array([percentile([ sum(sigmas[ip][ib]>=sigma) for ib in range(nboot)],90) for ip in range(len(profiles)) ]) for sigma in plotv ])

    # do the Min-Vmax abundance matching!
    for ip in range(len(profiles)):
        nsubs = len(sigmas[ip][0]) if nboot > 1 else len(sigmas[ip])
        mf_sampled = interp( logspace(0,log10(min(nsubs,ngaltot))), ngal[::-1], plotm[::-1], left=None, right=None )
        vf_sampled = interp( logspace(0,log10(min(nsubs,ngaltot))), vfxn[::-1,ip], plotv[::-1], left=None, right=None )

        #plt.plot(vf_sampled*sqrt(3),mf_sampled,label=rdist_names[ip]+' (cc)')
        plt.plot(vf_sampled,mf_sampled,label=rdist_names[ip]+' (cc)')


    # -------------------------------------------------------------------------------------------
    # we're done! just touch up plot

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(array(plt.xlim())/2)
    plt.ylim(array(plt.ylim())*array([10,0.1]))
    plt.xlabel('vmax (km/s)')
    plt.ylabel('mass (MSUN)')
    plt.legend(loc=2,fontsize=13)
    figfn = 'mass_vmax.pdf'
    plt.savefig(figfn)
    print('wrote',figfn)
