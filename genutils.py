# genutils.py
# created 2018.08.10 by stacy kim

from numpy import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import os
hostname = os.uname()[1]
if hostname == 'CLWS00081':
    DIR = '/home/stacykim/research/pyutils/'
elif 'astro' in hostname or 'cluster' in hostname:
    DIR = '/user/HS103/sk0048/research/pyutils/'
else:
    DIR = '/Users/hgzxbprn/Documents/research/pyutils/'

from colossus.cosmology import cosmology
from colossus.halo.mass_defs import changeMassDefinition
from colossus.halo.concentration import concentration as colossus_cNFW

cosmoWMAP5 = cosmology.setCosmology('WMAP5')
cosmoP13   = cosmology.setCosmology('planck13')
cosmoP15   = cosmology.setCosmology('planck15')
cosmoP18   = cosmology.setCosmology('planck18') # default


CDM_MF = 'd17'
WDM_MF = 'schneider'

# =============================================================================================
# CONSTANTS

PC          = 3.086e18        # in cm
KPC         = 1e3*PC          # in cm
MPC         = 1e6*PC          # in cm
KM          = 1e5             # in cm
KMS         = 1e5             # cm/s
                                                                                                  
TIME        = KPC/1e5         # seconds in a unit of Gadget's time
GYR         = 3600*24*365.25*1e9 # seconds in a Gyr
TIME_IN_GYR = TIME/GYR        # conversion from Gadget time units to Gyr
MSUN        = 1.9891e33       # in g
G           = 6.67e-8         # in cgs



# =============================================================================================
# GENERAL REDSHIFT-DEPENDENT QUANTITIES

def rho_bar(z,method='d15'):  # in MSUN/KPC^3
    if   method == 'd08':      return cosmoWMAP5.rho_m(z) * (cosmoWMAP5.Hz(0)/100.)**2  # Duffy+ 2008
    elif method == 'd14':      return cosmoP13.rho_m(z)   * (cosmoP13.Hz(0)/100.)**2   # Dutton+ 2014
    elif method == 'd15-wmap': return cosmoWMAP5.rho_m(z) * (cosmoWMAP5.Hz(0)/100.)**2  # cosmology in Duffy+ 2008 but DJ19 c-M relation
    elif 'd15' in method:      return cosmoP18.rho_m(z)   * (cosmoP18.Hz(0)/100.)**2   # Diemer & Joyce 2019

def h(z,method='d15'):  # returns H(z)/100.
    if   method == 'd08':      return cosmoWMAP5.Hz(z)/100.  # Duffy+ 2008
    elif method == 'd14':      return cosmoP13.Hz(z)/100.    # Dutton+ 2014
    elif method == 'd15-wmap': return cosmoWMAP5.Hz(z)/100.  # cosmology in Duffy+ 2008 but DJ19 c-M relation
    elif 'd15' in method:      return cosmoP18.Hz(z)/100.    # Diemer & Joyce 2019

def age(z,method='d15'):  # in Gyr
    if   method == 'd08':      return cosmoWMAP5.age(z)  # Duffy+ 2008
    elif method == 'd14':      return cosmoP13.age(z)    # Dutton+ 2014
    elif method == 'd15-wmap': return cosmoWMAP5.age(z)  # cosmology in Duffy+ 2008 but DJ19 c-M relation
    elif 'd15' in method:      return cosmoP18.age(z)    # Diemer & Joyce 2019

def rhoc(z,method='d15'):
    #rhoc  = 3/(8*pi*G) * (7.0/PC)**2 # critical density in g/cm^3
    if   method == 'd08':      return cosmoWMAP5.rho_c(z) * (cosmoWMAP5.Hz(0)/100)**2 * MSUN/KPC**3 # Duffy+ 2008
    elif method == 'd14':      return cosmoP13.rho_c(z)   * (cosmoP13.Hz(0)  /100)**2 * MSUN/KPC**3 # Dutton+ 2014
    elif method == 'd15-wmap': return cosmoWMAP5.rho_c(z) * (cosmoWMAP5.Hz(0)/100)**2 * MSUN/KPC**3 # cosmology in Duffy+ 2008 but DJ19 c-M relation
    elif 'd15' in method:      return cosmoP18.rho_c(z)   * (cosmoP18.Hz(0)  /100)**2 * MSUN/KPC**3 # Diemer & Joyce 2019

def omega_m(z,method='d15'):
    if   method == 'd08'     : return cosmoWMAP5.Om(z)
    elif method == 'd14'     : return cosmoP13.Om(z)
    elif method == 'd15-wmap': return cosmoWMAP5.Om(z)
    elif 'd15' in method:      return cosmoP18.Om(z)



# =============================================================================================
# MASS PROFILE FUNCTIONS
# for all the routines that follow, masses are in units of MSUN

def nfw_r(mass,c=None,delta=200.,z=0,cNFW_method='d15'):
    if (not hasattr(c,'__iter__')) and c==None:  c = cNFW(mass,z=z,massdef=str(int(delta))+'c',method=cNFW_method)
    rvir3 = mass*MSUN / (4*pi/3*delta*rhoc(z,method=cNFW_method))
    rvir  = rvir3**(1/3.)
    rs    = rvir/c
    return array([rs, rvir])/KPC

def nfw_vmax(m,z=0,cNFW_method='d15',wdm=False,mWDM=5.):
    c       = cNFW(m,z=z,method=cNFW_method,wdm=wdm,mWDM=mWDM)  # duffy08(m) changed 11/22/19
    rs,rvir = nfw_r(m,c,z=z,cNFW_method=cNFW_method)
    v200    = sqrt(G*m*MSUN/rvir/KPC) # work in cgs
    alpha   = log(1+c) - c/(1+c)
    return v200*sqrt(0.216*c/alpha)/KMS



# =============================================================================================
# MASS-CONCENTRATION RELATIONS

def cNFW(m,z=0,virial=False,massdef=None,method='d15', wdm=False,mWDM=5., kcut=50,ncut=5.0):
    """
    Returns the NFW concentration, calculated according to the given mass concentration relation 'method'.
    Written to use the versions from COLOSSUS by Diemer+ 2017, but can use versions I coded up by uncommenting them.
    Supports WDM concentrations as given by Schneider+ 2012's relation, given mWDM in keV.
    https://bdiemer.bitbucket.io/colossus/halo_concentration.html
    """

    if massdef==None:  massdef = 'vir' if virial else '200c'
    h0 = h(0,method=method)

    if   method=='d08':
        #return duffy08 (m,z=z,virial=virial)
        cosmology.setCurrent(cosmoWMAP5)
        c = colossus_cNFW(m/h0, massdef, z, model='duffy08')
    elif method=='d14':
        #return dutton14(m,z=z,virial=virial)
        cosmology.setCurrent(cosmoP13)
        c = colossus_cNFW(m/h0, massdef, z, model='dutton14')
    elif method=='d15': # Diemer & Joyce 2019
        cosmology.setCurrent(cosmoP18)
        #cosmology.setCurrent(cosmoWMAP5)
        c = colossus_cNFW(m/h0, massdef, z, model='diemer19')

    elif method=='d15+1s': # Diemer & Joyce 2019
        cosmology.setCurrent(cosmoP18)
        c = colossus_cNFW(m/h0, massdef, z, model='diemer19') * 10**0.16
    elif method=='d15-1s': # Diemer & Joyce 2019
        cosmology.setCurrent(cosmoP18)
        c = colossus_cNFW(m/h0, massdef, z, model='diemer19') / 10**0.16
    elif method=='d15-wmap':
        cosmology.setCurrent(cosmoWMAP5)
        c = colossus_cNFW(m/h0, massdef, z, model='diemer19')

    elif method=='enhanced':

        assert (kcut==5 or kcut==8 or kcut==14 or kcut==23 or kcut==39 or kcut==50), 'kcut must be 5, 8, 14, 23, 39, or 50 for enhanced small-scale power spectrum model'
        assert (ncut==1.4 or ncut==2.0 or ncut==2.9 or ncut==4.2 or ncut==5.0), 'ncut must be 1.4, 2.0, 2.9, 4.2, or 5.0 for enhanced small-scale power spectrum model'
    
        k_string = '50' if kcut==50 else str(round(float(kcut),1))
        fn = DIR+'assets/mf/mf_enhanced/concentrations/cut_'+k_string+'_index_'+str(round(ncut,1))+'.dat'
        m200c, c200c = loadtxt(fn,unpack=True)

        if hasattr(m,'__iter__'):
            assert min(m) >= m200c[0], 'given mass '+str(min(m))+' falls below range where data for enhanced power spectrum concentrations exist, '+str(m200c[0])+' msun'
            assert max(m) <= m200c[-1], 'given mass '+str(max(m))+' falls above range where data for enhanced power spectrum concentrations exist, '+str(m200c[-1])+' msun'
        else:
            assert m >= m200c[0], 'given mass '+str(m)+' falls below range where data for enhanced power spectrum concentrations exist, '+str(m200c[0])+' msun'
            assert m <= m200c[-1], 'given mass '+str(m)+' falls above range where data for enhanced power spectrum concentrations exist, '+str(m200c[-1])+' msun'

        return interp(m, m200c,c200c)
        
    else:
        print('did not recognize given mass-concentration relation',relation,'!  Aborting...')
        exit()

    if not wdm:
        return c
    else:
        m1m_div_h, r1m_div_h, c1m = changeMassDefinition(m/h0, c, z, '200c', '1m')  # Schneider+ 2012 uses rho_bar in mass def
        m1m = m1m_div_h * h0
        return c * (1 + GAMMA1*mass_hm(mWDM,cNFW_method=method)/m1m)**(-GAMMA2)  # Schneider+ 2012


# WDM concentrations from Schneider+ 2012
# parameters to convert cCDM to cWDM via their eq. 39
GAMMA1 = 15.
GAMMA2 = 0.3


# Duffy et al. 2008 mass-concentration relation
# technically only fit to 1e11-1e15 MSUN halos...
# note that their relation was only fit to galaxies out to z = 2.
A200_DUFFY = 5.71     # coefficient
B200_DUFFY = -0.084   # mass scaling
C200_DUFFY = -0.47    # redshift scaling

Avir_DUFFY = 5.71     # coefficient
Bvir_DUFFY = -0.084   # mass scaling
Cvir_DUFFY = -0.47    # redshift scaling

MPIVOT  = 2e12/(cosmoWMAP5.Hz(0)/100.)   # 2e12/h = mormalized mass, in MSUN

def duffy08(m,z=0,virial=False):
    # m = mass in MSUN
    if virial:  return Avir_DUFFY * (m/MPIVOT)**Bvir_DUFFY * (1+z)**Cvir_DUFFY
    else:       return A200_DUFFY * (m/MPIVOT)**B200_DUFFY * (1+z)**C200_DUFFY


# Dutton+ 2014 mass-concentration relation
# technically only fit down to 1e11 MSUN halos...
# their relation was fit out to z = 5.
def dutton14(m,z=0,virial=False):
    """
    Assumes masses in units of MSUN.
    If virial==False, then assumes masses in 200*rhocrit definition.
    """
    if virial:
        a = 0.537 + (1.025-0.537) * exp(-0.718 * z**1.08)
        b = -0.097 + 0.024*z
    else: # M200crit
        a = 0.520 + (0.905-0.520) * exp(-0.617 * z**1.21)
        b = -0.101 + 0.026*z
    log10c = a + b*log10(m/(1e12/h))
    return 10**log10c



# ==============================================================================
# ABUNDANCE MATCHING

#mhM13,msM13 = loadtxt(DIR+'moster.dat',unpack=True)
#mhaloM13 = interp1d(log(msM13),log(mhM13),kind='linear',fill_value='extrapolate', bounds_error=False)
#mstarM13 = interp1d(log(mhM13),log(msM13),kind='linear',fill_value='extrapolate', bounds_error=False)

# Moster+ 2013's redshift-dependent SMHM relation (all in MSUN units)
M10_M13 = 11.590  # +- 0.236
M11_M13 =  1.195  # +- 0.353
N10_M13 =  0.0351 # +- 0.0058
N11_M13 = -0.0247 # +- 0.0069
b10_M13 =  1.376  # +- 0.153
b11_M13 = -0.826  # +- 0.225
g10_M13 =  0.608  # +- 0.608
g11_M13 =  0.329  # +- 0.173

def moster13(mhalo,z=0.):
    """
    Gives M* given Mhalo, all in MSUN units.
    """
    M1    = 10**( M10_M13 + M11_M13 * z/(z+1) )
    N     = N10_M13 + N11_M13 * z/(z+1)
    beta  = b10_M13 + b11_M13 * z/(z+1)
    gamma = g10_M13 + g11_M13 * z/(z+1)
    return 2 * N * mhalo / ( (mhalo/M1)**-beta + (mhalo/M1)**gamma )


# Behroozi+ 2013 z=0 relation
mhB13,msB13 = loadtxt(DIR+'behroozi.dat' ,unpack=True)
mhaloB13 = interp1d(log(msB13),log(mhB13),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarB13 = interp1d(log(mhB13),log(msB13),kind='linear',fill_value='extrapolate',bounds_error=False)

# Behroozi+ 2013 redshift-dependent SMHM relation (all in MSUN units)


def behroozi13(mhalo,z=0.):
    """
    Gives M* given Mhalo, all in MSUN units.
    """
    return 1



# Brook+ 2014 z=0 relation
f = lambda c,x: (log(1+c)-c/(1+c))*(350/200.)*x**3 - log(1+c*x) + 1/(1+1/c/x)  # fxn to find root for R350 (see pg. 5B of cat notebook)
c200   = arange(5,25+1.)  # NFW concentrations
x350   = array( [ brentq(lambda x: f(float(cc),x),0.1,1) for cc in c200 ] )  # R350/R200
a200   = log(1+c200) - c200/(1+c200)
a350   = log(1+c200*x350) - 1/(1+1/c200/x350)
mratio = a200/a350  # M200/M350 as a function of c200, i.e. correction factor!
m200   = (c200/A200_DUFFY)**(1/B200_DUFFY)*MPIVOT  # invert D08 relation
m350   =  m200/mratio

mh350B14,msB14 = loadtxt(DIR+'brook.dat' ,unpack=True)
mh200B14 = interp(mh350B14,m350[::-1],m200[::-1])  # mhalo = peak M350, convert to M200 assuming NFW
mhaloB14 = interp1d(log(msB14),log(mh200B14),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarB14 = interp1d(log(mh200B14),log(msB14),kind='linear',fill_value='extrapolate',bounds_error=False)


# Dooley+ 2017 tuned-bent model
mhD17,msD17 = loadtxt(DIR+'dooley.dat' ,unpack=True)
mhaloD17 = interp1d(log(msD17),log(mhD17),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarD17 = interp1d(log(mhD17),log(msD17),kind='linear',fill_value='extrapolate',bounds_error=False)


# Munshi+ 2021 model
log10mhM21,log10msM21 = loadtxt(DIR+'munshi.dat' ,unpack=True)
mhaloM21 = interp1d(log(10**log10msM21),log(10**log10mhM21),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarM21 = interp1d(log(10**log10mhM21),log(10**log10msM21),kind='linear',fill_value='extrapolate',bounds_error=False)


# ==============================================================================
# MASS FUNCTIONS

# CDM mass function
def mf_cdm(m,mhost=1e12):
    if CDM_MF == 'd17':  # m in MSUN, Dooley+ 2017a
        #return 1.88e-3 * m**-1.87 * mhost  # peak virial mass
        return 8.54e-4 * m**-1.84 * mhost
    elif CDM_MF == 'gk14':
        #mf_cdm = lambda m: 1.11 * (m/)**-1.87 * MHOST  # m in MSUN, from GK's ELVIS paper
        print('no support for GK14 ELVIS subhalo MF!  Aborting...')
        exit()
    else:
        print('no support for CDM MF',CDM_MF,'! Aborting...')
        exit()


        
# WDM mass function definitions
MU        = 1.12   # exponent in transfer function in Schneider+ 2012
if WDM_MF == 'schneider':
    BETA, GAMMA = 1.16,1  # Schneider+ 2012
elif WDM_MF == 'lovell':
    BETA, GAMMA = 0.99,2.7  # Lovell+ 2014
else:
    print('no support for WDM MF',WDM_MF,'! Aborting...')
    exit()

def mass_hm(mWDM,cNFW_method='d15'):
    """
    Returns the WDM half-mode mass in MSUN.
    """
    h0        = h      (0,method=cNFW_method)
    rho_bar0  = rho_bar(0,method=cNFW_method)
    OMEGA_WDM = omega_m(0,method=cNFW_method)
    
    alpha_hm  = 49.0 * mWDM**-1.11 * (OMEGA_WDM/0.25)**0.11 * (h0/0.7)**1.22 / h0 # kpc # incorrectly had (H/0.7)*1.22 here, 10/30/17
    lambda_hm = 2*pi* alpha_hm * (2**(MU/5.) - 1)**(-1./2/MU)
    return  4*pi/3 * rho_bar0 * (lambda_hm/2)**3



# enhanced power spectrum (w/Ivan+Annika)
def mf_enhanced(m,mhost=1e12,kcut=50,ncut=5.0):

    assert (mhost==1e12 or mhost==5e11 or mhost==2e12), 'mhost must be 5e11, 1e12, or 2e12 MSUN for enhanced small-scale power spectrum mass function'
    assert (kcut==5 or kcut==8 or kcut==14 or kcut==23 or kcut==39 or kcut==50), 'kcut must be 5, 8, 14, 23, 39, or 50 for enhanced small-scale power spectrum mass function'
    assert (ncut==1.4 or ncut==2.0 or ncut==2.9 or ncut==4.2 or ncut==5.0), 'ncut must be 1.4, 2.0, 2.9, 4.2, or 5.0 for enhanced small-scale power spectrum mass function'
    
    k_string = '50' if kcut==50 else str(round(float(kcut),1))
    fn = DIR+'assets/mf/mf_enhanced/shmf/subhalo_cut_'+k_string+'_index_'+str(round(ncut,1))+'.dat'
    q, dNdm, stat_uncert = loadtxt(fn,unpack=True)
    msub = q*mhost

    if hasattr(m,'__iter__'):
        assert min(m) >= msub[0], 'given mass '+str(min(m))+' falls below range where data for enhanced power spectrum mass function exists, '+str(msub[0])+' msun'
    else:
        assert m >= msub[0], 'given mass '+str(m)+' falls below range where data for enhanced power spectrum mass function exists, '+str(msub[0])+' msun'
    
    return interp(m, msub,dNdm, right=0)
