# am.py
# created by stacy kim on 2017.09.14

from sys import *
from numpy import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
#import matplotlib.pyplot as plt


CHECK_EXTRAP = False

# duffy et al. 2008 mass-concentration relation
h       = 0.7      # normalized hubble's constant
A_DUFFY = 3.93     # coefficient
B_DUFFY = -0.097   # mass scaling
MPIVOT  = 1e14/h   # mormalized mass, in MSUN
def duffy08(m):  return A_DUFFY*(m/MPIVOT)**B_DUFFY


if len(argv) != 2:
    print('usage: {0} [mstar]'.format(argv[0]))
    exit()

mstar = float(argv[1])


# moster+ 2013
mhaloM13,mstarM13 = loadtxt('moster.dat',unpack=True)
m13 = interp1d(log(mstarM13),log(mhaloM13),kind='linear',fill_value='extrapolate', bounds_error=False)


# brook+ 2014, with conversion from M350 --> M200
f = lambda c,x: (log(1+c)-c/(1+c))*(350/200.)*x**3 - log(1+c*x) + 1/(1+1/c/x)  # fxn to find root for R350 (see pg. 5B of cat notebook)
c200   = arange(5,25+1.)  # NFW concentrations
x350   = array( [ brentq(lambda x: f(float(cc),x),0.1,1) for cc in c200 ] )  # R350/R200
a200   = log(1+c200) - c200/(1+c200)
a350   = log(1+c200*x350) - 1/(1+1/c200/x350)
mratio = a200/a350  # M200/M350 as a function of c200, i.e. correction factor!
m200   = (c200/A_DUFFY)**(1/B_DUFFY)*MPIVOT  # invert D08 relation
m350   =  m200/mratio

mhaloB14,mstarB14 = loadtxt('brook.dat' ,unpack=True)
mhaloB14 = interp(mhaloB14,m350[::-1],m200[::-1])  # mhalo = peak M350, convert to M200 assuming NFW
b14 = interp1d(log(mstarB14),log(mhaloB14),kind='linear',fill_value='extrapolate',bounds_error=False)


# behroozi+ 2013
mhaloB13,mstarB13 = loadtxt('behroozi.dat' ,unpack=True)
b13 = interp1d(log(mstarB13),log(mhaloB13),kind='linear',fill_value='extrapolate',bounds_error=False)

mstar_segI = 340*2. # V-band luminosity x (M/L = 2)

#print(exp(m13(log(mstar_segI))),exp(b14(log(mstar_segI))))
print('halo mass according to...')
print('moster+ 2013  :  '+('{0:.2e}'.format( exp(m13(log(mstar))) ))+' MSUN')
print('brook+  2014  :  '+('{0:.2e}'.format( exp(b14(log(mstar))) ))+' MSUN')
print('behroozi+ 2013:  '+('{0:.2e}'.format( exp(b13(log(mstar))) ))+' MSUN')



if CHECK_EXTRAP:
    plt.plot(mhaloM13,mstarM13,ls='-')
    plt.plot(mhaloB14,mstarB14,ls='-')
    plt.plot(mhaloB13,mstarB13,ls='-')

    mstar = logspace(2,10)
    plt.plot(exp(m13(log(mstar))),mstar,'.')
    plt.plot(exp(b14(log(mstar))),mstar,'.')
    plt.plot(exp(b13(log(mstar))),mstar,'.')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['moster+13','brook+14','behroozi+13','m13 interp','b14 interp','b13 interp'])
    plt.ylabel('Mstar/MSUN')
    plt.xlabel('Mhalo/MSUN')
    figfn = 'am.pdf'
    plt.savefig(figfn)
    print('wrote',figfn)
