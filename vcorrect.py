# semianalytics.py
# created 2017.05.25 by stacy kim

from sys import *
from numpy import *
from numpy.random import normal
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from genutils import *

RDIST_DATDIR = '/Users/hgzxbprn/Documents/research/projects/msp/semianalytics/'


# for instances where radial and angular are separable
#C_OMEGA = 5.
mMW    = 1e12  ## HERE 1.5 vs 1

# MW dwarf data, from Jethwa+2017's compilation, who got mags and dist from
# McConnachie 2012, and M* from Woo+ 2008 or assuming M*/L = 2 MSUN/LSUN
#         name                                        v-band Mabs  MSUN            kpc           velocity disp, error (km/s)          reff (pc)      ueff (M/arcsec^2)
dwarfs = {'LMC'              : {'type': 'classical', 'mv': -18.1, 'mstar': 1.1e9, 'dsun': 51.0, 'sigma': 20.2, 'sigerr':        0.5, 'reff':  None, 'ueff': None},
          'SMC'              : {'type': 'classical', 'mv': -16.8, 'mstar': 3.7e8, 'dsun': 64.0, 'sigma': 27.6, 'sigerr':        0.5, 'reff':  None, 'ueff': None},
          'Sagittarius'      : {'type': 'classical', 'mv': -13.5, 'mstar': 3.4e7, 'dsun': 26.0, 'sigma': 11.4, 'sigerr':        0.7, 'reff': 2587., 'ueff': 26.0},
          'Fornax'           : {'type': 'classical', 'mv': -13.4, 'mstar': 3.3e7, 'dsun': 147., 'sigma': 11.7, 'sigerr':        0.9, 'reff':  710., 'ueff': 24.0},
          'Leo I'            : {'type': 'classical', 'mv': -12.0, 'mstar': 8.8e6, 'dsun': 254., 'sigma':  9.2, 'sigerr':        1.4, 'reff':  251., 'ueff': 23.3},
          'Sculptor'         : {'type': 'classical', 'mv': -11.1, 'mstar': 3.7e6, 'dsun': 86.0, 'sigma':  9.2, 'sigerr':        1.4, 'reff':  283., 'ueff': 24.3},
          'Leo II'           : {'type': 'classical', 'mv':  -9.8, 'mstar': 1.2e6, 'dsun': 233., 'sigma':  6.6, 'sigerr':        0.7, 'reff':  176., 'ueff': 24.8},
          'Sextans I'        : {'type': 'classical', 'mv':  -9.3, 'mstar': 7.0e5, 'dsun': 86.0, 'sigma':  7.9, 'sigerr':        1.3, 'reff':  695., 'ueff': 28.0},
          'Carina'           : {'type': 'classical', 'mv':  -9.1, 'mstar': 6.0e5, 'dsun': 105., 'sigma':  6.6, 'sigerr':        1.2, 'reff':  250., 'ueff': 26.0},
          'Draco'            : {'type': 'classical', 'mv':  -8.8, 'mstar': 4.5e5, 'dsun': 76.0, 'sigma':  9.1, 'sigerr':        1.2, 'reff':  221., 'ueff': 26.1},
          'Ursa Minor'       : {'type': 'classical', 'mv':  -8.8, 'mstar': 4.5e5, 'dsun': 76.0, 'sigma':  9.5, 'sigerr':        1.2, 'reff':  181., 'ueff': 25.2},

          'Canes Venatici I' : {'type': 'ufd'      , 'mv':  -8.6, 'mstar': 1.2e5, 'dsun': 218., 'sigma':  7.6, 'sigerr':        0.4, 'reff':  564., 'ueff': 28.2},
          'Hercules'         : {'type': 'ufd tidal', 'mv':  -6.6, 'mstar': 1.9e4, 'dsun': 132., 'sigma':  3.7, 'sigerr':        0.9, 'reff':  330., 'ueff': 28.3},
          'Bootes I'         : {'type': 'ufd'      , 'mv':  -6.3, 'mstar': 1.4e4, 'dsun': 66.0, 'sigma':  2.4, 'sigerr':  [0.5,0.9], 'reff':  242., 'ueff': 28.7}, # technically +0.9 -0.5
          'Leo IV'           : {'type': 'ufd'      , 'mv':  -5.8, 'mstar': 9.0e3, 'dsun': 154., 'sigma':  3.3, 'sigerr':        1.7, 'reff':  206., 'ueff': 28.6},
          'Ursa Major I'     : {'type': 'ufd'      , 'mv':  -5.5, 'mstar': 6.8e3, 'dsun': 97.0, 'sigma':  7.6, 'sigerr':        1.0, 'reff':  319., 'ueff': 28.8},
          'Leo V'            : {'type': 'ufd tidal', 'mv':  -5.2, 'mstar': 5.1e3, 'dsun': 178., 'sigma':  3.7, 'sigerr':  [1.4,2.3], 'reff':  135., 'ueff': 28.2}, # techinically +2.3 -1.4
          'Pisces II'        : {'type': 'ufd'      , 'mv':  -5.0, 'mstar': 4.3e3, 'dsun': 182., 'sigma': None, 'sigerr':       None, 'reff':   58., 'ueff': 26.8},
          'Canes Venatici II': {'type': 'ufd'      , 'mv':  -4.9, 'mstar': 3.9e3, 'dsun': 160., 'sigma':  4.6, 'sigerr':        1.0, 'reff':   74., 'ueff': 27.2},
          'Ursa Major II'    : {'type': 'ufd'      , 'mv':  -4.2, 'mstar': 2.1e3, 'dsun': 32.0, 'sigma':  6.7, 'sigerr':        1.4, 'reff':  149., 'ueff': 29.1},
          'Coma Berenices'   : {'type': 'ufd'      , 'mv':  -4.1, 'mstar': 1.9e3, 'dsun': 44.0, 'sigma':  4.6, 'sigerr':        0.8, 'reff':   77., 'ueff': 28.4},
          #'Bootes II'        : {'type': 'ufd'      , 'mv':  -2.7, 'mstar': 5.1e2, 'dsun': 42.0, 'sigma': 10.5, 'sigerr':        7.4, 'reff':   51., 'ueff': 29.1}, # AM
          'Bootes II'        : {'type': 'ufd'      , 'mv':  -2.7, 'mstar': 5.1e2, 'dsun': 42.0, 'sigma':  4.4, 'sigerr':        1.0, 'reff':   51., 'ueff': 29.1}, # MG
          'Willman 1'        : {'type': 'ufd tidal', 'mv':  -2.7, 'mstar': 5.1e2, 'dsun': 38.0, 'sigma':  4.3, 'sigerr':  [1.3,2.3], 'reff':   25., 'ueff': 27.2},
          'Segue II'         : {'type': 'ufd tidal', 'mv':  -2.5, 'mstar': 4.3e2, 'dsun': 35.0, 'sigma':  3.4, 'sigerr':  [1.2,2.5], 'reff':   35., 'ueff': 28.6},
          'Segue I'          : {'type': 'ufd'      , 'mv':  -1.5, 'mstar': 1.7e2, 'dsun': 23.0, 'sigma':  3.9, 'sigerr':        0.8, 'reff':   29., 'ueff': 28.7}} # AM
          #'Segue I'          : {'type': 'ufd'      , 'mv':  -1.5, 'mstar': 1.7e2, 'dsun': 23.0, 'sigma':  3.7, 'sigerr':  [1.1,1.4], 'reff':   29., 'ueff': 28.7}} # MANOJ
#          'Tucana III'       : {'type': 'tidal'    , 'mv': },}}


# M31 dwarfs (from McConnachie+ 2012)
#             name                           v-band Mabs  MSUN            kpc            velocity disp, error (km/s)         reff (pc)      ueff (M/arcsec^2)
dwarfsM31 = {'M32'        : {'type': 'm31', 'mv': -16.4, 'mstar': 3.2e8, 'dm31':  23.0, 'sigma': 92.0, 'sigerr':       5.0, 'reff':  110., 'ueff': 17.0},
             'And IX'     : {'type': 'm31', 'mv':  -8.1, 'mstar': 1.5e5, 'dm31':  40.0, 'sigma':  4.5, 'sigerr':       3.6, 'reff':  557., 'ueff': 29.2},
             'NGC 205'    : {'type': 'm31', 'mv': -16.5, 'mstar': 3.3e8, 'dm31':  42.0, 'sigma': 35.0, 'sigerr':       5.0, 'reff':  590., 'ueff': 20.3},
             'And XVII'   : {'type': 'm31', 'mv':  -8.7, 'mstar': 2.6e5, 'dm31':  45.0, 'sigma': None, 'sigerr':      None, 'reff':  286., 'ueff': 26.8},
             'And I'      : {'type': 'm31', 'mv': -11.7, 'mstar': 3.9e6, 'dm31':  58.0, 'sigma': 10.6, 'sigerr':       1.1, 'reff':  672., 'ueff': 25.8},
             'And XXVII'  : {'type': 'm31', 'mv':  -7.9, 'mstar': 1.2e5, 'dm31':  75.0, 'sigma': None, 'sigerr':      None, 'reff':  434., 'ueff': 28.3},
             'And III'    : {'type': 'm31', 'mv': -10.0, 'mstar': 8.3e5, 'dm31':  75.0, 'sigma':  4.7, 'sigerr':       1.8, 'reff':  479., 'ueff': 26.2},
             'And XXV'    : {'type': 'm31', 'mv':  -9.7, 'mstar': 6.8e5, 'dm31':  88.0, 'sigma': None, 'sigerr':      None, 'reff':  709., 'ueff': 27.8},
             'And XXVI'   : {'type': 'm31', 'mv':  -7.1, 'mstar': 6.0e4, 'dm31': 103.0, 'sigma': None, 'sigerr':      None, 'reff':  222., 'ueff': 27.9},
             'And XI'     : {'type': 'm31', 'mv':  -6.9, 'mstar': 4.9e4, 'dm31': 104.0, 'sigma':  4.6, 'sigerr':      None, 'reff':  157., 'ueff': 27.6}, # sigma upper limit
             'And V'      : {'type': 'm31', 'mv':  -9.1, 'mstar': 3.9e5, 'dm31': 110.0, 'sigma': 11.5, 'sigerr': [4.4,5.3], 'reff':  315., 'ueff': 26.7},
             'And X'      : {'type': 'm31', 'mv':  -7.6, 'mstar': 9.6e4, 'dm31': 110.0, 'sigma':  3.9, 'sigerr':       1.2, 'reff':  265., 'ueff': 27.4},
             'And XXIII'  : {'type': 'm31', 'mv': -10.2, 'mstar': 1.1e6, 'dm31': 127.0, 'sigma': None, 'sigerr':      None, 'reff': 1029., 'ueff': 27.8},
             'And XX'     : {'type': 'm31', 'mv':  -6.3, 'mstar': 2.9e4, 'dm31': 129.0, 'sigma': None, 'sigerr':      None, 'reff':  124., 'ueff': 27.3},
             'And XII'    : {'type': 'm31', 'mv':  -6.4, 'mstar': 3.1e4, 'dm31': 133.0, 'sigma':  2.6, 'sigerr': [2.6,5.1], 'reff':  304., 'ueff': 29.6},
             'NGC 147'    : {'type': 'm31', 'mv': -14.6, 'mstar': 6.2e7, 'dm31': 142.0, 'sigma': 16.0, 'sigerr':       1.0, 'reff':  623., 'ueff': 22.3},
             'And XXI'    : {'type': 'm31', 'mv':  -9.9, 'mstar': 7.6e5, 'dm31': 150.0, 'sigma': None, 'sigerr':      None, 'reff':  875., 'ueff': 28.2},
             'And XIV'    : {'type': 'm31', 'mv':  -8.4, 'mstar': 2.0e5, 'dm31': 162.0, 'sigma':  5.4, 'sigerr':       1.3, 'reff':  363., 'ueff': 27.5},
             'And XV'     : {'type': 'm31', 'mv':  -9.4, 'mstar': 4.9e5, 'dm31': 174.0, 'sigma': 11.0, 'sigerr':   [5.,7.], 'reff':  222., 'ueff': 25.9},
             'And XIII'   : {'type': 'm31', 'mv':  -6.7, 'mstar': 4.1e4, 'dm31': 180.0, 'sigma':  9.7, 'sigerr': [4.5,8.9], 'reff':  207., 'ueff': 28.4},
             'And II'     : {'type': 'm31', 'mv': -12.4, 'mstar': 7.6e6, 'dm31': 184.0, 'sigma':  7.3, 'sigerr':       0.8, 'reff': 1176., 'ueff': 26.3},
             'NGC 185'    : {'type': 'm31', 'mv': -14.8, 'mstar': 6.8e7, 'dm31': 187.0, 'sigma': 24.0, 'sigerr':       1.0, 'reff':  458., 'ueff': 21.9},
             'And XXIX'   : {'type': 'm31', 'mv':  -8.3, 'mstar': 1.8e5, 'dm31': 188.0, 'sigma': None, 'sigerr':      None, 'reff':  361., 'ueff': 27.6},
             'And XIX'    : {'type': 'm31', 'mv':  -9.2, 'mstar': 4.3e5, 'dm31': 189.0, 'sigma': None, 'sigerr':      None, 'reff': 1683., 'ueff': 30.2},
             'Triangulum' : {'type': 'm31', 'mv': -18.8, 'mstar': 2.9e9, 'dm31': 206.0, 'sigma': None, 'sigerr':      None, 'reff':  None, 'ueff': None},
             'And XXIV'   : {'type': 'm31', 'mv':  -7.6, 'mstar': 9.3e4, 'dm31': 208.0, 'sigma': None, 'sigerr':      None, 'reff':  367., 'ueff': 28.5},
             'And VII'    : {'type': 'm31', 'mv': -12.6, 'mstar': 9.5e6, 'dm31': 218.0, 'sigma':  9.7, 'sigerr':       1.6, 'reff':  776., 'ueff': 25.3},
             'And XXII'   : {'type': 'm31', 'mv':  -6.5, 'mstar': 3.4e4, 'dm31': 221.0, 'sigma': None, 'sigerr':      None, 'reff':  217., 'ueff': 27.9},
             'IC 10'      : {'type': 'm31', 'mv': -15.0, 'mstar': 8.6e7, 'dm31': 252.0, 'sigma': None, 'sigerr':      None, 'reff':  612., 'ueff': 22.3},
             'LGS 3'      : {'type': 'm31', 'mv': -10.1, 'mstar': 9.6e5, 'dm31': 269.0, 'sigma':  7.9, 'sigerr': [2.9,5.3], 'reff':  470., 'ueff': 26.6},
             'And VI'     : {'type': 'm31', 'mv': -11.3, 'mstar': 2.8e6, 'dm31': 269.0, 'sigma':  9.4, 'sigerr': [2.4,3.2], 'reff':  524., 'ueff': 25.3},
             'And XVI'    : {'type': 'm31', 'mv':  -9.2, 'mstar': 4.1e5, 'dm31': 279.0, 'sigma': 10.0, 'sigerr':      None, 'reff':  136., 'ueff': 25.0}} # sigma upper limit

# 'Isolated' Local Group dwarfs (from McConnachie+ 2012)
#             name                                  v-band Mabs  MSUN             kpc            velocity disp, error (km/s)          reff (pc)
isolatedLG = {'And XXVIII'   : {'type': 'LG'     , 'mv':  -8.5, 'mstar':  2.1e5, 'dsun':  661., 'sigma': None, 'sigerr':       None, 'reff':  213.},
              'IC 1613'      : {'type': 'LG'     , 'mv': -15.2, 'mstar':  1.0e8, 'dsun':  755., 'sigma': None, 'sigerr':       None, 'reff': 1496.},
              'Phoenix'      : {'type': 'LG'     , 'mv':  -9.9, 'mstar':  7.7e5, 'dsun':  415., 'sigma': None, 'sigerr':       None, 'reff':  454.},
              'NGC 6822'     : {'type': 'LG'     , 'mv': -15.2, 'mstar':  1.0e8, 'dsun':  459., 'sigma': None, 'sigerr':       None, 'reff':  354.},
              'Cetus'        : {'type': 'LG'     , 'mv': -11.2, 'mstar':  2.6e6, 'dsun':  755., 'sigma': 17.0, 'sigerr':        2.0, 'reff':  703.},
              'Pegasus dIrr' : {'type': 'LG'     , 'mv': -12.2, 'mstar': 6.61e6, 'dsun':  920., 'sigma': None, 'sigerr':       None, 'reff':  562.},
              'Leo T'        : {'type': 'LG'     , 'mv':  -8.0, 'mstar':  1.4e5, 'dsun':  417., 'sigma':  7.5, 'sigerr':        1.6, 'reff':  120.},
              'WLM'          : {'type': 'LG'     , 'mv': -14.2, 'mstar':  4.3e7, 'dsun':  933., 'sigma': 17.5, 'sigerr':        2.0, 'reff': 2111.},
              'Leo A'        : {'type': 'LG'     , 'mv': -12.1, 'mstar':  6.0e6, 'dsun':  798., 'sigma':  9.3, 'sigerr':        1.3, 'reff':  499.},
              'And XVIII'    : {'type': 'LG'     , 'mv':  -9.7, 'mstar':  6.3e5, 'dsun': 1355., 'sigma': None, 'sigerr':       None, 'reff':  363.},
              'Aquarius'     : {'type': 'LG'     , 'mv': -10.6, 'mstar':  1.6e6, 'dsun': 1072., 'sigma': None, 'sigerr':       None, 'reff':  458.},
              'Tucana'       : {'type': 'LG'     , 'mv':  -9.5, 'mstar':  5.6e5, 'dsun':  887., 'sigma': 15.8, 'sigerr':  [3.1,4.1], 'reff':  284.},
              'Sag dIrr'     : {'type': 'LG'     , 'mv': -11.5, 'mstar':  3.5e6, 'dsun': 1067., 'sigma': None, 'sigerr':       None, 'reff':  282.},
              'UGC 4879'     : {'type': 'LG'     , 'mv': -12.5, 'mstar':  8.3e6, 'dsun': 1361., 'sigma': None, 'sigerr':       None, 'reff':  162.},
              'NGC 3109'     : {'type': 'LG'     , 'mv': -14.9, 'mstar':  7.6e7, 'dsun': 1300., 'sigma': None, 'sigerr':       None, 'reff': 1626.},
              'Sextans B'    : {'type': 'LG'     , 'mv': -14.5, 'mstar':  5.2e7, 'dsun': 1426., 'sigma': None, 'sigerr':       None, 'reff':  440.},
              'Antila'       : {'type': 'LG'     , 'mv': -10.4, 'mstar':  1.3e6, 'dsun': 1349., 'sigma': None, 'sigerr':       None, 'reff':  471.},
              'Sextans A'    : {'type': 'LG'     , 'mv': -14.3, 'mstar':  4.4e7, 'dsun': 1432., 'sigma': None, 'sigerr':       None, 'reff': 1029.},
              'KKR 25'       : {'type': 'LG'     , 'mv': -10.5, 'mstar':  1.4e6, 'dsun': 1905., 'sigma': None, 'sigerr':       None, 'reff':  222.},
              'ESO 410-G005' : {'type': 'LG'     , 'mv': -11.5, 'mstar':  3.5e6, 'dsun': 1923., 'sigma': None, 'sigerr':       None, 'reff':  280.},
              'NGC 55'       : {'type': 'LG'     , 'mv': -18.5, 'mstar':  2.2e9, 'dsun': 1932., 'sigma': None, 'sigerr':       None, 'reff': 2900.},
              'ESO 294-G010' : {'type': 'LG'     , 'mv': -11.2, 'mstar':  2.7e6, 'dsun': 2032., 'sigma': None, 'sigerr':       None, 'reff':  248.},
              'NGC 300'      : {'type': 'LG'     , 'mv': -18.5, 'mstar':  2.1e9, 'dsun': 2080., 'sigma': None, 'sigerr':       None, 'reff': 3025.},
              'IC 5152'      : {'type': 'LG'     , 'mv': -15.6, 'mstar':  2.7e8, 'dsun': 1950., 'sigma': None, 'sigerr':       None, 'reff':  550.},
              'KKH 98'       : {'type': 'LG'     , 'mv': -11.8, 'mstar':  4.5e6, 'dsun': 2523., 'sigma': None, 'sigerr':       None, 'reff':  470.},
              'UKS 2323-326' : {'type': 'LG'     , 'mv': -13.2, 'mstar':  1.7e7, 'dsun': 2208., 'sigma': None, 'sigerr':       None, 'reff':  578.},
              'KKR 3'        : {'type': 'LG'     , 'mv':  -9.5, 'mstar':  5.4e5, 'dsun': 2188., 'sigma': None, 'sigerr':       None, 'reff':  229.},
              'GR 8'         : {'type': 'LG'     , 'mv': -12.2, 'mstar':  6.4e6, 'dsun': 2178., 'sigma': None, 'sigerr':       None, 'reff':  203.},
              'UGC 9128'     : {'type': 'LG'     , 'mv': -12.4, 'mstar':  7.8e6, 'dsun': 2291., 'sigma': None, 'sigerr':       None, 'reff':  427.},
              'UGC 8508'     : {'type': 'LG'     , 'mv': -13.4, 'mstar':  1.9e7, 'dsun': 2582., 'sigma': None, 'sigerr':       None, 'reff':  315.},
              'IC 3104'      : {'type': 'LG'     , 'mv': -14.0, 'mstar':  6.2e7, 'dsun': 2270., 'sigma': None, 'sigerr':       None, 'reff': 1327.},
              'DDO 125'      : {'type': 'LG'     , 'mv': -14.4, 'mstar':  4.7e7, 'dsun': 2582., 'sigma': None, 'sigerr':       None, 'reff':  781.},
              'UGCA 86'      : {'type': 'LG'     , 'mv': -13.2, 'mstar':  1.6e7, 'dsun': 2965., 'sigma': None, 'sigerr':       None, 'reff':  811.},
              'DDO 99'       : {'type': 'LG'     , 'mv': -13.2, 'mstar':  1.6e7, 'dsun': 2594., 'sigma': None, 'sigerr':       None, 'reff':  679.},
              'IC 4662'      : {'type': 'LG'     , 'mv': -15.8, 'mstar':  1.9e8, 'dsun': 2443., 'sigma': None, 'sigerr':       None, 'reff':  341.},
              'DDO 190'      : {'type': 'LG'     , 'mv': -14.4, 'mstar':  5.1e7, 'dsun': 2783., 'sigma': None, 'sigerr':       None, 'reff':  520.},
              'KKH 86'       : {'type': 'LG'     , 'mv': -10.0, 'mstar':  8.2e5, 'dsun': 2582., 'sigma': None, 'sigerr':       None, 'reff':  210.},
              'NGC 4163'     : {'type': 'LG'     , 'mv': -14.1, 'mstar':  3.7e7, 'dsun': 2858., 'sigma': None, 'sigerr':       None, 'reff':  374.},
              'DDO 113'      : {'type': 'LG'     , 'mv': -11.0, 'mstar':  2.1e6, 'dsun': 2951., 'sigma': None, 'sigerr':       None, 'reff':  601.}}



NCLASSICAL = 11


MLIMIT_SDSS  = 22.0
MLIMIT_LSST1 = 23.8
MLIMIT_LSSTC = 26.8
MLIMIT_DES   = 24.7

def dmax(mobs,mlimit,algorithm='walsh',router=300.):
    """
    Returns maximum distance a dwarf of magnitude 'mobs' could be detected in a survey
    with limiting magnitude 'mlimit' based on the detection limit 'algorithm' 
    (default 'walsh') in a halo of radius 'router' kpc (default 300).

    mlimit = a number or 'full'
    """

    if mlimit == 'full':
        return router
    else:
        mlimit = float(mlimit)

    if algorithm == 'walsh':
        return 10**( -0.204*mobs + 1.164 + (mlimit-MLIMIT_SDSS)/5. )
    if algorithm == 'kopos':
        return 10**( (-0.6/3.)*mobs + log10(1071.6) + (-5.23/3) + (mlimit-MLIMIT_SDSS)/5. )


C      = 9.0 # duffy08(mMW) # 12.
RVIR   = 300. # kpc
rs     = RVIR/C
print('--> NOTE: for completeness corrections, hard-coding MW NFW parameters to C =',C,' RS',rs,'kpc  RVIR',RVIR,'kpc')

AREA_DR5     =  8000. # area of SDSS DR5
AREA_DR8     = 12000. # area of SDSS DR8 (actually 14,500 but need to cut out low Galactic latitudes)
AREA_LSST    = 20000.
AREA_DES     =  5000.
AREA_SKY     = 41253. # total area of the sky (sq. deg)



# ------------------------------------------------------------------------------
# radial correction functions

def correct(profiles):

    menc_fxns   = []
    rdist_names = []
    interp_fxns = []
    ifxn = 0

    r = arange(10,300)

    for profile in profiles:

        base,mod = (profile,None) if ',' not in profile else profile.split(',')

        if   base == 'nfw' :
            menc_fxns += [ lambda rc,t: (log(1.+rc/rs)-1./(rs/rc+1.)) / (log(1.+C) - 1./(1/C + 1)) ]
            rdist_names += ['NFW']
        elif base == 'sis' :
            menc_fxns += [ lambda rc,t: rc/RVIR ]
            rdist_names += ['SIS']
        elif base == 'hern':
            menc_fxns += [ lambda rc,t: ( (1./C+1) / (rs/rc+1) )**2 ]
            rdist_names += ['Hernquist']
        elif base == 'ein':
            C200  = 4.9
            ALPHA = 0.24
            rho = lambda rr: exp(-2./ALPHA * (C200*rr)**ALPHA)
            menc = array([ quad(lambda rr: rho(rr)*rr**2, 0,rend)[0] for rend in r/RVIR ])
            menc /= menc[-1]
            menc = concatenate([[0],menc,[1]])
            interp_fxns += [ interp1d(concatenate([[0],r,[300]]),menc) ] #,fill_value=[0,1],bounds_error=False) ]
            menc_fxns += [ lambda rc,t,i=ifxn: interp_fxns[i](rc) ]
            ifxn += 1
            rdist_names += ['Einasto']
        else:  # we'll try to read menc profile from a data file

            datfn = RDIST_DATDIR+'nenc3d/'+base+'-menc.dat'

            if 'tidal' not in base:
                rr,mm = loadtxt(datfn,unpack=True)
                rr *= 300 if rr[-1]/100 < 1 else 1
                interp_fxns += [ interp1d(rr,mm/mm[-1]) ]
                menc_fxns += [ lambda rc,t,i=ifxn: interp_fxns[i](rc) ]
                ifxn += 1
            else:
                rr,mm = loadtxt(datfn,unpack=True,usecols=(0,3)) # assume they're all severely tidally stripped
                rr *= 300 if rr[-1]/100 < 1 else 1
                interp_fxns += [ interp1d(rr,mm/mm[-1]) ]
                menc_fxns += [ lambda rc,t,i=ifxn: interp_fxns[i](rc) ]
                ifxn += 1
                """
                rts,rnts, mts,mnts = loadtxt(datfn,unpack=True,usecols=(0,1,3,4))
                rts  *= 300 if rts[-1] /100 < 1 else 1
                rnts *= 300 if rnts[-1]/100 < 1 else 1
                interp_fxns += [ interp1d(rts ,mts /mts [-1]) ]  # tidally stripped
                interp_fxns += [ interp1d(rnts,mnts/mnts[-1]) ]  # not tidally stripped
                menc_fxns += [ lambda rc,t,i=ifxn,j=(ifxn+1): (interp_fxns[i] if 'tidal' in t else interp_fxns[j])(rc) ]
                ifxn += 2
                """

            rdist_names += [ base[:].upper() ]

        # read in modification to profile, if provided
        if mod != None:
            datfn = RDIST_DATDIR+'nenc3d/'+mod+'-mod.dat'
            print('reading modification from',datfn)
            rr,mm = loadtxt(datfn,unpack=True)
            rr *= 300 if rr[-1]/100 < 1 else 1
            interp_fxns += [ menc_fxns[len(menc_fxns)-1] ]
            interp_fxns += [ interp1d(rr,mm/mm[-1]) ]
            menc_fxns[-1] = lambda rc,t,i=ifxn: interp_fxns[i](rc,t)*interp_fxns[i+1](rc)
            ifxn += 2

            rdist_names[-1] += '+'+mod

        # solve for radius that encloses half the satellites
        result = minimize_scalar(lambda rr: abs(menc_fxns[-1](rr,1)-0.5), bounds=(10,300), method='Bounded')
        print('for',rdist_names[-1],'r1/2 =',result.x,'kpc')



    # -------------------------------------------------------------------------
    # correction calculation

    crs_tot = [[],[],[]]

    for dwarf_name,dwarf in dwarfs.items():

        if 'classical' in dwarf['type']:  continue

        rc = dmax(dwarf['mv'],MLIMIT_SDSS)
        rc = RVIR if rc > RVIR else rc

        print('{0:<20}  {1:>7} kpc'.format(dwarf_name,round(rc,2)),end=' ')

        for im,maglim in enumerate(['full', MLIMIT_DES, MLIMIT_LSST1]):
            rout = dmax(dwarf['mv'],maglim)
            rout = RVIR if rout > RVIR else rout
            print('| {0:>7} kpc '.format(round(rout,2)),end='')
            crs_tot[im] += [ [ menc(rout,dwarf['type'])/menc(rc,dwarf['type']) for menc in menc_fxns ] ]
            #crs_tot[im] += [ [ crfxn(rc,dwarf['type'])/crfxn(rout,dwarf['type']) for crfxn in crfxns ] ]
            print(' '.join(['{0:<7}'.format(round(cr,3)) for cr in crs_tot[im][-1]]),end=' ')
        print()


    # and print the results
    for iml,maglim in enumerate(['full','DES', 'LSST']):

        print()
        print('MAGLIM',maglim,end=' ')

        if maglim == 'full':  C_OMEGA = AREA_SKY  / AREA_DR8
        if maglim == 'DES' :  C_OMEGA = AREA_DES  / AREA_DR8
        if maglim == 'LSST':  C_OMEGA = AREA_LSST / AREA_DR8

        print('C_OMEGA',C_OMEGA)

        for ird,name in enumerate(rdist_names):
            crs = [crs_tot[iml][i][ird] for i in range(len(crs_tot[0]))]
            print(name+':',int(round(sum(crs)*C_OMEGA,2)),'total satellites (not including classicals)') # + NCLASSICAL


    return crs_tot, menc_fxns, rdist_names # crs_tot[maglims][dwarfs][profiles]



# -----------------------------------------------------------------------------
# velocity function, completeness corrected, for full sky

def vcorrect(profiles,bootstrap=True,nboot=1000,obs_uncertainty=None):
    """
    Returns a list of the velocities of a completeness-corrected sample of galaxies.
    obs_uncertainty = None, 'sigdiv2' to divide all uncertainties by a factor of 2, or
       'BooIIAM' to use the McConnachie+ 2012 values/uncertainties for Boo II, or 
       'strip' to use 2*sig for unobserved analogs of likely stripped dwarfs
    """

    nprofiles = len(profiles)
    id = -1  # UFD dwarf index

    # do radial completeness correction
    crs_tot, menc_fxns, rdist_names = correct(profiles)

    if bootstrap:
        sigmas = [ [ [] for ib in range(nboot) ] for ird in range(nprofiles) ]
    else:
        sigmas = [ [] for ird in range(nprofiles) ]


    for dwarf_name,dwarf in dwarfs.items():

        if dwarf['sigma'] == None: continue

        if dwarf['type'] == 'classical':
            nsigmas = ones(nprofiles)
        else:
            id += 1
            nsigmas = array(crs_tot[0][id])*(AREA_SKY/AREA_DR8) # for all profiles, full volume + full sky correction

        if not bootstrap:
            for ip in range(nprofiles): sigmas[ip] += [ [dwarf['sigma']]*int(round(nsigmas[ip])) ]
        else:
            for ip in range(nprofiles):
                for ib in range(nboot):
                    if dwarf_name=='Bootes II' and obs_uncertainty=='BooIIAM':
                        sigmas[ip][ib] += [ normal(loc=10.5,scale=7.4,
                                                   size=int(round(nsigmas[ip] * (1 if nsigmas[ip]==1 else normal(loc=5.15,scale=1.)/5.15) ))) ]  # anisotropy if completeness-corrected
                    elif dwarf['type']=='ufd tidal' and obs_uncertainty=='strip': # dwarf_name=='Segue I'  dwarf_name=='Hercules'
                        sigmas[ip][ib] += [ normal(loc=dwarf['sigma']*2,scale=mean(dwarf['sigerr']),
                                                   size=int(round(nsigmas[ip] * (1 if nsigmas[ip]==1 else normal(loc=5.15,scale=1.)/5.15) ))) ] 
                    else:
                        sigmas[ip][ib] += [ normal(loc=dwarf['sigma'],scale=mean(dwarf['sigerr'])*(0.5 if obs_uncertainty=='sigdiv2' else 1.),  ## modify all sigmas
                                                   size=int(round(nsigmas[ip] * (1 if nsigmas[ip]==1 else normal(loc=5.15,scale=1.)/5.15) ))) ]

    if bootstrap:
        for ip in range(nprofiles):
            for ib in range(nboot): sigmas[ip][ib] = concatenate(sigmas[ip][ib])

    return array(sigmas), rdist_names, nboot


# ==============================================================================
# MAIN

if __name__ == '__main__':
    if len(argv) < 2:
        print('usage: {0} [menc1](,mod1) .. [mencN](,modN)'.format(argv[0]))
        print("   mencX and modX can be nfw, sis, hern, or point to file [mencX]-menc.dat or [modX]-mod.dat")
        exit()

    profiles = argv[1:]

    vcorrect(profiles)
