# dwarfs.py
# modified 2019.11.04 by stacy kim
# vcorrect.py created 2017.05.25

from numpy import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import os.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



# MW dwarf data, from Jethwa+2017's compilation, who got mags and dist from
# McConnachie 2012, and M* from Woo+ 2008 or assuming M*/L = 2 MSUN/LSUN
#           name                                        v-band Mabs  MSUN            kpc           velocity disp, error (km/s)          reff (pc)      ueff (M/arcsec^2)
dwarfsMW = {'LMC'              : {'type': 'classical', 'mv': -18.1, 'mstar': 1.1e9, 'dsun': 51.0, 'sigma': 20.2, 'sigerr':        0.5, 'reff':  None, 'ueff': None},
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
            #'Tucana III'       : {'type': 'tidal'    , 'mv': },}}


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
              #'IC 1613'      : {'type': 'LG'     , 'mv': -15.2, 'mstar':  1.0e8, 'dsun':  755., 'sigma': None, 'sigerr':       None, 'reff': 1496.},
              'Phoenix'      : {'type': 'LG'     , 'mv':  -9.9, 'mstar':  7.7e5, 'dsun':  415., 'sigma': None, 'sigerr':       None, 'reff':  454.},
              'NGC 6822'     : {'type': 'LG'     , 'mv': -15.2, 'mstar':  1.0e8, 'dsun':  459., 'sigma': None, 'sigerr':       None, 'reff':  354.},
              'Cetus'        : {'type': 'LG'     , 'mv': -11.2, 'mstar':  2.6e6, 'dsun':  755., 'sigma': 17.0, 'sigerr':        2.0, 'reff':  703.},
              'Pegasus dIrr' : {'type': 'LG'     , 'mv': -12.2, 'mstar': 6.61e6, 'dsun':  920., 'sigma': None, 'sigerr':       None, 'reff':  562.},
              #'Leo T'        : {'type': 'LG'     , 'mv':  -8.0, 'mstar':  1.4e5, 'dsun':  417., 'sigma':  7.5, 'sigerr':        1.6, 'reff':  120.},
              #'WLM'          : {'type': 'LG'     , 'mv': -14.2, 'mstar':  4.3e7, 'dsun':  933., 'sigma': 17.5, 'sigerr':        2.0, 'reff': 2111.},
              'Leo A'        : {'type': 'LG'     , 'mv': -12.1, 'mstar':  6.0e6, 'dsun':  798., 'sigma':  9.3, 'sigerr':        1.3, 'reff':  499.},
              'And XVIII'    : {'type': 'LG'     , 'mv':  -9.7, 'mstar':  6.3e5, 'dsun': 1355., 'sigma': None, 'sigerr':       None, 'reff':  363.},
              #'Aquarius'     : {'type': 'LG'     , 'mv': -10.6, 'mstar':  1.6e6, 'dsun': 1072., 'sigma': None, 'sigerr':       None, 'reff':  458.},
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
              #'UGC 8508'     : {'type': 'LG'     , 'mv': -13.4, 'mstar':  1.9e7, 'dsun': 2582., 'sigma': None, 'sigerr':       None, 'reff':  315.},
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



##################################################
# MASS-CONCENTRATION RELATION
# relic! only in here b/c needed for Brook+2014 conversion.

# Duffy et al. 2008 mass-concentration relation
# technically only fit to 1e11-1e15 MSUN halos...
# note that their relation was only fit to galaxies out to z = 2.
A200_DUFFY = 5.71     # coefficient
B200_DUFFY = -0.084   # mass scaling
C200_DUFFY = -0.47    # redshift scaling

Avir_DUFFY = 5.71     # coefficient
Bvir_DUFFY = -0.084   # mass scaling
Cvir_DUFFY = -0.47    # redshift scaling

h       = 0.671    # normalized hubble's constant
MPIVOT  = 2e12/h   # mormalized mass, in MSUN




##################################################
# SMHM RELATIONS

# Moster+ 2013's redshift-dependent SMHM relation (all in MSUN units)
M10_M13 = 11.590  # +- 0.236
M11_M13 =  1.195  # +- 0.353
N10_M13 =  0.0351 # +- 0.0058
N11_M13 = -0.0247 # +- 0.0069
b10_M13 =  1.376  # +- 0.153
b11_M13 = -0.826  # +- 0.225
g10_M13 =  0.608  # +- 0.608
g11_M13 =  0.329  # +- 0.173

mhM13,msM13 = loadtxt(SCRIPT_DIR+'/moster.dat',unpack=True) # just for z=0
mhaloM13 = interp1d(log(msM13),log(mhM13),kind='linear',fill_value='extrapolate', bounds_error=False)


# Behroozi+ 2013 z=0 relation
mhB13,msB13 = loadtxt(SCRIPT_DIR+'/behroozi.dat' ,unpack=True)
mhaloB13 = interp1d(log(msB13),log(mhB13),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarB13 = interp1d(log(mhB13),log(msB13),kind='linear',fill_value='extrapolate',bounds_error=False)


# Brook+ 2014 z=0 relation
f = lambda c,x: (log(1+c)-c/(1+c))*(350/200.)*x**3 - log(1+c*x) + 1/(1+1/c/x)  # fxn to find root for R350 (see pg. 5B of cat notebook)
c200   = arange(5,25+1.)  # NFW concentrations
x350   = array( [ brentq(lambda x: f(float(cc),x),0.1,1) for cc in c200 ] )  # R350/R200
a200   = log(1+c200) - c200/(1+c200)
a350   = log(1+c200*x350) - 1/(1+1/c200/x350)
mratio = a200/a350  # M200/M350 as a function of c200, i.e. correction factor!
m200   = (c200/A200_DUFFY)**(1/B200_DUFFY)*MPIVOT  # invert D08 relation
m350   =  m200/mratio

mh350B14,msB14 = loadtxt(SCRIPT_DIR+'/brook.dat' ,unpack=True)
mh200B14 = interp(mh350B14,m350[::-1],m200[::-1])  # mhalo = peak M350, convert to M200 assuming NFW
mhaloB14 = interp1d(log(msB14),log(mh200B14),kind='linear',fill_value='extrapolate',bounds_error=False)
mstarB14 = interp1d(log(mh200B14),log(msB14),kind='linear',fill_value='extrapolate',bounds_error=False)




def smhm_mstar(mhalo,z=0,model='moster13'):
    """Gives M* given Mhalo, all in MSUN units."""

    if model == 'moster13':
        M1    = 10**( M10_M13 + M11_M13 * z/(z+1) )
        N     = N10_M13 + N11_M13 * z/(z+1)
        beta  = b10_M13 + b11_M13 * z/(z+1)
        gamma = g10_M13 + g11_M13 * z/(z+1)
        return 2 * N * mhalo / ( (mhalo/M1)**-beta + (mhalo/M1)**gamma )

    elif model == 'behroozi13':
        if z != 0:  raise ValueError('No support for z != 0 for Behroozi+ 2013 SMHM relation.')
        return exp(mstarB13(log(mhalo)))

    elif model == 'brook14':
        if z != 0:  raise ValueError('No support for z != 0 for Brook+ 2014 SMHM relation.')
        return exp(mstarB14(log(mhalo)))

    else:
        raise ValueError('No support for given SMHM model.')
    


def smhm_mhalo(mstar,z=0,model='moster13'):
    """Gives M* given Mhalo, all in MSUN units."""

    if z != 0:
        raise ValueError('No support for z != 0.')
    
    if   model == 'moster13'  :  return exp(mhaloM13(log(mstar)))
    elif model == 'behroozi13':  return exp(mhaloB13(log(mstar)))
    elif model == 'brook14'   :  return exp(mhaloB14(log(mstar)))
    else:
        raise ValueError('No support for given SMHM model.')

