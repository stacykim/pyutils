# nearby_galaxies.py
# created 2019.08.27
#
# reads the table by alan mcconachie

import sys

dwarfs = {}
m31dwarfs = False

for il,line in enumerate(open('NearbyGalaxies.dat','r').readlines()[34:]):
    if il == 1: continue

    name  = line[:18]
    ra    = line[19:29]
    dec   = line[30:39]
    evb   = line[40:45]   # Foreground reddening, E(B-V), measured directly from the Schlegel et al 1998 maps (they do not include the recalibration by Schlafly & Finbiener 2011)
    mM    = line[46:61]   # Distance Modulus, (m-M)o err+ err-  
    vh    = line[62:78]   # Heliocentric radial velocity, vh(km/s) err+ err- 
    Vmag  = line[79:91]   # Apparent V magnitude in Vega mags, Vmag err+ err- 
    pa    = line[92:107]  # Position Angle of major axis in degrees measured east from north, PA err+ err- 
    e     = line[108:122] # Projected ellipticity, e=1-b/a err+ err- 
    muVo  = line[123:135] # Central V surface brightness, muVo(mag/sq.arcsec) err+ err-
    rh    = line[136:154] # Half-light radius measured on major axis, rh(arcmins) err+ err-
    sigS  = line[155:169] # Stellar radial velocity dispersion, sigma_s(km/s) err+ err-
    vrotS = line[170:184] # Stellar peak/max rotation velocity, vrot_s(km/s) err+ err-
    mHI   = line[185:189] # Mass of HI (calculated for the adopted distance modulus), M_HI (10^6M_sun)
    sigG  = line[190:204] # HI radial velocity dispersion, sigma_g(km/s) err+ err-
    vrotG = line[205:220] # HI peak/max rotation velocity, vrot_g(km/s) err+ err-
    FeH   = line[221:236] # Stellar mean metallicity in dex, [Fe/H] err+ err-
    zflag = line[237:238] # flag for metallicity measurement technique
    refs  = line[239:-1]  # references

    if il == 1:
        column_names = [ra,dec,evb,mM,vh,Vmag,pa,e,muVo,rh,sigS,vrotS,mHI,sigG,vrotG,FeH,zflag,refs]

    if name == 'Andromeda         ': m31dwarfs = True
    if name == 'Andromeda XXVIII  ': m31dwarfs = False
    if m31dwarfs: print name,sigS

    dwarfs[name] = { 'sigS': sigS, 'rhalf': rh, 'vmax': vrotS }
