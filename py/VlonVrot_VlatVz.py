
#
# VlonVrot_VlatVz.py
#
# reading gaia_mock/galaxia_gaia
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy import optimize
from galpy.util import bovy_coords


##### main programme start here #####

# flags
# 0: A star, 1: F star, otherwise all stars
flagAF = 2
# 0: l=0, otherwise: l=180
flagglon = 1

# constant for proper motion unit conversion
pmvconst = 4.74047
usun = 11.1
vsun = 239.08
wsun = 7.25
# circular velocity at Rsun, p.8 of Sharma et al. (2011)
vcircsun = 226.84
rsun = 8.0
zsun = 0.015

# input data
if flagglon == 0:
    print ' read l=0 data'
    infile = 'galaxia_gaia_l0.fits'
else:
    print ' read l=180 data'
    infile = 'galaxia_gaia_l180.fits'

star_hdus = pyfits.open(infile)
star = star_hdus[1].data
star_hdus.close()

print ' number of stars read=', len(star['RA_true'])

# select stars 
e_plxlim = 0.15
zmaxlim = 0.2
ymaxlim = 0.5
gabsmag = star['G_obs']-(5.0*np.log10(100.0/star['Plx_obs']))
zabs = np.fabs((1.0/star['Plx_obs']) \
               *np.sin(np.pi*star['GLAT_true']/180.0)+zsun)
yabs = np.fabs((1.0/star['Plx_obs']) \
               *np.sin(np.pi*star['GLON_true']/180.0))
# sindx=np.where((zabs<zmaxlim) & np.logical_or(star['GLON_true']<90.0,star['GLON_true']>270.0))

if flagAF == 0:
    # A star
    print ' for A stars'
    Tefflow = 7330.0
    Teffhigh = 10000.0
elif flagAF == 1: 
    # F star 
    print ' for F stars'
    # Tefflow = 6000.0
    Tefflow = 6600.0
    Teffhigh = 7330.0
    # Teffhigh = 6900.0
else:
    # F star 
    print ' for all stars'
    Tefflow = 0.0
    Teffhigh = 1000000.0

# minimum distance limit
distmin = 0.0000000001

sindx = np.where((zabs<zmaxlim) & (yabs<ymaxlim) &
                 (gabsmag > -(2.5/4000.0)*(star['Teff_obs']-6000.0)+1.0) &
                 (star['Plx_obs']>0.0) & (star['Plx_obs']<1.0/distmin) & 
                 (star['e_Plx']/star['Plx_obs']<e_plxlim) & 
                 (star['Teff_obs']>Tefflow) & (star['Teff_obs']<Teffhigh))
nstars = len(star['RA_true'][sindx])

print ' N selected=',nstars
# extract the stellar data
ras = star['RA_obs'][sindx]
decs = star['DEC_obs'][sindx]
glons = star['GLON_true'][sindx]
glats = star['GLAT_true'][sindx]
plxs_true = star['Plx_true'][sindx]
pmras_true = star['pmRA_true'][sindx]
pmdecs_true = star['pmDEC_true'][sindx]
plxs_obs = star['Plx_obs'][sindx]
pmras_obs = star['pmRA_obs'][sindx]
pmdecs_obs = star['pmDEC_obs'][sindx]
e_plxs = star['e_Plx'][sindx]
e_pmras = star['e_pmRA'][sindx]
e_pmdecs = star['e_pmDEC'][sindx]
# HRV
hrvs_true = star['HRV_true'][sindx]
hrvs_obs = star['HRV_obs'][sindx]
e_hrvs = star['e_HRV'][sindx]
# G, G_BP, G_RP
gmag_true = star['G_true'][sindx]
gbpmag_true = star['G_BP_true'][sindx]
grpmag_true = star['G_RP_true'][sindx]
gmag_obs = star['G_obs'][sindx]
gbpmag_obs = star['G_BP_obs'][sindx]
grpmag_obs = star['G_RP_obs'][sindx]
e_gmag = star['e_G'][sindx]
e_gbpmag = star['e_G_BP'][sindx]
e_grpmag = star['e_G_RP'][sindx]
# Teff
teff_true = star['Teff_true'][sindx]
teff_obs = star['Teff_obs'][sindx]
e_teff = star['e_Teff'][sindx]
# age [Fe/H]
fehs_true = star['[Fe/H]_true'][sindx]
ages_true = star['Age'][sindx]

# convert deg -> rad
glonrads = glons*np.pi/180.0
glatrads = glats*np.pi/180.0

# get true position and velocity
dists_true = 1.0/plxs_true
# velocity
Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb(pmras_true, pmdecs_true, ras, \
            decs, degree=True, epoch=2000.0)
pmlons_true = Tpmllpmbb[:,0]
pmlats_true = Tpmllpmbb[:,1]
# mas/yr -> km/s
vlons_true = pmvconst*pmlons_true*dists_true
vlats_true = pmvconst*pmlats_true*dists_true
Tvxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(hrvs_true, Tpmllpmbb[:,0], \
          Tpmllpmbb[:,1], glons, glats, dists_true, XYZ=False, degree=True)
vxs_true = Tvxvyvz[:,0]
vys_true = Tvxvyvz[:,1]
vzs_true = Tvxvyvz[:,2]
# Galactocentric position and velcoity
distxys_true = dists_true*np.cos(glatrads)
xpos_true = distxys_true*np.cos(glonrads)
ypos_true = distxys_true*np.sin(glonrads)
zpos_true = dists_true*np.sin(glatrads)
hrvxys_true = hrvs_true*np.cos(glatrads)
vxgals_true = vxs_true+usun
vygals_true = vys_true+vsun
xposgals_true = xpos_true-rsun
yposgals_true = ypos_true
rgals_true = np.sqrt(xposgals_true**2+yposgals_true**2)
vrots_true = (vxgals_true*yposgals_true-vygals_true*xposgals_true)/rgals_true
vrads_true = (vxgals_true*xposgals_true+vygals_true*yposgals_true)/rgals_true

# get observed position and velocity
dists_obs = 1.0/plxs_obs
# velocity
Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb(pmras_obs, pmdecs_obs, ras, \
            decs, degree=True, epoch=2000.0)
pmlons_obs = Tpmllpmbb[:,0]
pmlats_obs = Tpmllpmbb[:,1]
# mas/yr -> km/s
vlons_obs = pmvconst*pmlons_obs*dists_obs
vlats_obs = pmvconst*pmlats_obs*dists_obs
Tvxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(hrvs_obs, Tpmllpmbb[:,0], \
          Tpmllpmbb[:,1], glons, glats, dists_obs, XYZ=False, degree=True)
vxs_obs = Tvxvyvz[:,0]
vys_obs = Tvxvyvz[:,1]
vzs_obs = Tvxvyvz[:,2]+wsun
# Galactocentric position and velcoity
distxys_obs = dists_obs*np.cos(glatrads)
xpos_obs = distxys_obs*np.cos(glonrads)
ypos_obs = distxys_obs*np.sin(glonrads)
zpos_obs = dists_obs*np.sin(glatrads)
hrvxys_obs = hrvs_obs*np.cos(glatrads)
vxgals_obs = vxs_obs+usun
vygals_obs = vys_obs+vsun
xposgals_obs = xpos_obs-rsun
yposgals_obs = ypos_obs
rgals_obs = np.sqrt(xposgals_obs**2+yposgals_obs**2)
vrots_obs = (vxgals_obs*yposgals_obs-vygals_obs*xposgals_obs)/rgals_obs
vrads_obs = (vxgals_obs*xposgals_obs+vygals_obs*yposgals_obs)/rgals_obs

# approximate vrot from vlon
vrotlons_obs =  np.copy(vlons_obs)
vrotlons_obs[np.logical_or(glons<90, glons>270)] = \
    vlons_obs[np.logical_or(glons<90, glons>270)]+vsun
vrotlons_obs[np.logical_and(glons>=90, glons<=270)] = \
    -vlons_obs[np.logical_and(glons>=90, glons<=270)]+vsun
vzlats_obs = np.copy(vlats_obs)+wsun

angs=np.copy(glons)
if flagglon == 0:
    # for l= 0 case
    angs[glons>180.0]=angs[glons>180.0]-360.0
else:
    # for l= 180 case
    angs=glons-180.0

# linear regression of vrots vs. vlons
# obs
# vrotres_obs=vlons_obs-(intercept_obs+slope_obs*vrots_obs)
vrotres_obs=vrotlons_obs-vrots_obs
# linear regression of vrotles vs. angs
slope_obs, intercept_obs, r_value, p_value, std_err = \
    stats.linregress(angs, vrotres_obs)
print 'obs slope, intercept DVrot vs. l=',slope_obs,intercept_obs
# vrotres_obs=vlons_obs-(intercept_obs+slope_obs*vrots_obs)
vzres_obs=vzlats_obs-vzs_obs
# linear regression of vzres vs. angs
slope_obs, intercept_obs, r_value, p_value, std_err = \
    stats.linregress(angs,vzres_obs)
print 'obs slope, intercept DVz vs. l=',slope_obs,intercept_obs
slope_obs, intercept_obs, r_value, p_value, std_err = \
    stats.linregress(angs[plxs_obs>1.0/2.0], vrotres_obs[plxs_obs>1.0/2.0])
print 'd<2 kpc, obs slope, intercept DVz vs. l=',slope_obs,intercept_obs
slope_obs, intercept_obs, r_value, p_value, std_err = \
    stats.linregress(angs[plxs_obs<1.0/2.0], vrotres_obs[plxs_obs<1.0/2.0])
print 'd>2 kpc, obs slope, intercept DVz vs. l=',slope_obs,intercept_obs

print ' std error for Vlon obs =',np.std(vlons_true-vlons_obs)
print ' mean and dispersion of vrot/vlon = ', vrotres_obs.mean(), \
    vrotres_obs.std()
print ' std error for Vlat obs =',np.std(vlats_true-vlats_obs)
print ' mean and dispersion of vz/vlat obs = ',vzres_obs.mean(), \
    vzres_obs.std()

dang = 1.0
nang = 20
ndis = 3
angbin = np.zeros(nang)
vrotres_ang_mean = np.zeros((ndis,nang))
vzres_ang_mean = np.zeros((ndis,nang))
vrotres_ang_std = np.zeros((ndis,nang))
vzres_ang_std = np.zeros((ndis,nang))

distlim = 2.0

for idis in range(ndis):
    filename = 'dvrotdvzglon'+str(flagglon)+'samp'+str(flagAF)+'d'+ \
        str(idis)+'.asc'
    f=open(filename,'w')
    anglimlow = -10.0
    anglimhigh = -9.0
    for ii in range(nang):
        angbin[ii] = 0.5*(anglimlow+anglimhigh)
        if idis == 0:
            sindx = np.where((angs>=anglimlow) & (angs<anglimhigh))
        elif idis == 1:
            sindx = np.where((angs>=anglimlow) & (angs<anglimhigh) & \
            (plxs_obs>1.0/distlim))
        else:
            sindx = np.where((angs>=anglimlow) & (angs<anglimhigh) & \
            (plxs_obs<1.0/distlim))

        vrotres_ang_mean[idis,ii] = np.mean(vrotres_obs[sindx])
        vrotres_ang_std[idis,ii] = np.std(vrotres_obs[sindx])
        vzres_ang_mean[idis,ii] = np.mean(vzres_obs[sindx])
        vzres_ang_std[idis,ii] = np.std(vzres_obs[sindx])
        print >>f, "%f %f %f %f %f %d" %(angbin[ii], \
            vrotres_ang_mean[idis,ii], \
            vrotres_ang_std[idis,ii],vzres_ang_mean[idis,ii], \
            vzres_ang_std[idis,ii],len(vrotres_obs[sindx]))
        anglimlow += dang
        anglimhigh += dang
f.close()

filename = 'star_true_obs_glon'+str(flagglon)+'samp'+str(flagAF)+'.asc'

f=open(filename,'w')
for i in range(nstars):
    print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
    %(xpos_true[i], ypos_true[i], zpos_true[i], \
      rgals_true[i], xpos_obs[i], ypos_obs[i], \
      zpos_obs[i], rgals_obs[i], vlons_true[i], \
      vlats_true[i], vrots_true[i], vrads_true[i], \
      vzs_true[i], vlons_obs[i], vlats_obs[i], \
      vzs_obs[i], vrots_obs[i], vrads_obs[i], \
      angs[i], glats[i], vrotlons_obs[i])
f.close()

# plot mean trend
plt.errorbar(angbin, vrotres_ang_mean[0,:], yerr=vrotres_ang_std[0,:], color='b')
plt.errorbar(angbin, vrotres_ang_mean[1,:], yerr=vrotres_ang_std[1,:], color='r')
plt.errorbar(angbin, vrotres_ang_mean[2,:], yerr=vrotres_ang_std[2,:], color='y')
# linear fit
# slope_obs, intercept_obs, r_value, p_value, std_err = \
#    stats.linregress(angs, vrotres_obs)
# print 'obs slope, intercept DVrot vs. l=',slope_obs,intercept_obs
# xs = np.array([-10.0, 10.0])
# ys = slope_obs*xs+intercept_obs
# plt.plot(xs,ys)
p = np.polyfit(angs,vrotres_obs,3)
print ' n=3 polyfit p=',p
xs = np.linspace(-10.0, 10.0, 100)
ys = p[0]*xs**3+p[1]*xs**2+p[2]*xs+p[0]
plt.plot(xs,ys)
plt.xlabel(r"Angle (deg)", fontsize=18, fontname="serif")
plt.ylabel(r" dVrot (km/s)", fontsize=18, fontname="serif")
plt.grid(True)
plt.show()
plt.errorbar(angbin, vzres_ang_mean[0,:], yerr=vzres_ang_std[0,:], color='r')
plt.xlabel(r"Angle (deg)", fontsize=18, fontname="serif")
plt.ylabel(r" dVz (km/s)", fontsize=18, fontname="serif")
plt.grid(True)
plt.show()

# plot x-y map
plt.scatter(angs, vrotres_obs, c=angs, marker='.')
plt.xlabel(r"Angle (deg)", fontsize=18, fontname="serif")
plt.ylabel(r"d (Vlon-Vrot) (km/s)", fontsize=18, fontname="serif")
# plt.axis([-1.0,1.0,-1.0,1.0],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'angs')
plt.show()

# plot R vs. vrot
plt.scatter(rgals_obs, vrots_obs, c=ages_true, marker='.')
plt.xlabel(r"Rgal (kpc)", fontsize=18, fontname="serif")
plt.ylabel(r"Vrot (km/s)", fontsize=18, fontname="serif")
# plt.axis([-1.0,1.0,-1.0,1.0],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'Age')
plt.show()
