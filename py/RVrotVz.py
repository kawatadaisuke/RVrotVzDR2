
#
# RVrotVz_FRVS.py
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
import scipy.interpolate
from galpy.util import bovy_coords

##### main programme start here #####

# flags

# constant for proper motion unit conversion
pmvconst = 4.74047
usun = 11.1
vsun = 239.08
wsun = 7.25
# circular velocity at Rsun, p.8 of Sharma et al. (2011)
vcircsun = 226.84
rsun = 8.0
zsun = 0.015

# condition to select stars 
e_plxlim = 0.15
zmaxlim = 0.2
ymaxlim = 0.5

nsample = 3

# for isamp in range(nsample):
for isamp in range(nsample):
    # input data
    if isamp == 0:
        # Bright F stars with RVS data
        infile = 'galaxia_gaia_V13.fits'
        star_hdus = pyfits.open(infile)
        star = star_hdus[1].data
        star_hdus.close()

    elif isamp == 1:
        infilel0 = 'galaxia_gaia_l0.fits'
        starl0 = pyfits.open(infilel0)
        infilel180 = 'galaxia_gaia_l180.fits'
        starl180 = pyfits.open(infilel180)
        nrowsl0 = starl0[1].data.shape[0]
        nrowsl180 = starl180[1].data.shape[0]
        nrows = nrowsl0 + nrowsl180
        star_hdu = pyfits.BinTableHDU.from_columns(starl0[1].columns, nrows=nrows)
        for colname in starl0[1].columns.names:
            star_hdu.data[colname][nrowsl0:] = starl180[1].data[colname]
        star = star_hdu.data
        starl0.close()
        starl180.close()

    print ' number of stars read=', len(star['Plx_obs'])

    # assume Av_obs~AG_obs for Galaxia
    gabsmag = star['G_obs']-(5.0*np.log10(100.0/np.fabs(star['Plx_obs'])))+star['Av_obs']
    zabs = np.fabs((1.0/star['Plx_obs']) \
          *np.sin(np.pi*star['GLAT_true']/180.0)+zsun)
    yabs = np.fabs((1.0/star['Plx_obs']) \
          *np.sin(np.pi*star['GLON_true']/180.0))
    # sindx=np.where((zabs<zmaxlim) & np.logical_or(star['GLON_true']<90.0,star['GLON_true']>270.0))

    if np.logical_or(isamp == 0, isamp == 1):
        # F star 
        if isamp == 0:
            print ' for bright F stars'  
        else:
            print ' for faint F stars'  
        Tefflow = 6600.0
        Teffhigh = 6900.0
    elif isamp == 2:
        print ' for A stars'
        Tefflow = 7330.0
        Teffhigh = 10000.0

    # minimum distance limit
    distmin = 0.0000000001

    sindx = np.where((zabs < zmaxlim) & (yabs < ymaxlim) &
                 (gabsmag > -(2.5/4000.0)*(star['Teff_obs']-6000.0)+1.0) &
                 (star['Plx_obs']>0.0) & (star['Plx_obs']<1.0/distmin) & 
                 (star['e_Plx']/star['Plx_obs']<e_plxlim) & 
                 (star['Teff_obs']>Tefflow) & (star['Teff_obs']<Teffhigh))
    nstars = len(star['RA_obs'][sindx])

    print ' N selected=',nstars
    # extract the stellar data
    ras = star['RA_obs'][sindx]
    decs = star['DEC_obs'][sindx]
    glons = star['GLON_true'][sindx]
    glats = star['GLAT_true'][sindx]
    plxs_obs = star['Plx_obs'][sindx]
    pmras_obs = star['pmRA_obs'][sindx]
    pmdecs_obs = star['pmDEC_obs'][sindx]
    e_plxs = star['e_Plx'][sindx]
    e_pmras = star['e_pmRA'][sindx]
    e_pmdecs = star['e_pmDEC'][sindx]
    # HRV
    hrvs_obs = star['HRV_obs'][sindx]
    e_hrvs = star['e_HRV'][sindx]
    # G, G_BP, G_RP
    gmag_obs = star['G_obs'][sindx]
    gbpmag_obs = star['G_BP_obs'][sindx]
    grpmag_obs = star['G_RP_obs'][sindx]
    e_gmag = star['e_G'][sindx]
    e_gbpmag = star['e_G_BP'][sindx]
    e_grpmag = star['e_G_RP'][sindx]
    # Teff
    teff_obs = star['Teff_obs'][sindx]
    e_teff = star['e_Teff'][sindx]
    # Av
    av_obs = star['Av_obs'][sindx]

    # age [Fe/H] only for Galaxia
    fehs_true = star['[Fe/H]_true'][sindx]
    ages_true = star['Age'][sindx]

    # convert deg -> rad
    glonrads = glons*np.pi/180.0
    glatrads = glats*np.pi/180.0

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
    # galactic position
    distxys_obs = dists_obs*np.cos(glatrads)
    xpos_obs = distxys_obs*np.cos(glonrads)
    ypos_obs = distxys_obs*np.sin(glonrads)
    zpos_obs = dists_obs*np.sin(glatrads)
    xposgals_obs = xpos_obs-rsun
    yposgals_obs = ypos_obs
    rgals_obs = np.sqrt(xposgals_obs**2+yposgals_obs**2)

    if isamp == 0:
        Tvxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(hrvs_obs, Tpmllpmbb[:,0], \
          Tpmllpmbb[:,1], glons, glats, dists_obs, XYZ=False, degree=True)
        vxs_obs = Tvxvyvz[:,0]
        vys_obs = Tvxvyvz[:,1]
        vzs_obs = Tvxvyvz[:,2]+wsun
        # Galactocentric position and velcoity
        hrvxys_obs = hrvs_obs*np.cos(glatrads)
        vxgals_obs = vxs_obs+usun
        vygals_obs = vys_obs+vsun
        vrots_obs = (vxgals_obs*yposgals_obs-vygals_obs*xposgals_obs)/rgals_obs
        vrads_obs = (vxgals_obs*xposgals_obs+vygals_obs*yposgals_obs)/rgals_obs
    else:
        # approximation
        vrots_obs = np.copy(vlons_obs)
        vrots_obs[np.logical_or(glons<90, glons>270)] = \
            vrots_obs[np.logical_or(glons<90, glons>270)]+vsun
        vrots_obs[np.logical_and(glons>=90, glons<=270)] = \
            -vrots_obs[np.logical_and(glons>=90, glons<=270)]+vsun
        vrads_obs = np.zeros_like(vrots_obs)
        vzs_obs = np.copy(vlats_obs)+wsun

    # Vrot defined w.r.t. solar velocity
    vrots_obs -= vcircsun

    if isamp == 0:
        f=open('star_RVrotVz_FRVS.asc','w')
    elif isamp == 1:
        f=open('star_RVrotVz_FF.asc','w')
    else:
        f=open('star_RVrotVz_A.asc','w')
    for i in range(nstars):
        print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f" \
          %(xpos_obs[i], ypos_obs[i], zpos_obs[i], rgals_obs[i], vrots_obs[i], \
            vrads_obs[i], vzs_obs[i], dists_obs[i], glons[i], glats[i], \
            fehs_true[i], ages_true[i])
    f.close()

    # output velocity dispersion of the sample
    print ' velocity dispersion Vrot, Vz = ', np.std(vrots_obs), np.std(vzs_obs)

    # plot R vs Vrot
    plt.scatter(rgals_obs, vrots_obs, c=ages_true)
    plt.xlabel(r"Rgal (kpc)", fontsize=18, fontname="serif")
    plt.ylabel(r"Vrot (km/s)", fontsize=18, fontname="serif")
    cbar=plt.colorbar()
    cbar.set_label(r'Age')
    plt.show()

    # plot R vs Vz
    plt.scatter(rgals_obs, vzs_obs, c=ages_true)
    plt.xlabel(r"Rgal (kpc)", fontsize=18, fontname="serif")
    plt.ylabel(r"Vz (km/s)", fontsize=18, fontname="serif")
    # plt.axis([-1.0,1.0,-1.0,1.0],'scaled')
    cbar=plt.colorbar()
    cbar.set_label(r'Age')
    plt.show()

    # minimum number of stars in each column
    nsmin = 25
    # set number of grid
    ngridx = 40
    ngridy = 40
    # grid plot for R vs. Vrot
    rrange = np.array([rsun-4.0, rsun+4.0])
    vrotrange = np.array([-30, 30.0])

    # 2D histogram 
    H, xedges, yedges = np.histogram2d(rgals_obs, vrots_obs, \
                   bins=(ngridx, ngridy), \
                   range=(rrange, vrotrange))
    # set x-axis (Rgal) is axis=1
    H = H.T
    # normalised by column
    # print ' hist = ',H
    # print ' np column = ',np.sum(H, axis=0)
    H[:, np.sum(H, axis=0)<nsmin] = 0.0
    H[:, np.sum(H, axis=0)>=nsmin] = H[:, np.sum(H, axis=0)>=nsmin] \
      / np.sum(H[:, np.sum(H, axis=0)>=nsmin], axis=0)
    # print ' normalised hist = ',H
    # plt.imshow(H, interpolation='gaussian', origin='lower', aspect='auto', \
    #    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # plt.colorbar(im)
    # plt.show()

    if isamp == 0:
        HFRVS_RVrot = np.copy(H)
        HFRVS_RVrot_xedges = np.copy(xedges)
        HFRVS_RVrot_yedges = np.copy(yedges)
    elif isamp == 1:
        HFF_RVrot = np.copy(H)
        HFF_RVrot_xedges = np.copy(xedges)
        HFF_RVrot_yedges = np.copy(yedges)
    else:
        HA_RVrot = np.copy(H)
        HA_RVrot_xedges = np.copy(xedges)
        HA_RVrot_yedges = np.copy(yedges)


    # grid plot for R vs. Vz
    ngridx = 40
    ngridy = 20
    rrange = np.array([rsun-4.0, rsun+4.0])
    vzrange = np.array([-20.0, 20.0])
    # 2D histogram 
    H, xedges, yedges = np.histogram2d(rgals_obs, vzs_obs, \
                   bins=(ngridx, ngridy), \
                   range=(rrange, vzrange))
    # set x-axis (Rgal) is axis=1
    H = H.T
    # normalised by column
    H[:, np.sum(H, axis=0)<nsmin] = 0.0
    H[:, np.sum(H, axis=0)>=nsmin] = H[:, np.sum(H, axis=0)>=nsmin] \
      / np.sum(H[:, np.sum(H, axis=0)>=nsmin], axis=0)
    # plt.imshow(H, interpolation='nearest', origin='low', aspect='auto', \
    # plt.imshow(H, interpolation='gaussian', origin='lower', aspect='auto', \
    #       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # plt.colorbar()
    # plt.show()

    if isamp == 0:
        HFRVS_RVz = np.copy(H)
        HFRVS_RVz_xedges = np.copy(xedges)
        HFRVS_RVz_yedges = np.copy(yedges)
    elif isamp == 1:
        HFF_RVz = np.copy(H)
        HFF_RVz_xedges = np.copy(xedges)
        HFF_RVz_yedges = np.copy(yedges)
    else:
        HA_RVz = np.copy(H)
        HA_RVz_xedges = np.copy(xedges)
        HA_RVz_yedges = np.copy(yedges)


# combined plot for R vs Vrot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16

# colour mapscale
cmin = 0.0
cmax = 0.1

f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(6,8))
ax1.imshow(HFRVS_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFRVS_RVrot_xedges[0], HFRVS_RVrot_xedges[-1], \
                   HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
ax1.set_ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
ax1.set_yticks([-20.0, -10, 0.0, 10.0, 20.0])
ax2.imshow(HFF_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFF_RVrot_xedges[0], HFF_RVrot_xedges[-1], \
                   HFF_RVrot_yedges[0], HFF_RVrot_yedges[-1]])
ax2.set_ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
ax2.set_yticks([-20.0, -10, 0.0, 10.0, 20.0])
im = ax3.imshow(HA_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HA_RVrot_xedges[0], HA_RVrot_xedges[-1], \
                   HA_RVrot_yedges[0], HA_RVrot_yedges[-1]])
ax3.set_yticks([-20.0, -10, 0.0, 10.0, 20.0])
plt.xlabel(r"R$_{\rm gal}$ (kpc)", fontsize=18)
plt.ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
f.subplots_adjust(hspace=0.01, right = 0.8)
cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
f.colorbar(im, cax=cbar_ax)
plt.show()

# R vs. Vz
f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(6,8))
ax1.imshow(HFRVS_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFRVS_RVz_xedges[0], HFRVS_RVz_xedges[-1], \
                   HFRVS_RVz_yedges[0], HFRVS_RVz_yedges[-1]])
ax1.set_ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
ax1.set_yticks([-10, 0.0, 10.0])
ax2.imshow(HFF_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFF_RVz_xedges[0], HFF_RVz_xedges[-1], \
                   HFF_RVz_yedges[0], HFF_RVz_yedges[-1]])
ax2.set_ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
ax2.set_yticks([-10, 0.0, 10.0])
ax3.imshow(HA_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HA_RVz_xedges[0], HA_RVz_xedges[-1], \
                   HA_RVz_yedges[0], HA_RVz_yedges[-1]])
ax3.set_yticks([-10, 0.0, 10.0])
plt.xlabel(r"R$_{\rm z}$ (kpc)", fontsize=18)
plt.ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
f.subplots_adjust(hspace=0.01)
f.subplots_adjust(hspace=0.01, right = 0.8)
cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
f.colorbar(im, cax=cbar_ax)
plt.show()

plt.show()





