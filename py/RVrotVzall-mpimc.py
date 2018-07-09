
#
# RVrotVzall-mpimc.py
#
# reading gaia_mock/galaxia_gaia
#

import pyfits
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import patches
from scipy import stats
from scipy import optimize
import scipy.interpolate
from galpy.util import bovy_coords, bovy_plot
from extreme_deconvolution import extreme_deconvolution
from mpi4py import MPI

# for not displaying
matplotlib.use('Agg')

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

##### main programme start here #####

# flags
# True: MC sampling on, otherwise no MC or XD
MCsample = False
# True: read star_error*.npy
FileErrors = True
# True: read gaussxd*.asc
FileGauXD = True
# True: output Gaussian model
FigGauMod = False
# flagGalaxia if False, assume the real Gaia data
flagGalaxia = False
# figure for paper
FigPaper = False

if FileGauXD == True:
    MCsample = False

# number of MC sampling
nmc = 100
# number of gaussian model
ngauss_sam=np.array([5, 2])
# epoch
if flagGalaxia ==True:
    epoch = 2000.0
else:
    epoch = 2000.0
# constant for proper motion unit conversion
pmvconst = 4.74047
if flagGalaxia == True:
    # circular velocity at Rsun, p.8 of Sharma et al. (2011)
    usun = 11.1
    vsun = 239.08
    wsun = 7.25
    vcircsun = 226.84
    rsun = 8.0
    zsun = 0.015
else:
    # from Bland-Hawthorn & Gerhard
    usun = 10.0
    vsun = 248.0
    wsun = 7.0
    vcircsun = 248.0-11.0
    rsun = 8.2
    zsun = 0.025

# condition to select stars 
# e_plxlim = 0.15
e_plxlim = 0.1
zmaxlim = 0.2
ymaxlim = 0.2
# submitted
# vloslim = 5.0
vloslim = 100.0
# minimum plx
plxlim=0.001

if myrank == 0:
    if MCsample == True:
        print ' MCsample is on. Nmc = ',nmc

nsample = 2

for isamp in range(nsample):
# for isamp in range(1,3):
    if FileErrors == False:
        # read the data and compute errors
        # input data
        if isamp == 0:
            # Bright F stars with RVS data
            if flagGalaxia == True:
                infile = 'galaxia_gaiadr2_V13.fits'
            else:
                infile = 'DR2/gaiadr2_RVSall.fits'
            star_hdus = pyfits.open(infile)
            star = star_hdus[1].data
            star_hdus.close()
        elif isamp == 1:
            if flagGalaxia == True:
                infilel0 = 'galaxia_gaiadr2_l0.fits'
            else:
                infilel0 = 'DR2/gaiadr2_l0all.fits'
            starl0 = pyfits.open(infilel0)
            if flagGalaxia == True:
                infilel180 = 'galaxia_gaiadr2_l180.fits'
            else:
                infilel180 = 'DR2/gaiadr2_l180all.fits'
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

        if myrank == 0:
            print isamp,' sample number of stars =', len(star['parallax'])

        # assume Av_obs~AG_obs for Galaxia
        gabsmag = star['phot_g_mean_mag'] \
            -(5.0*np.log10(100.0/np.fabs(star['parallax']))) \
            +star['a_g_val']
        zabs = np.fabs((1.0/star['parallax']) \
            *np.sin(np.pi*star['b']/180.0)+zsun)
        yabs = np.fabs((1.0/star['parallax']) \
            *np.sin(np.pi*star['l']/180.0))
        # sindx=np.where((zabs<zmaxlim) & np.logical_or(star['GLON_true']<90.0,star['GLON_true']>270.0))
        if isamp == 0:
            if myrank == 0:
                print ' for RVS stars'  
            Tefflow = 6600.0
            Teffhigh = 6900.0
        elif isamp == 1: 
            if myrank == 0:
                print ' for all faint stars'  
            Tefflow = 6600.0
            # Teffhigh = 7330.0
            Teffhigh = 6000.0
        elif isamp == 2:
            if myrank == 0:
                print ' for A stars'
            Tefflow = 7330.0
            Tefflow = 6600.0
            Teffhigh = 10000.0

        # minimum distance limit
        distmin = 0.0000000001

        if isamp == 0:
            sindx = np.where((zabs < zmaxlim) & (yabs < ymaxlim) &
                 (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
                 (star['parallax_error']/star['parallax']<e_plxlim) & 
                 (star['radial_velocity_error']>0.0) &
                 (star['radial_velocity_error']<vloslim))
#                 (np.logical_or(np.logical_or(star['l']<10.0,star['l']>350.0),
#                  np.logical_and(star['l']<190.0,star['l']>170.0))) &
#                 (np.fabs(star['b'])<10.0) &
        elif isamp == 1:
            # sindx = np.where((zabs < zmaxlim) & (yabs < ymaxlim) &
            #     (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
            #     (star['parallax_error']/star['parallax']<e_plxlim))
            sindx = np.where((zabs < zmaxlim) & (yabs < ymaxlim) &
                 (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
                 (star['parallax_error']/star['parallax']<e_plxlim) &
                 (np.logical_or(np.logical_or(star['l']>355.0,star['l']<5.0),
                  np.fabs(star['l']-180.0)<5.0)))
        else:
            sindx = np.where((zabs < zmaxlim) & (yabs < ymaxlim) &
                 (gabsmag > -(2.5/4000.0)*(star['teff_val']-6000.0)+1.0) &
                 (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
                 (star['parallax_error']/star['parallax']<e_plxlim) & 
                 (star['teff_val']>Tefflow) & (star['teff_val']<Teffhigh))

        nstars = len(star['ra'][sindx])

        if myrank == 0:
            print ' N selected=',nstars
        # extract the stellar data
        ras = star['ra'][sindx]
        decs = star['dec'][sindx]
        glons = star['l'][sindx]
        glats = star['b'][sindx]
        plxs_obs = star['parallax'][sindx]
        pmras_obs = star['pmra'][sindx]
        pmdecs_obs = star['pmdec'][sindx]
        e_plxs = star['parallax_error'][sindx]
        e_pmras = star['pmra_error'][sindx]
        e_pmdecs = star['pmdec_error'][sindx]
        # HRV
        hrvs_obs = star['radial_velocity'][sindx]
        e_hrvs = star['radial_velocity_error'][sindx]
        # G, G_BP, G_RP
        gmag_obs = star['phot_g_mean_mag'][sindx]
        gbpmag_obs = star['phot_bp_mean_mag'][sindx]
        grpmag_obs = star['phot_rp_mean_mag'][sindx]
        # e_gmag = star['e_G'][sindx]
        # e_gbpmag = star['e_G_BP'][sindx]
        # e_grpmag = star['e_G_RP'][sindx]
        # Teff
        teff_obs = star['teff_val'][sindx]
        # e_teff = star['e_Teff'][sindx]
        # Av
        av_obs = star['a_g_val'][sindx]
        # error correalation
        if flagGalaxia == True:
            plxpmra_corrs = np.zeros_like(e_plxs) 
            plxpmdec_corrs = np.zeros_like(e_plxs) 
            pmradec_corrs = np.zeros_like(e_plxs) 
            # age [Fe/H] only for Galaxia
            fehs_true = star['[Fe/H]_true'][sindx]
            ages_true = star['Age'][sindx]
        else:
            plxpmra_corrs = star['parallax_pmra_corr'][sindx]
            plxpmdec_corrs = star['parallax_pmdec_corr'][sindx]
            pmradec_corrs = star['pmra_pmdec_corr'][sindx]
            # age [Fe/H] only for Galaxia
            fehs_true = np.zeros_like(e_plxs)
            ages_true = np.zeros_like(e_plxs)

        # convert deg -> rad
        glonrads = glons*np.pi/180.0
        glatrads = glats*np.pi/180.0

        # get observed position and velocity
        dists_obs = 1.0/plxs_obs

        # velocity
        if flagGalaxia == True:
            Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
                pmras_obs, pmdecs_obs, ras, \
                decs, degree=True, epoch=epoch)
        else:
            Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
                pmras_obs, pmdecs_obs, ras, \
                decs, degree=True, epoch=None)
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
            Tvxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(\
                hrvs_obs, Tpmllpmbb[:,0], Tpmllpmbb[:,1], \
                glons, glats, dists_obs, XYZ=False, degree=True)
            vxs_obs = Tvxvyvz[:,0]
            vys_obs = Tvxvyvz[:,1]
            vzs_obs = Tvxvyvz[:,2]+wsun
            # Galactocentric position and velcoity
            hrvxys_obs = hrvs_obs*np.cos(glatrads)
            vxgals_obs = vxs_obs+usun
            vygals_obs = vys_obs+vsun
            vrots_obs = (vxgals_obs*yposgals_obs-vygals_obs*xposgals_obs) \
                /rgals_obs
            vrads_obs = (vxgals_obs*xposgals_obs+vygals_obs*yposgals_obs) \
                /rgals_obs
        else:
            # approximation
            vrots_obs = np.copy(vlons_obs)
            vrots_obs[np.logical_or(glons<90, glons>270)] = \
                vrots_obs[np.logical_or(glons<90, glons>270)]+vsun
            vrots_obs[np.logical_and(glons>=90, glons<=270)] = \
                -vrots_obs[np.logical_and(glons>=90, glons<=270)]+vsun
            vrads_obs = np.zeros_like(vrots_obs)
            vzs_obs = np.copy(vlats_obs)+wsun

        # set error zero
        e_rgals = np.zeros_like(rgals_obs)
        e_vrots = np.zeros_like(vrads_obs)
        e_vzs = np.zeros_like(vrads_obs)

        if MCsample == True:
            # sample from parallax proper motion covariance matrix
            # parallel MC
            rindx = range(myrank,nstars,nprocs)
            nsrank = len(plxs_obs[rindx])
            plxpmradec_mc = np.empty((nsrank, 3, nmc))
            plxpmradec_mc[:, 0, :] = np.atleast_2d(plxs_obs[rindx]).T
            plxpmradec_mc[:, 1, :] = np.atleast_2d(pmras_obs[rindx]).T
            plxpmradec_mc[:, 2, :] = np.atleast_2d(pmdecs_obs[rindx]).T
            for ii in range(nsrank):
                ip = rindx[ii]
                # constract covariance matrix
                tcov = np.zeros((3, 3))
                # /2 because of symmetrization below
                tcov[0, 0] = e_plxs[ip]**2.0 / 2.0
                tcov[1, 1] = e_pmras[ip]**2.0 / 2.0
                tcov[2, 2] = e_pmdecs[ip]**2.0 / 2.0
                tcov[0, 1] = plxpmra_corrs[ip] * e_plxs[ip] * e_pmras[ip]
                tcov[0, 2] = plxpmdec_corrs[ip] * e_plxs[ip] * e_pmdecs[ip]
                tcov[1, 2] = pmradec_corrs[ip] * e_pmras[ip] * e_pmdecs[ip]
                # symmetrise
                tcov = (tcov + tcov.T)
                # Cholesky decomp.
                L = np.linalg.cholesky(tcov)
                plxpmradec_mc[ii] += np.dot(L, np.random.normal(size=(3, nmc)))

            # distribution of velocity and distance.
            # -> pml pmb
            ratile = np.tile(ras[rindx], (nmc, 1)).flatten()
            dectile = np.tile(decs[rindx], (nmc, 1)).flatten()
            if flagGalaxia == True:
                pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb( \
                    plxpmradec_mc[:, 1, :].T.flatten(), \
                    plxpmradec_mc[:, 2, :].T.flatten(), \
                    ratile, dectile, degree=True, epoch=None)
            else:
                pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb( \
                    plxpmradec_mc[:, 1, :].T.flatten(), \
                    plxpmradec_mc[:, 2, :].T.flatten(), \
                    ratile, dectile, degree=True)
            # reshape
            pmllbb_sam = pmllbb_sam.reshape((nmc, nsrank, 2))
            # distance MC sampling
            plxs_sam = plxpmradec_mc[:, 0, :].T
            # check negative parallax
            plxs_samflat= plxs_sam.flatten()
            copysamflat=np.copy(plxs_samflat)
            if len(copysamflat[plxs_samflat<plxlim])>0: 
                print len(copysamflat[plxs_samflat<plxlim]),' plx set to ',plxlim
            plxs_samflat[copysamflat<plxlim]=plxlim
            plxs_sam = np.reshape(plxs_samflat,(nmc,nsrank))
            # distance
            dists_sam = 1.0/plxs_sam
            # mas/yr -> km/s
            vlons_sam = pmvconst*pmllbb_sam[:,:,0]*dists_sam
            vlats_sam = pmvconst*pmllbb_sam[:,:,1]*dists_sam
            # galactic position
            distxys_sam = dists_sam*np.cos(glatrads[rindx])
            xpos_sam = distxys_sam*np.cos(glonrads[rindx])
            ypos_sam = distxys_sam*np.sin(glonrads[rindx])
            zpos_sam = dists_sam*np.sin(glatrads[rindx])
            rgals_sam = np.sqrt((xpos_sam-rsun)**2+ypos_sam**2)

            if isamp == 0:
                hrvs_sam = np.random.normal(hrvs_obs[rindx], \
                    e_hrvs[rindx], (nmc, nsrank))
                vxvyvz_sam = bovy_coords.vrpmllpmbb_to_vxvyvz( \
                    hrvs_sam.flatten(), pmllbb_sam[:,:,0].flatten(), \
                    pmllbb_sam[:,:,1].flatten(), \
                    np.tile(glons[rindx], (nmc, 1)).flatten(), \
                    np.tile(glats[rindx], (nmc, 1)).flatten(), \
                    dists_sam.flatten(), degree=True)
                vxvyvz_sam = vxvyvz_sam.reshape((nmc, nsrank, 3))
                vxs_sam = vxvyvz_sam[:,:,0]
                vys_sam = vxvyvz_sam[:,:,1]
                vzs_sam = vxvyvz_sam[:,:,2]+wsun
                # 2D velocity
                hrvxys_sam = hrvs_sam*np.cos(glatrads[rindx])
                vxgals_sam = vxs_sam+usun
                vygals_sam = vys_sam+vsun
                vrots_sam = (vxgals_sam*ypos_sam-vygals_sam*(xpos_sam-rsun)) \
                        /rgals_sam
                vrads_sam = (vxgals_sam*(xpos_sam-rsun)+vygals_sam*ypos_sam) \
                        /rgals_sam
                # f = open('mcsample_stars.asc','w')
                # for j in range(100000,100100):
                #    for i in range(nmc):
                #        print >>f, "%d %d %f %f %f %f %f %f" % (i, j, \
                #            plxs_sam[i,j], plxs_obs[j], \
                #            rgals_sam[i,j] , rgals_obs[j], \
                #            vrots_sam[i,j] , vrots_obs[j])
                # f.close()
            else:
                vrots_sam = np.copy(vlons_sam)
                glons_rank = glons[rindx]
                indx = np.where(np.logical_or(glons_rank<90, \
                    glons_rank>270))
                vrots_sam[:,indx] = vrots_sam[:,indx]+vsun
                indx = np.where(np.logical_and(glons_rank>=90, \
                    glons_rank<=270))
                vrots_sam[:,indx] = -vrots_sam[:,indx]+vsun
                vrads_sam = np.zeros_like(vrots_sam)
                vzs_sam = np.copy(vlats_sam)+wsun

            # error estimats dispersion (use observed one for mean value)
            # rgals_obs = np.mean(rgals_sam, axis=0).reshape(nstars)
            e_rgals[rindx] = \
                np.std(rgals_sam, axis=0).reshape(nsrank)
            # vzs_obs = np.mean(vzs_sam, axis=0).reshape(nstars)
            e_vzs[rindx] = \
                np.std(vzs_sam, axis=0).reshape(nsrank)
            # vrots_obs = np.mean(vrots_sam, axis=0).reshape(nstars)
            e_vrots[rindx] = \
                np.std(vrots_sam, axis=0).reshape(nsrank)
            # position
            # xpos_obs = np.mean(xpos_sam, axis=0).reshape(nstars)
            # ypos_obs = np.mean(ypos_sam, axis=0).reshape(nstars)
            # zpos_obs = np.mean(zpos_sam, axis=0).reshape(nstars)
            # dists_obs = np.mean(dists_sam, axis=0).reshape(nstars)
            # vrads_obs = np.mean(vrads_sam, axis=0).reshape(nstars)
            if nprocs > 1:
                # MPI
                # e_rgals
                ncom = len(e_rgals)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_rgals
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_rgals = recvbuf
                # e_vzs
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vzs
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vzs = recvbuf
                # e_vrots
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vrots
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vrots = recvbuf

        # Vrot defined w.r.t. solar velocity
        vrots_obs -= vcircsun
        if MCsample == True:
            vrots_sam -= vcircsun

        if myrank == 0:
            if isamp == 0:
                f=open('star_RVrotVz_RVS.asc','w')
            elif isamp == 1:
                f=open('star_RVrotVz_d10.asc','w')
            else:
                f=open('star_RVrotVz_A.asc','w')
            for i in range(nstars):
                print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
                    %(xpos_obs[i], ypos_obs[i], zpos_obs[i], rgals_obs[i], vrots_obs[i], \
                    vrads_obs[i], vzs_obs[i], dists_obs[i], glons[i], glats[i], \
                    fehs_true[i], ages_true[i], e_rgals[i], e_vrots[i], e_vzs[i])
            f.close()
            # save numpy data
            if isamp == 0:
                f=np.savez('star_RVrotVzerror_FRVS.npz',rgals_obs=rgals_obs, \
                    vrots_obs=vrots_obs,vzs_obs=vzs_obs,
                    e_rgals=e_rgals, e_vrots=e_vrots, e_vzs=e_vzs )
            elif isamp == 1:
                f=np.savez('star_RVrotVzerror_FF.npz',rgals_obs=rgals_obs, \
                    vrots_obs=vrots_obs,vzs_obs=vzs_obs,
                    e_rgals=e_rgals, e_vrots=e_vrots, e_vzs=e_vzs )
            else:
                f=np.savez('star_RVrotVzerror_A.npz',rgals_obs=rgals_obs, \
                    vrots_obs=vrots_obs,vzs_obs=vzs_obs,
                    e_rgals=e_rgals, e_vrots=e_vrots, e_vzs=e_vzs )


    else:
        # read the error files
        if isamp == 0:
            rdata = np.load('star_RVrotVzerror_FRVS.npz')
        elif isamp == 1:
            rdata = np.load('star_RVrotVzerror_FF.npz')
        else:
            rdata = np.load('star_RVrotVzerror_A.npz')
        sindx = np.where(rdata['e_vrots']<5.0)
        rgals_obs = rdata['rgals_obs'][sindx]
        vrots_obs = rdata['vrots_obs'][sindx]
        vzs_obs = rdata['vzs_obs'][sindx]
        e_rgals = rdata['e_rgals'][sindx]
        e_vrots = rdata['e_vrots'][sindx]
        e_vzs = rdata['e_vzs'][sindx]
        if myrank == 0:
            print ' sample, Ns=',isamp,len(e_vzs)

    # output velocity dispersion of the sample
    if myrank == 0:
        print ' velocity dispersion Vrot, Vz = ', \
            np.std(vrots_obs), np.std(vzs_obs)

    # plot R vs Vrot
    # plt.scatter(rgals_obs, vrots_obs, c=ages_true)
    # plt.xlabel(r"Rgal (kpc)", fontsize=18, fontname="serif")
    # plt.ylabel(r"Vrot (km/s)", fontsize=18, fontname="serif")
    # cbar=plt.colorbar()
    # cbar.set_label(r'Age')
    # plt.show()

    # plot R vs Vz
    # plt.scatter(rgals_obs, vzs_obs, c=ages_true)
    # plt.xlabel(r"Rgal (kpc)", fontsize=18, fontname="serif")
    # plt.ylabel(r"Vz (km/s)", fontsize=18, fontname="serif")
    # plt.axis([-1.0,1.0,-1.0,1.0],'scaled')
    # cbar=plt.colorbar()
    # cbar.set_label(r'Age')
    # plt.show()

    if np.logical_or(MCsample == True, FileGauXD == True):

        # fit with mix of Gaussian
        # nvel = 2, 0: Vrot, 1: Vrot
        nvel = 2
        # set range
        rw = 0.2
        if isamp == 0:
            rrangexd = np.array([rsun-3.0, rsun+3.0])
            # nradgridxd = 15+1  
            nradgridxd = 30+1  
            # rrangexd = np.array([rsun-1.0, rsun+1.0])
            # nradgridxd = 3+1  
        elif isamp == 1: 
            rrangexd = np.array([rsun-4.0, rsun+4.0])
            # nradgridxd = 20+1
            nradgridxd = 40+1
            # rrangexd = np.array([rsun-1.0, rsun+1.0])
            # nradgridxd = 3+1  
        rgridxd = np.linspace(rrangexd[0], rrangexd[1], nradgridxd)
        # print ' rgridxd = ',rgridxd

        for ivel in range(nvel):
            if ivel == 0: 
                ngauss = np.max(ngauss_sam)
                # Vrot
                if isamp == 0:
                    gauxd_sig_FRVS = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_amp_FRVS = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_mean_FRVS = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_rr_FRVS = np.zeros((nradgridxd))
                elif isamp == 1:
                    gauxd_sig_FF = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_amp_FF = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_mean_FF = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_rr_FF = np.zeros((nradgridxd))

    if MCsample == True:
        for ivel in range(nvel):
            ngauss = ngauss_sam[ivel]
            # fit with Mixture of Gaussian using XD
            if myrank == 0:
                if ivel == 0:
                    print ngauss,'Gaussian Mixture model for Vrot'
                else:
                    print ngauss,'Gaussian Mixture model for Vz'
            gauxd_amp = np.zeros((nradgridxd,ngauss))
            gauxd_mean = np.zeros((nradgridxd,ngauss))
            gauxd_std = np.zeros((nradgridxd,ngauss))
            gauxd_rr = np.zeros((nradgridxd))
            gauxd_ns = np.zeros((nradgridxd))

            for irad in range(myrank,nradgridxd,nprocs):
                rr = rgridxd[irad]
                cirad = '{0:02d}'.format(int(rr*10))
                # print ' irad = ',cirad
                indx = np.fabs(rgals_obs-rr)<rw
                # fit with mix of Gaussians
                if ivel == 0:
                    # for Vrot
                    vals = vrots_obs[indx]
                    e_vals = e_vrots[indx]
                else:
                    # for Vz
                    vals = vzs_obs[indx]
                    e_vals = e_vzs[indx]

                ydata = np.atleast_2d(vals).T
                ycovar = e_vals**2
                # print ' input shape =',ydata.shape, ycovar
                initamp = np.random.uniform(size=ngauss)
                initamp /= np.sum(initamp)
                m = np.median(vals)
                s= 1.4826*np.median(np.fabs(vals-m))
                print ' iv, ir initial guess of median, sig=',ivel,irad,m,s
                initmean= []
                initcovar= []
                for ii in range(ngauss):
                    initcovar.append(s**2.)
                initcovar= np.array([[initcovar]]).T
                # Now let the means vary
                for ii in range(ngauss):
                    initmean.append(m+np.random.normal()*s)
                initmean= np.array([initmean]).T
                print("iv, ir, lnL",ivel,irad, \
                    extreme_deconvolution(ydata,ycovar, \
                    initamp,initmean,initcovar))
                print("iv, ir, amp, mean, std. dev.",ivel,irad, \
                    initamp,initmean[:,0], \
                    np.sqrt(initcovar[:,0,0]))
                # store the amp and mean
                # sort with amplitude
                sortindx = np.argsort(initamp)
                sortindx = sortindx[::-1]
                # print ' sorted amp, mean = ', initamp[sortindx], \
                #    initmean[sortindx,0]
                gauxd_amp[irad,:] = initamp[sortindx]
                gauxd_mean[irad,:] = initmean[sortindx,0]
                gauxd_std[irad,:] = np.sqrt(initcovar[sortindx,0,0])
                gauxd_rr[irad] = rr
                gauxd_ns[irad] = len(vals)

                # for plot
                if FigGauMod == True:
                    xs = np.linspace(-50.,50.,1001)
                    ys = np.sum(np.atleast_2d( \
                        initamp/np.sqrt(initcovar[:,0,0])).T\
                        *np.exp(-0.5*(xs-np.atleast_2d(initmean[:,0]).T)**2. \
                        /np.atleast_2d(initcovar[:,0,0]).T),axis=0)\
                        /np.sqrt(2.*np.pi)
                    _= bovy_plot.bovy_hist(vals,bins=41,range=[-50.,50.],
                       histtype='step',lw=2.,normed=True,overplot=True)
                    plt.plot(xs,ys)
                    for ii in range(ngauss):
                        ys = (initamp[ii]/np.sqrt(initcovar[ii,0,0]))\
                            *np.exp(-0.5*(xs-initmean[ii,0])**2. \
                            /initcovar[ii,0,0])/np.sqrt(2.*np.pi)
                        plt.plot(xs,ys)
                    # print("Combined <v^2>, sqrt(<v^2>):",combined_sig2( \
                    #    initamp,initmean[:,0],initcovar[:,0,0]),
                    #    np.sqrt(combined_sig2(initamp,initmean[:,0],initcovar[:,0,0])))
                    plt.xlabel(r'$V\,(\mathrm{km\,s}^{-1})$')
                    filename = 'samp'+str(isamp)+'v'+str(ivel)+'r'+cirad+'.png'
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()

            if nprocs > 1:
                # MPI
                # gauxd_ns
                ncom = len(gauxd_ns)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = gauxd_ns
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                gauxd_ns = recvbuf
                # gauxd_rr
                ncom = len(gauxd_rr)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = gauxd_rr
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                gauxd_rr = recvbuf
                # gauxd_amp
                gauxd_ampfl = gauxd_amp.flatten()
                ncom = len(gauxd_ampfl)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = gauxd_ampfl
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                gauxd_ampfl = recvbuf
                gauxd_amp = np.reshape(gauxd_ampfl, \
                    (nradgridxd,ngauss))
                # gauxd_mean
                gauxd_meanfl = gauxd_mean.flatten()
                ncom = len(gauxd_meanfl)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = gauxd_meanfl
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                gauxd_meanfl = recvbuf
                gauxd_mean = np.reshape(gauxd_meanfl, \
                    (nradgridxd,ngauss))
                # gauxd_std
                gauxd_stdfl = gauxd_std.flatten()
                ncom = len(gauxd_stdfl)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = gauxd_stdfl
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                gauxd_stdfl = recvbuf
                gauxd_std = np.reshape(gauxd_stdfl, \
                    (nradgridxd,ngauss))

            if myrank == 0:
                # output amg, mean dispersion    
                
                filename = 'gaussxd'+str(isamp)+'vel'+str(ivel)+'.asc'
                fv = open(filename, 'w')
                print >>fv, "# ngauss = %d, nrgrid = %d" % (ngauss, nradgridxd)

                for irad in range(nradgridxd):
                    if ivel == 0:
                        for ii in range(ngauss):
                            print >>fv, "%d %f %f %f %f %d" % (ii, \
                                gauxd_rr[irad], gauxd_amp[irad,ii], \
                                gauxd_mean[irad,ii], gauxd_std[irad,ii], \
                                gauxd_ns[irad])
                    else:
                        for ii in range(ngauss):
                            print >>fv, "%d %f %f %f %f %d" % (ii, \
                                gauxd_rr[irad], gauxd_amp[irad,ii], \
                                gauxd_mean[irad,ii], gauxd_std[irad,ii], \
                                gauxd_ns[irad])
            if isamp == 0:
                gauxd_sig_FRVS[ivel,:,:ngauss] = gauxd_std
                gauxd_amp_FRVS[ivel,:,:ngauss] = gauxd_amp
                gauxd_mean_FRVS[ivel,:,:ngauss] = gauxd_mean
                gauxd_rr_FRVS = gauxd_rr
            elif isamp == 1:
                gauxd_sig_FF[ivel,:,:ngauss] = gauxd_std
                gauxd_amp_FF[ivel,:,:ngauss] = gauxd_amp
                gauxd_mean_FF[ivel,:,:ngauss] = gauxd_mean
                gauxd_rr_FF = gauxd_rr

            if myrank == 0:
                fv.close()

    # minimum number of stars in each column
    nsmin = 25
    # set number of grid
    # used in the paper submitted
    # ngridx = 40
    # ngridy = 40
    if isamp == 0:
        ngridx = 180
        ngridy = 90
    else:
        ngridx = 150
        ngridy = 50  
        # ngridx = 300
        # ngridy = 150
    # grid plot for R vs. Vrot
    # submitted version
    # rrange = np.array([rsun-4.0, rsun+4.5])
    rrange = np.array([rsun-3.5, rsun+4.0])
    # submitted version
    # vrotrange = np.array([-50, 50.0])
    # vrotticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])
    vrotrange = np.array([-50, 50.0])
    vrotticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])


    # 2D histogram 
    # if MCsample == True:
    # This makes spurious features, because of correlation
    # H, xedges, yedges = np.histogram2d(rgals_sam.flatten(), \
    #               vrots_sam.flatten(), \
    #               bins=(ngridx, ngridy), \
    #               range=(rrange, vrotrange))
    # else : 
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
    # log: linear seems to be better
    # cmin = -5.0
    # cmax = -2.0
    # hlimit = 10.0**cmin
    # hlogindx = np.where(H>=hlimit)
    # hlowindx = np.where(H<hlimit)  
    # H[hlogindx]=np.log10(H[hlogindx])
    # H[hlowindx]=cmin

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
    # submitted high-resolution makes worse
    ngridx = 40
    ngridy = 20
  
    rrange = np.array([rsun-4.0, rsun+4.0])
    vzrange = np.array([-20.0, 20.0])
    vzticks = np.array([-10.0, 0.0, 10.0])
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

    if FileGauXD == True:
        # read Gaussian fitting results of XD
        for ivel in range(nvel):
            ngauss = ngauss_sam[ivel]
            filev = 'gaussxd'+str(isamp)+'vel'+str(ivel)+'.asc'
            # read the data
            rdatav = np.loadtxt(filev, comments='#')
            iline = 0
            for irad in range(nradgridxd):
                rr = rgridxd[irad]
                cirad = '{0:02d}'.format(int(rr*10))
                # print ' irad = ',cirad,rr
                indx = np.fabs(rgals_obs-rr)<rw
                # fit with mix of Gaussians
                if ivel == 0:
                    vals = vrots_obs[indx]
                else:
                    vals = vzs_obs[indx]

                # for plot
                gausig = np.zeros(ngauss)
                gaumean = np.zeros(ngauss)
                gauamp = np.zeros(ngauss)
                for ii in range(ngauss):
                    if isamp == 0:
                        gauxd_sig_FRVS[ivel,irad,ii] = rdatav[iline,4]
                        gauxd_amp_FRVS[ivel,irad,ii] = rdatav[iline,2]
                        gauxd_mean_FRVS[ivel,irad,ii] = rdatav[iline,3]
                        gauxd_rr_FRVS[irad] = rdatav[iline,1]
                    elif isamp == 1:
                        gauxd_sig_FF[ivel,irad,ii] = rdatav[iline,4]
                        gauxd_amp_FF[ivel,irad,ii] = rdatav[iline,2]
                        gauxd_mean_FF[ivel,irad,ii] = rdatav[iline,3]
                        gauxd_rr_FF[irad] = rdatav[iline,1]
                    gausig[ii] = rdatav[iline,4]
                    gaumean[ii] = rdatav[iline,3]
                    gauamp[ii] = rdatav[iline,2]
                    iline += 1
                if FigGauMod == True:
                    xs = np.linspace(-80.,80.,1001)
                    ys = np.zeros_like(xs)
                    for ii in range(ngauss):
                        ys += (gauamp[ii]/gausig[ii])\
                            *np.exp(-0.5*(xs-gaumean[ii])**2. \
                           /gausig[ii]**2)/np.sqrt(2.*np.pi)
                    _= bovy_plot.bovy_hist(vals,bins=41,range=[-80.,80.],
                       histtype='step',lw=2.,normed=True,overplot=True)
                    plt.plot(xs,ys)
                    for ii in range(ngauss):
                        ys = (gauamp[ii]/gausig[ii])\
                            *np.exp(-0.5*(xs-gaumean[ii])**2. \
                           /gausig[ii]**2)/np.sqrt(2.*np.pi)
                        plt.plot(xs,ys)
                    # print("Combined <v^2>, sqrt(<v^2>):",combined_sig2( \
                    #    initamp,initmean[:,0],initcovar[:,0,0]),
                    #    np.sqrt(combined_sig2(initamp,initmean[:,0],initcovar[:,0,0])))
                    plt.xlabel(r'$V\,(\mathrm{km\,s}^{-1})$')
                    filename = 'samp'+str(isamp)+'v'+str(ivel)+'r'+cirad+'.png'
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()

    # Vz median
    # median
    if isamp == 0:
        nrad = 30+1
        lowrad = rsun-3.0
        highrad =  rsun+3.0
    else:
        nrad = 40+1
        lowrad = rsun-4.0
        highrad =  rsun+4.0

    drad = (highrad-lowrad)/(nrad-1)
    vzmed_rr = np.zeros(nrad)
    rgrid_vz = np.zeros(nrad)
    rlow = lowrad
    rhigh = lowrad+drad
    rw = 0.2
    for ii in range(nrad):
        # sindx = np.where((rgals_obs>=rlow) & (rgals_obs<rhigh))
        rr = 0.5*(rlow+rhigh)
        sindx = np.fabs(rgals_obs-rr)<rw
        rgrid_vz[ii] = 0.5*(rlow+rhigh)
        vzmed_rr[ii] = np.median(vzs_obs[sindx])
        rlow += drad
        rhigh += drad
    # print ' r vzmed=',rgrid_vz,vzmed_rr
    if isamp == 0:
        vzmed_FRVS = np.copy(vzmed_rr)
        vzmed_rr_FRVS = np.copy(rgrid_vz)
    else:
        vzmed_FF = np.copy(vzmed_rr)
        vzmed_rr_FF = np.copy(rgrid_vz)
    # fileoutput
    filename = 'vzmed'+str(isamp)+'.asc'
    fv = open(filename, 'w')
    for ii in range(nrad):
        print >>fv, "%d %f %f" % (ii, \
            rgrid_vz[ii], vzmed_rr[ii])
    fv.close()

# spiral arm positoin at l=0 or l=180 from Reid et al. (2014)
rsunr14 = 8.34
nsparm = 5
rsparm = np.zeros(nsparm)
for ii in range(nsparm):
    if ii == 0:
        # Sctum arm
        tanpa = np.tan(19.8*np.pi/180.0)
        angref = 27.6*np.pi/180.0
        rref = 5.0
    elif ii == 1:
        # Sagittarius arm
        tanpa = np.tan(6.9*np.pi/180.0)
        angref = 25.6*np.pi/180.0
        rref = 6.6
    elif ii == 2:
        # Local arm
        tanpa = np.tan(12.8*np.pi/180.0)
        angref = 8.9*np.pi/180.0
        rref = 8.4
    elif ii == 3:
        # Perseus arm
        tanpa = np.tan(9.4*np.pi/180.0)
        angref = 14.2*np.pi/180.0
        rref = 9.9
    else:
        # Outer arm
        tanpa = np.tan(13.8*np.pi/180.0)
        angref = 18.6*np.pi/180.0
        rref = 13.0

    # adjust to the current rsun
    rsparm[ii] = np.exp(tanpa*angref)*rref-rsunr14+rsun
    if myrank == 0:
        print ii,' arm Rgal = ',rsparm[ii]


# Final plot
if myrank == 0:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams["font.size"] = 16

    # combined plot for R vs Vrot
    ngauss = ngauss_sam[0]
    # colour mapscale
    # used in the paper
    # cmin = 0.0
    # cmax = 0.07
    #
    # for RVS
    cmin = -0.005
    cmax = 0.0275
    gauamplim=0.2
    gausiglim=50.0
    f, (ax1, ax2) = plt.subplots(2, sharex = True, figsize=(10,8))
    # submitted version
    # labpos = np.array([4.5, 40.0])
    labpos = np.array([5.0, 40.0])
    # ax1.imshow(HFRVS_RVrot, interpolation='nearest', origin='lower', \
    im1 = ax1.imshow(HFRVS_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFRVS_RVrot_xedges[0], HFRVS_RVrot_xedges[-1], \
                   HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]], \
           cmap=cm.jet)
    ax1.set_xlim(HFRVS_RVrot_xedges[0], HFRVS_RVrot_xedges[-1])
    ax1.set_ylim(HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1])
    # spiral arm position
    ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
    for ii in range(nsparm):
        xsp = np.array([rsparm[ii], rsparm[ii]])
        if FigPaper == True:
            ax1.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')
        else: 
            if ii == 0 or ii == 3: 
                ax1.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')

    # Sun's position
    xsp = np.array([rsun,rsun])
    ax1.plot(xsp, ysp, linestyle='solid', linewidth=2, color='w')
    if np.logical_or(MCsample == True, FileGauXD == True):
        for ii in range(ngauss):
            sindx = np.where(gauxd_sig_FRVS[0,:,ii]<gausiglim)
            if ii == 0: 
                marker = 's'
            elif ii == 1:
                marker = '^'
            else:
                marker = 'o'
            # ax1.scatter(gauxd_rr_FRVS[sindx],gauxd_mean_FRVS[0,sindx,ii], \
            #   c='cyan', marker = marker)
    # draw line for features
    # Hercules gap 
    # ax1.scatter(7.6,0.0)
    # F1 - Hercules low, -28 (x-6.8666)
    xsp1 = np.array([6.1523, 8.652])
    ysp1 = np.array([20.0,   -50])
    ax1.plot(xsp1,ysp1,color='w')
    if FigPaper == True:
        ax1.text(xsp1[1], ysp1[1]+3.0, r'Hercules', fontsize=16, color='w')
    # F2 - -28 (x-7.25)
    xsp2 = np.array([6.536, 8.679])
    ysp2 = np.array([20.0, -40.0])
    ax1.plot(xsp2,ysp2,color='w')
    # F3 -28 (x-7.9)
    xsp3 = np.array([7.186, 9.329])
    ysp3 = np.array([20.0, -40.0])
    ax1.plot(xsp3,ysp3,color='w')
    # F4 -28 (x-8.6)
    xsp4 = np.array([7.886, 10.029])
    ysp4 = np.array([20.0, -40.0])
    ax1.plot(xsp4,ysp4,color='w')
    # F5 -28 (x-9)
    xsp5 = np.array([8.286, 10.429])
    ysp5 = np.array([20.0, -40.0])
    ax1.plot(xsp5,ysp5,color='w')
    # F6  -10 (x-8.25)
    xsp6 = np.array([9.25, 10.75])
    ysp6 = np.array([-10.0, -25])
    ax1.plot(xsp6,ysp6,color='w')
    # F7  -8 (x-9.5)
    xsp7 = np.array([10.0, 11.375])
    ysp7 = np.array([-4.0, -15.0])
    ax1.plot(xsp7,ysp7,color='w')
    # F8  -28 (x-9.5)
    # xsp8 = np.array([8.071, 9.5])
    # ysp8 = np.array([40.0, 0.0])
    # ax1.plot(xsp8,ysp8,color='w')

    # H17  20 (H17) +11.0 (Vsun)
    if FigPaper == True:
        xsph17 = np.array([rsun+0.1, rsun+0.6])
        ysph17 = np.array([31.0, 31.0])
        ax1.plot(xsph17,ysph17,color='w')
        ax1.text(xsph17[1], ysph17[1], r'H17', fontsize=16, color='w')
    # 
        ax1.text(labpos[0], labpos[1], r'RVS', fontsize=16, color='w')
        ax1.set_ylabel(r"$V_{\rm rot}-V_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
        ax1.tick_params(labelsize=16, color='k')
        ax1.set_yticks(vrotticks)
    else:
        ax1.tick_params(labelsize=24, color='k')
        ax1.set_yticks(vrotticks)

    cmin = 0.0
    # cmax = 0.03
    cmax = 0.06
    # im = ax2.imshow(HFF_RVrot, interpolation='nearest', origin='lower', \
    im2 = ax2.imshow(HFF_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFF_RVrot_xedges[0], HFF_RVrot_xedges[-1], \
                   HFF_RVrot_yedges[0], HFF_RVrot_yedges[-1]], \
           cmap=cm.jet)
    ax2.set_xlim(HFF_RVrot_xedges[0], HFF_RVrot_xedges[-1])
    ax2.set_ylim(HFF_RVrot_yedges[0], HFF_RVrot_yedges[-1])
    # spiral arm position
    ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
    for ii in range(nsparm):
        xsp = np.array([rsparm[ii], rsparm[ii]])
        if FigPaper == True:
            ax2.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')
#        else: 
#            if ii == 0 or ii == 3: 
#                # ax2.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')
        if FigPaper == True:
            if ii == 0:
                ax2.text(xsp[1]+0.05, ysp[1]-10.0, r'Scutum', fontsize=16, color='w')
            if ii == 1:
                ax2.text(xsp[1]+0.05, ysp[1]-10.0, r'Sagittarius', fontsize=16, color='w')
            if ii == 2:
                ax2.text(xsp[1]+0.05, ysp[1]-10.0, r'Local', fontsize=16, color='w')
            if ii == 3:
                ax2.text(xsp[1]+0.05, ysp[1]-10.0, r'Perseus', fontsize=16, color='w')
            if ii == 4:
                ax2.text(xsp[1]+0.05, ysp[1]-10.0, r'Outer', fontsize=16, color='w')
    # Sun's position
    xsp = np.array([rsun,rsun])
    ax2.plot(xsp, ysp, linestyle='solid', linewidth=2, color='w')
    if np.logical_or(MCsample == True, FileGauXD == True):
        for ii in range(ngauss):
            sindx = np.where(gauxd_sig_FF[0,:,ii]<gausiglim)
            if ii == 0: 
                marker = 's'
            elif ii == 1:
                marker = '^'
            else:
                marker = 'o'
            # ax2.scatter(gauxd_rr_FF[sindx],gauxd_mean_FF[0,sindx,ii], \
            #  c='cyan', marker = marker)
            #        c='w', s = 100*gauxd_amp_FF[0,sindx,ii], marker = marker)
    if FigPaper == True:
        ax2.text(labpos[0], labpos[1], r'All', fontsize=16, color='w')
    # draw line for features
    # F1
    ax2.plot(xsp1,ysp1,color='w')
    if FigPaper == True:
        ax2.text(xsp1[0], ysp1[0], r'F1', fontsize=16, color='w')
    # F2
    ax2.plot(xsp2,ysp2,color='w')
    if FigPaper == True:
        ax2.text(xsp2[0], ysp2[0], r'F2', fontsize=16, color='w')
    # F3
    ax2.plot(xsp3,ysp3,color='w')
    if FigPaper == True:
        ax2.text(xsp3[0], ysp3[0], r'F3', fontsize=16, color='w')
    # F4
    ax2.plot(xsp4,ysp4,color='w')
    if FigPaper == True:
        ax2.text(xsp4[0], ysp4[0], r'F4', fontsize=16, color='w')
    # F5
    ax2.plot(xsp5,ysp5,color='w')
    if FigPaper == True:
        ax2.text(xsp5[0], ysp5[0], r'F5', fontsize=16, color='w')
    # F6
    ax2.plot(xsp6,ysp6,color='w')
    if FigPaper == True:
        ax2.text(xsp6[1], ysp6[1], r'F6', fontsize=16, color='w')
    # F7
    ax2.plot(xsp7,ysp7,color='w')
    if FigPaper == True:
        ax2.text(xsp7[1], ysp7[1], r'F7', fontsize=16, color='w')
    # F8
    # ax2.plot(xsp8,ysp8,color='w')
    # ax2.text(xsp8[1], ysp8[1], r'F8', fontsize=16, color='w')
    # H17
    if FigPaper == True:
        ax2.plot(xsph17,ysph17,color='w')

    #
    if FigPaper == True:
        ax2.set_ylabel(r"$V_{\rm rot}-V_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
        ax2.tick_params(labelsize=16, color='k')
        ax2.set_yticks(vrotticks)

        plt.xlabel(r"$R_{\rm gal}$ (kpc)", fontsize=18)
        plt.ylabel(r"$V_{\rm rot}-V_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
    else:
        ax2.tick_params(labelsize=24, color='k')
        ax2.set_yticks(vrotticks)

    if FigPaper == True:
        f.subplots_adjust(hspace=0.0, right = 0.8)
        cbar_ax1 = f.add_axes([0.8, 0.5, 0.05, 0.375])
        cb1 = f.colorbar(im1, cax=cbar_ax1)
        cb1.ax.tick_params(labelsize=16)
        cbar_ax2 = f.add_axes([0.8, 0.1, 0.05, 0.375])
        cb2 = f.colorbar(im2, cax=cbar_ax2)
        cb2.ax.tick_params(labelsize=16)
    else:
        f.subplots_adjust(hspace=0.0, right = 0.95)
    # plt.show()
    if FigPaper == True:
        plt.savefig('RVrot.eps')
    else:
        plt.savefig('RVrot.jpg')
    plt.close(f)

    # R vs. Vz
    ngauss = ngauss_sam[1]
    cmin = 0.0
    cmax = 0.1
    labpos = np.array([4.5, 16.0])
    f, (ax1, ax2) = plt.subplots(2, sharex = True, figsize=(10,8))
    # RVS
    ax1.imshow(HFRVS_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFRVS_RVz_xedges[0], HFRVS_RVz_xedges[-1], \
                   HFRVS_RVz_yedges[0], HFRVS_RVz_yedges[-1]], cmap=cm.jet)
    ax1.set_xlim(HFRVS_RVz_xedges[0], HFRVS_RVz_xedges[-1])
    ax1.set_ylim(HFRVS_RVz_yedges[0], HFRVS_RVz_yedges[-1])
    # spiral arm position
    ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
    for ii in range(nsparm):
        xsp = np.array([rsparm[ii], rsparm[ii]])
        ax1.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')
    # Sun's position
    xsp = np.array([rsun,rsun])
    ax1.plot(xsp, ysp, linestyle='solid', linewidth=2, color='w')
    if np.logical_or(MCsample == True, FileGauXD == True):
        for ii in range(0,1):
            sindx = np.where(gauxd_amp_FRVS[1,:,ii]>gauamplim)
            if ii == 0: 
                marker = 's'
            elif ii == 1:
                marker = '^'
            else:
                marker = 'o'
            ax1.scatter(gauxd_rr_FRVS[sindx],gauxd_mean_FRVS[1,sindx,ii], \
                        c='w', marker = 'o')

    # median
    # ax1.scatter(vzmed_rr_FRVS,vzmed_FRVS, c='w', marker = 'o')

    ax1.text(labpos[0], labpos[1], r'RVS', fontsize=16, color='w')
    ax1.set_ylabel(r"$V_{\rm z}$ (km s$^{-1}$)", fontsize=18)
    ax1.tick_params(labelsize=16, color='k')
    ax1.set_yticks(vzticks)

    im = ax2.imshow(HFF_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFF_RVz_xedges[0], HFF_RVz_xedges[-1], \
                   HFF_RVz_yedges[0], HFF_RVz_yedges[-1]], cmap=cm.jet)
    ax2.set_xlim(HFF_RVz_xedges[0], HFF_RVz_xedges[-1])
    ax2.set_ylim(HFF_RVz_yedges[0], HFF_RVz_yedges[-1])
    # spiral arm position
    ysp = np.array([HFRVS_RVz_yedges[0], HFRVS_RVz_yedges[-1]])
    for ii in range(nsparm):
        xsp = np.array([rsparm[ii], rsparm[ii]])
        ax2.plot(xsp, ysp, linestyle='dashed', linewidth=2, color='w')
        if ii == 0:
            ax2.text(xsp[1]+0.05, ysp[1]-4.0, r'Scutum', fontsize=16, color='w')
        if ii == 1:
            ax2.text(xsp[1]+0.05, ysp[1]-4.0, r'Sagittarius', fontsize=16, color='w')
        if ii == 2:
            ax2.text(xsp[1]+0.05, ysp[1]-4.0, r'Local', fontsize=16, color='w')
        if ii == 3:
            ax2.text(xsp[1]+0.05, ysp[1]-4.0, r'Perseus', fontsize=16, color='w')
        if ii == 4:
            ax2.text(xsp[1]+0.05, ysp[1]-4.0, r'Outer', fontsize=16, color='w')
    # Sun's position
    xsp = np.array([rsun,rsun])
    ax2.plot(xsp, ysp, linestyle='solid', linewidth=2, color='w')
    if np.logical_or(MCsample == True, FileGauXD == True):
        for ii in range(0,1):
            sindx = np.where(gauxd_amp_FF[1,:,ii]>gauamplim)
            if ii == 0: 
                marker = 's'
            elif ii == 1:
                marker = '^'
            else:
                marker = 'o'
            ax2.scatter(gauxd_rr_FF[sindx],gauxd_mean_FF[1,sindx,ii], \
                        c='w', marker = 'o')
    # median
    # ax2.scatter(vzmed_rr_FF,vzmed_FF, c='w', marker = 'o')

    ax2.text(labpos[0], labpos[1], r'All', fontsize=16, color='w')
    ax2.set_ylabel(r"$V_{\rm z}$ (km s$^{-1}$)", fontsize=18)
    ax2.tick_params(labelsize=16, color='k')
    ax2.set_yticks(vzticks)

    plt.xlabel(r"$R_{\rm gal}$ (kpc)", fontsize=18)
    plt.ylabel(r"$V_{\rm z}$ (km s$^{-1}$)", fontsize=18)
    f.subplots_adjust(hspace=0.0, right = 0.8)
    cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
    cb = f.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=16)
    # plt.show()
    plt.savefig('RVz.eps')
    plt.close(f)

