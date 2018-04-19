
#
# RVrotVz_FRVS.py
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
from scipy import stats
from scipy import optimize
import scipy.interpolate
from galpy.util import bovy_coords, bovy_plot
from extreme_deconvolution import extreme_deconvolution

# for not displaying
# matplotlib.use('Agg')

##### main programme start here #####

# flags
# True: MC sampling on
MCsample = True
# True: read gaussxd*.asc
FileGauXD = False

if FileGauXD == True:
    MCsample = False

# number of MC sampling
nmc = 100
# epoch
epoch = 2000.0
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
# minimum plx
plxlim=0.001

if MCsample == True:
   print ' MCsample is on. Nmc = ',nmc

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

    if isamp == 0:
        print ' for bright F stars with RVS'  
        Tefflow = 6600.0
        Teffhigh = 6900.0
    elif isamp == 1: 
        print ' for faint F stars'  
        Tefflow = 6600.0
        Teffhigh = 7330.0
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
    # error correalation
    plxpmra_corrs = np.zeros_like(e_plxs) 
    plxpmdec_corrs = np.zeros_like(e_plxs) 
    pmradec_corrs = np.zeros_like(e_plxs) 

    # age [Fe/H] only for Galaxia
    fehs_true = star['[Fe/H]_true'][sindx]
    ages_true = star['Age'][sindx]

    # convert deg -> rad
    glonrads = glons*np.pi/180.0
    glatrads = glats*np.pi/180.0

    # get observed position and velocity
    dists_obs = 1.0/plxs_obs

    # velocity
    Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
        pmras_obs, pmdecs_obs, ras, \
        decs, degree=True, epoch=epoch)
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

        plxpmradec_mc = np.empty((nstars, 3, nmc))
        plxpmradec_mc[:, 0, :] = np.atleast_2d(plxs_obs).T
        plxpmradec_mc[:, 1, :] = np.atleast_2d(pmras_obs).T
        plxpmradec_mc[:, 2, :] = np.atleast_2d(pmdecs_obs).T
        for ii in range(nstars):
            # constract covariance matrix
            tcov = np.zeros((3, 3))
            # /2 because of symmetrization below
            tcov[0, 0] = e_plxs[ii]**2.0 / 2.0
            tcov[1, 1] = e_pmras[ii]**2.0 / 2.0
            tcov[2, 2] = e_pmdecs[ii]**2.0 / 2.0
            tcov[0, 1] = plxpmra_corrs[ii] * e_plxs[ii] * e_pmras[ii]
            tcov[0, 2] = plxpmdec_corrs[ii] * e_plxs[ii] * e_pmdecs[ii]
            tcov[1, 2] = pmradec_corrs[ii] * e_pmras[ii] * e_pmdecs[ii]
            # symmetrise
            tcov = (tcov + tcov.T)
            # Cholesky decomp.
            L = np.linalg.cholesky(tcov)
            plxpmradec_mc[ii] += np.dot(L, np.random.normal(size=(3, nmc)))

        # distribution of velocity and distance.
        # -> pml pmb
        ratile = np.tile(ras, (nmc, 1)).flatten()
        dectile = np.tile(decs, (nmc, 1)).flatten()
        pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb( \
            plxpmradec_mc[:, 1, :].T.flatten(), \
            plxpmradec_mc[:, 2, :].T.flatten(), \
            ratile, dectile, degree=True, epoch=epoch)
        # reshape
        pmllbb_sam = pmllbb_sam.reshape((nmc, nstars, 2))
        # distance MC sampling
        plxs_sam = plxpmradec_mc[:, 0, :].T
        # check negative parallax
        plxs_samflat= plxs_sam.flatten()
        copysamflat=np.copy(plxs_samflat)
        if len(copysamflat[plxs_samflat<plxlim])>0: 
            print len(copysamflat[plxs_samflat<plxlim]),' plx set to ',plxlim
        plxs_samflat[copysamflat<plxlim]=plxlim
        plxs_sam = np.reshape(plxs_samflat,(nmc,nstars))
        # distance
        dists_sam = 1.0/plxs_sam
        # mas/yr -> km/s
        vlons_sam = pmvconst*pmllbb_sam[:,:,0]*dists_sam
        vlats_sam = pmvconst*pmllbb_sam[:,:,1]*dists_sam
        # galactic position
        distxys_sam = dists_sam*np.cos(glatrads)
        xpos_sam = distxys_sam*np.cos(glonrads)
        ypos_sam = distxys_sam*np.sin(glonrads)
        zpos_sam = dists_sam*np.sin(glatrads)
        rgals_sam = np.sqrt((xpos_sam-rsun)**2+ypos_sam**2)

        if isamp == 0:
            hrvs_sam = np.random.normal(hrvs_obs, e_hrvs, (nmc, nstars))
            vxvyvz_sam = bovy_coords.vrpmllpmbb_to_vxvyvz( \
                hrvs_sam.flatten(), pmllbb_sam[:,:,0].flatten(), \
                pmllbb_sam[:,:,1].flatten(), \
                np.tile(glons, (nmc, 1)).flatten(), \
                np.tile(glats, (nmc, 1)).flatten(), \
                dists_sam.flatten(), degree=True)
            vxvyvz_sam = vxvyvz_sam.reshape((nmc, nstars, 3))
            vxs_sam = vxvyvz_sam[:,:,0]
            vys_sam = vxvyvz_sam[:,:,1]
            vzs_sam = vxvyvz_sam[:,:,2]+wsun
            # 2D velocity
            hrvxys_sam = hrvs_sam*np.cos(glatrads)
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
            vrots_sam[:,np.logical_or(glons<90, glons>270)] = \
                vrots_sam[:,np.logical_or(glons<90, glons>270)]+vsun
            vrots_sam[:,np.logical_and(glons>=90, glons<=270)] = \
                -vrots_sam[:,np.logical_and(glons>=90, glons<=270)]+vsun
            vrads_sam = np.zeros_like(vrots_sam)
            vzs_sam = np.copy(vlats_sam)+wsun
        # error estimats dispersion (use observed one for mean value)
        # rgals_obs = np.mean(rgals_sam, axis=0).reshape(nstars)
        e_rgals = np.std(rgals_sam, axis=0).reshape(nstars)
        # vzs_obs = np.mean(vzs_sam, axis=0).reshape(nstars)
        e_vzs = np.std(vzs_sam, axis=0).reshape(nstars)
        # vrots_obs = np.mean(vrots_sam, axis=0).reshape(nstars)
        e_vrots = np.std(vrots_sam, axis=0).reshape(nstars)
        # position
        # xpos_obs = np.mean(xpos_sam, axis=0).reshape(nstars)
        # ypos_obs = np.mean(ypos_sam, axis=0).reshape(nstars)
        # zpos_obs = np.mean(zpos_sam, axis=0).reshape(nstars)
        # dists_obs = np.mean(dists_sam, axis=0).reshape(nstars)
        # vrads_obs = np.mean(vrads_sam, axis=0).reshape(nstars)


    # Vrot defined w.r.t. solar velocity
    vrots_obs -= vcircsun
    if MCsample == True:
        vrots_sam -= vcircsun

    if isamp == 0:
        f=open('star_RVrotVz_FRVS.asc','w')
    elif isamp == 1:
        f=open('star_RVrotVz_FF.asc','w')
    else:
        f=open('star_RVrotVz_A.asc','w')
    for i in range(nstars):
        print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
          %(xpos_obs[i], ypos_obs[i], zpos_obs[i], rgals_obs[i], vrots_obs[i], \
            vrads_obs[i], vzs_obs[i], dists_obs[i], glons[i], glats[i], \
            fehs_true[i], ages_true[i], e_rgals[i], e_vrots[i], e_vzs[i])
    f.close()

    # output velocity dispersion of the sample
    print ' velocity dispersion Vrot, Vz = ', np.std(vrots_obs), np.std(vzs_obs)

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
        # number of Gaussian
        ngauss = 3
        # nvel = 2, 0: Vrot, 1: Vrot
        nvel = 2
        # set range
        rw = 0.5
        if isamp == 0:
            rrangexd = np.array([rsun-0.7, rsun+0.7])
            nradgridxd = 7+1  
            # nradgridxd = 3+1  
        elif isamp == 1: 
            rrangexd = np.array([rsun-2.0, rsun+2.0])
            nradgridxd = 20+1
            # nradgridxd = 5+1  
        else:
            rrangexd = np.array([rsun-3.0, rsun+3.0])
            nradgridxd = 30+1
            # nradgridxd = 7+1
        rgridxd = np.linspace(rrangexd[0], rrangexd[1], nradgridxd)
        print ' rgridxd = ',rgridxd

        for ivel in range(nvel):
            if ivel == 0: 
                print ngauss,'Gaussian Mixture model for Vrot'
                # Vrot
                if isamp == 0:
                    gauxd_amp_FRVS = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_mean_FRVS = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_rr_FRVS = np.zeros((nradgridxd))
                elif isamp == 1:
                    gauxd_amp_FF = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_mean_FF = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_rr_FF = np.zeros((nradgridxd))
                else:
                    gauxd_amp_A = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_mean_A = np.zeros((nvel,nradgridxd,ngauss))
                    gauxd_rr_A = np.zeros((nradgridxd))
            else:
                print ngauss,'Gaussian Mixture model for Vz'

    if MCsample == True:
        for ivel in range(nvel):
            # fit with Mixture of Gaussian using XD

            # output amg, mean dispersion
            filename = 'gaussxd'+str(isamp)+'vel'+str(ivel)+'.asc'
            fr = open(filename, 'w')
            filename = 'gaussxd'+str(isamp)+'vel'+str(ivel)+'.asc'
            fz = open(filename, 'w')

            print >>fr, "# ngauss = %d, nrgrid = %d" % (ngauss, nradgridxd)
            print >>fz, "# ngauss = %d, nrgrid = %d" % (ngauss, nradgridxd)

            for irad,rr in enumerate(rgridxd):
                cirad = '{0:02d}'.format(int(rr*10))
                # print ' irad = ',cirad
                indx = np.fabs(rgals_obs-rr)<rw
                # fit with mix of Gaussians
                # for Vrot
                print ' rr, nsamp =',rr, len(vrots_obs[indx])
                vals = vrots_obs[indx]
                e_vals = e_vrots[indx]
                ydata = np.atleast_2d(vals).T
                ycovar = e_vals**2
                # print ' input shape =',ydata.shape, ycovar
                initamp = np.random.uniform(size=ngauss)
                initamp /= np.sum(initamp)
                m = np.median(vals)
                s= 1.4826*np.median(np.fabs(vals-m))
                print ' initial guess of median, sig=',m,s
                initmean= []
                initcovar= []
                for ii in range(ngauss):
                    initcovar.append(s**2.)
                initcovar= np.array([[initcovar]]).T
                # Now let the means vary
                for ii in range(ngauss):
                    initmean.append(m+np.random.normal()*s)
                initmean= np.array([initmean]).T
                print("lnL",extreme_deconvolution(ydata,ycovar, \
                    initamp,initmean,initcovar))
                print("amp, mean, std. dev.",initamp,initmean[:,0], \
                    np.sqrt(initcovar[:,0,0]))
                # store the amp and mean
                # sort with amplitude
                sortindx = np.argsort(initamp)
                sortindx = sortindx[::-1]
                if ivel == 0:
                    for ii in range(ngauss):
                        ist = sortindx[ii]
                        print >>fr, "%d %f %f %f %f" % (ii, rr, \
                            initamp[ist], initmean[ist,0], \
                            np.sqrt(initcovar[ist,0,0]))
                else:
                    for ii in range(ngauss):
                        ist = sortindx[ii]
                        print >>fz, "%d %f %f %f %f" % (ii, rr, \
                            initamp[ist], initmean[ist,0], \
                            np.sqrt(initcovar[ist,0,0]))
                # print ' sorted amp, mean = ', initamp[sortindx], \
                #    initmean[sortindx,0]
                if isamp == 0:
                    gauxd_amp_FRVS[ivel,irad,:] = initamp[sortindx]
                    gauxd_mean_FRVS[ivel,irad,:] = initmean[sortindx,0]
                    gauxd_rr_FRVS[irad] = rr
                elif isamp == 1:
                    gauxd_amp_FF[ivel,irad,:] = initamp[sortindx]
                    gauxd_mean_FF[ivel,irad,:] = initmean[sortindx,0]
                    gauxd_rr_FF[irad] = rr
                else:
                    gauxd_amp_A[ivel,irad,:] = initamp[sortindx]
                    gauxd_mean_A[ivel,irad,:] = initmean[sortindx,0]
                    gauxd_rr_A[irad] = rr
                
                # for plot
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
                filename = 'samp'+str(isamp)+'vdisp'+cirad+'.png'
                plt.savefig(filename)
                plt.clf()
                plt.close()

            fr.close()
            fz.close()

    # minimum number of stars in each column
    nsmin = 25
    # set number of grid
    ngridx = 40
    ngridy = 40
    # grid plot for R vs. Vrot
    rrange = np.array([rsun-4.0, rsun+4.0])
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
            filev = 'gaussxd'+str(isamp)+'vel'+str(ivel)+'.asc'
            # read the data
            rdatav = np.loadtxt(filev, comments='#')
            iline = 0
            for irad in range(nradgridxd):
                for ii in range(ngauss):
                    if isamp == 0:
                        gauxd_amp_FRVS[ivel,irad,ii] = rdatav[iline,2]
                        gauxd_mean_FRVS[ivel,irad,ii] = rdatav[iline,3]
                        gauxd_rr_FRVS[irad] = rdatav[iline,1]
                    elif isamp == 1:
                        gauxd_amp_FF[ivel,irad,ii] = rdatav[iline,2]
                        gauxd_mean_FF[ivel,irad,ii] = rdatav[iline,3]
                        gauxd_rr_FF[irad] = rdatav[iline,1]
                    else:
                        gauxd_amp_A[ivel,irad,ii] = rdatav[iline,2]
                        gauxd_mean_A[ivel,irad,ii] = rdatav[iline,3]
                        gauxd_rr_A[irad] = rdatav[iline,1]
                    iline += 1

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
    print ii,' arm Rgal = ',rsparm[ii]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16
# combined plot for R vs Vrot
# colour mapscale
cmin = 0.0
cmax = 0.1
gauamplim=0.1
f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(8,8))
labpos = np.array([4.2, 40.0])
ax1.imshow(HFRVS_RVrot, interpolation='gaussian', origin='lower', \
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
    ax1.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_FRVS[0,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax1.scatter(gauxd_rr_FRVS[sindx],gauxd_mean_FRVS[0,sindx,ii], \
                c='cyan', marker = marker)
        #        c='w', s = 100*gauxd_amp_FRVS[0,sindx,ii], marker = marker)
ax1.text(labpos[0], labpos[1], r'F w/RVS', fontsize=16, color='w')
ax1.set_ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
ax1.tick_params(labelsize=16, color='k')
ax1.set_yticks(vrotticks)
ax2.imshow(HFF_RVrot, interpolation='gaussian', origin='lower', \
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
    ax2.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_FF[0,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax2.scatter(gauxd_rr_FF[sindx],gauxd_mean_FF[0,sindx,ii], \
                c='cyan', marker = marker)
        #        c='w', s = 100*gauxd_amp_FF[0,sindx,ii], marker = marker)
ax2.text(labpos[0], labpos[1], r'F', fontsize=16, color='w')
ax2.set_ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
ax2.tick_params(labelsize=16, color='k')
ax2.set_yticks(vrotticks)
im = ax3.imshow(HA_RVrot, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HA_RVrot_xedges[0], HA_RVrot_xedges[-1], \
                   HA_RVrot_yedges[0], HA_RVrot_yedges[-1]], \
           cmap=cm.jet)
ax3.set_xlim(HA_RVrot_xedges[0], HA_RVrot_xedges[-1])
ax3.set_ylim(HA_RVrot_yedges[0], HA_RVrot_yedges[-1])
# spiral arm position
ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
for ii in range(nsparm):
    xsp = np.array([rsparm[ii], rsparm[ii]])
    ax3.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_A[0,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax3.scatter(gauxd_rr_A[sindx],gauxd_mean_A[0,sindx,ii], \
               c='cyan', marker = marker)
        #       c='w', s = 100*gauxd_amp_A[0,sindx,ii], marker = marker)
ax3.text(labpos[0], labpos[1], r'A', fontsize=16, color='w')
ax3.tick_params(labelsize=16, color='k')
ax3.set_yticks(vrotticks)
plt.xlabel(r"R$_{\rm gal}$ (kpc)", fontsize=18)
plt.ylabel(r"V$_{\rm rot}$$-$V$_{\rm LSR}$ (km s$^{-1}$)", fontsize=18)
f.subplots_adjust(hspace=0.0, right = 0.8)
cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
cb = f.colorbar(im, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
plt.show()
plt.close(f)

# R vs. Vz
labpos = np.array([4.2, 16.0])
f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(8,8))
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
    ax1.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_FRVS[1,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax1.scatter(gauxd_rr_FRVS[sindx],gauxd_mean_FRVS[1,sindx,ii], \
                c='cyan', marker = marker)
ax1.text(labpos[0], labpos[1], r'F w/RVS', fontsize=16, color='w')
ax1.set_ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
ax1.tick_params(labelsize=16, color='k')
ax1.set_yticks(vzticks)
ax2.imshow(HFF_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HFF_RVz_xedges[0], HFF_RVz_xedges[-1], \
                   HFF_RVz_yedges[0], HFF_RVz_yedges[-1]], cmap=cm.jet)
ax2.set_xlim(HFF_RVz_xedges[0], HFF_RVz_xedges[-1])
ax2.set_ylim(HFF_RVz_yedges[0], HFF_RVz_yedges[-1])
# spiral arm position
ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
for ii in range(nsparm):
    xsp = np.array([rsparm[ii], rsparm[ii]])
    ax2.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_FF[1,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax2.scatter(gauxd_rr_FF[sindx],gauxd_mean_FF[1,sindx,ii], \
                c='cyan', marker = marker)
ax2.text(labpos[0], labpos[1], r'F', fontsize=16, color='w')
ax2.set_ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
ax2.tick_params(labelsize=16, color='k')
ax2.set_yticks(vzticks)
im = ax3.imshow(HA_RVz, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[HA_RVz_xedges[0], HA_RVz_xedges[-1], \
                   HA_RVz_yedges[0], HA_RVz_yedges[-1]], cmap=cm.jet)
ax3.set_xlim(HA_RVz_xedges[0], HA_RVz_xedges[-1])
ax3.set_ylim(HA_RVz_yedges[0], HA_RVz_yedges[-1])
# spiral arm position
ysp = np.array([HFRVS_RVrot_yedges[0], HFRVS_RVrot_yedges[-1]])
for ii in range(nsparm):
    xsp = np.array([rsparm[ii], rsparm[ii]])
    ax3.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='w')
if np.logical_or(MCsample == True, FileGauXD == True):
    for ii in range(ngauss):
        sindx = np.where(gauxd_amp_A[1,:,ii]>gauamplim)
        if ii == 0: 
            marker = 's'
        elif ii == 1:
            marker = '^'
        else:
            marker = 'o'
        ax3.scatter(gauxd_rr_A[sindx],gauxd_mean_A[1,sindx,ii], \
                c='cyan', marker = marker)
ax3.text(labpos[0], labpos[1], r'A', fontsize=16, color='w')
ax3.tick_params(labelsize=16, color='k')
ax3.set_yticks(vzticks)
plt.xlabel(r"R$_{\rm z}$ (kpc)", fontsize=18)
plt.ylabel(r"V$_{\rm z}$ (km s$^{-1}$)", fontsize=18)
f.subplots_adjust(hspace=0.0, right = 0.8)
cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
cb = f.colorbar(im, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
plt.show()
plt.close(f)




