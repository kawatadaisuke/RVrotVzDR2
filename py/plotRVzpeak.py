
#
# plotRVzpeak.py
#
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

# for not displaying
matplotlib.use('Agg')

# fitting function
def func(x, a, b, c, d, e):
    return a*(x-8.2)+b+c*np.sin((x-8.2)*2.0*np.pi/d+e)
    # return a*x+b+c*np.sin(x/d+e)
def func2s(x, a, b, c, d, e, f, g, h):
    return a*x+b+c*np.sin(x*2.0*np.pi/d+e)+f*np.sin(x*2.0*np.pi/g+h)

def funclin(x, a, b):
    return a*x+b

##### main programme start here #####

# True: add fake sin curve
FlagFake = False

# epoch
epoch = 2000.0
# constant for proper motion unit conversion
pmvconst = 4.74047
usun = 11.1
vsun = 239.08
wsun = 7.25
# from Bland-Hawthorn & Gerhard (2016)
rsun = 8.2
zsun = 0.025

# initial fitting parameters
# Schoenrich & Dehnen
ain = 0.56644
bin = -1.381
cin = 0.817
din = 2.0966
ein = -1.74

# fitting rlimit
rlow = rsun-3.0
rhigh = rsun+3.0

nsample = 2
for isamp in range(nsample):
    # number of Gaussian
    ngauss = 3
    # nvel = 2, 0: Vrot, 1: Vrot
    nvel = 2
    # set range
    rw = 0.2
    if isamp == 0:
        rrangexd = np.array([rsun-3.0, rsun+3.0])
        nradgridxd = 30+1  
        # nradgridxd = 3+1  
    elif isamp == 1: 
        rrangexd = np.array([rsun-4.0, rsun+4.0])
        nradgridxd = 40+1
        # nradgridxd = 5+1  
    else:
        rrangexd = np.array([rsun-3.0, rsun+3.0])
        nradgridxd = 30+1
        # nradgridxd = 7+1
        rgridxd = np.linspace(rrangexd[0], rrangexd[1], nradgridxd)
        # print ' rgridxd = ',rgridxd

    # read Gaussian fitting results of XD
    for ivel in range(nvel):
        if ivel == 0: 
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

             if FlagFake == True:
                 # fake data for the highest amplitude data
                 # parameters
                 if isamp == 0:
                     gauxd_mean_FRVS[ivel,irad,0] = func(gauxd_rr_FRVS[irad], \
                         ain,bin,cin,din,ein)
                 elif isamp == 1:
                     gauxd_mean_FF[ivel,irad,0] = func(gauxd_rr_FF[irad],
                         ain,bin,cin,din,ein)
                 else:
                     gauxd_mean_A[ivel,irad,0] = func(gauxd_rr_A[irad], \
                         ain,bin,0.8*cin,din,ein)

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

# Final plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16
# combined plot for R vs Vrot
# colour mapscale
cmin = 0.0
cmax = 0.1
gauamplim=0.2
# plt.figure(figsize=(7,4))
# spiral arm position
xprange=np.array([rsun-4.0,rsun+4.0])
yprange=np.array([-5.0,5.0])
plt.axis([xprange[0],xprange[1],yprange[0],yprange[1]])
ysp = np.array([yprange[0],yprange[1]])
for ii in range(nsparm):
    xsp = np.array([rsparm[ii], rsparm[ii]])
    plt.plot(xsp, ysp, linestyle='dashed', linewidth=3, color='k')
# Sun's position
xsp = np.array([rsun,rsun])
plt.plot(xsp, ysp, linestyle='solid', linewidth=2, color='k')
# plot only the highest amplitude peak
# plt.text(labpos[0], labpos[1], r'F w/RVS', fontsize=16, color='w')
# Fw/RVS
plt.scatter(gauxd_rr_FRVS,gauxd_mean_FRVS[1,:,0], \
            c='r', marker = 's', s=40)
# FF
plt.scatter(gauxd_rr_FF,gauxd_mean_FF[1,:,0], \
            c='k', marker = '^', s=40)
# fitting with func
poptini = np.array([ain, bin, cin, din, ein])
# poptini = np.array([ain, bin])
sindx = np.where((gauxd_rr_FF>rlow) & (gauxd_rr_FF<rhigh))
rvals = gauxd_rr_FF[sindx]
vzvals = gauxd_mean_FF[1,sindx,0].flatten()
# params, pcov = optimize.curve_fit(func,rvals,vzvals,p0=poptini)
params, pcov = optimize.curve_fit(func2s,rvals,vzvals)
# plot fitted data
xsp = np.linspace(rlow,rhigh,100)
print params
prams = poptini
ysp = func(xsp,params[0],params[1],params[2],params[3],params[4])
# ysp = func2s(xsp,params[0],params[1],params[2],params[3],params[4], \
#    params[5],params[6],params[7])
# ysp = funclin(xsp,params[0],params[1])
# plt.plot(xsp,ysp,color='k')
# A star
# plt.scatter(gauxd_rr_A,gauxd_mean_A[1,:,0], \
#            c='b', marker = 'o', s=40)
# fitting with func
# poptini = np.array([ain, bin, 1.5, 2.2, ein])
# params, pcov = optimize.curve_fit(func,gauxd_rr_A,gauxd_mean_A[1,:,0],
#    p0=poptini)
# plot fitted data
# xsp = np.linspace(xprange[0],xprange[1],100)
# print params
# ysp = func(xsp,params[0],params[1],params[2],params[3],params[4])
# plt.plot(xsp,ysp,color='b')
# lables
plt.ylabel(r"$V_{\rm z}$ (km s$^{-1}$)", fontsize=18)
plt.xlabel(r"$R_{\rm gal}$ (kpc)", fontsize=18)
plt.tick_params(labelsize=16, color='k')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('RVzpeak.eps')
plt.close()
