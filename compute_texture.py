import numpy as np
from scipy import ndimage
import cd_measures_pack as CDMP
from scipy import stats
from scipy import signal
from scipy import interpolate
from skimage import color
from skimage import draw
from skimage import feature
import cython_functions
import pywt
import cv2
import my_utilities as MU
import color_spaces as CAM


def ar2d(X,r=1.5):
    N, _, Scale = MU.number_neighbour(r)
    Y = MU.checkifRGB(X)
    Y = (Y-np.mean(Y))/np.std(Y)
    xx = np.arange(0, X.shape[1])
    yy = np.arange(0, X.shape[0])
    fY = interpolate.interp2d(xx, yy, Y, kind='linear')
    flag = False
    for ii in range(N):
        xi = xx - r * np.sin(2 * np.pi * ii / N)
        yi = yy + r * np.cos(2 * np.pi * ii / N)
        if flag:
            X_ii = np.hstack((X_ii,np.reshape(fY(xi, yi),(Y.size,1))))
        else:
            X_ii = np.reshape(fY(xi, yi), (Y.size, 1))
            flag = True
    c, _, _, _ = np.linalg.lstsq(X_ii, np.reshape(Y,(Y.size,1)))
    Yar = np.reshape(np.dot(X_ii,c),(Y.shape[0],Y.shape[1]))
    return Yar, c


def autocorr2d(X):
    Y = MU.checkifRGB(X)
    Rho = np.fft.ifft2(np.fft.fft2(Y) * np.conj(np.fft.fft2(Y)))
    Rho = np.fft.ifftshift(Rho)
    Rho = np.real(Rho / np.sum(Y * Y))
    idx = np.argmax(Rho)
    xc, yc = np.unravel_index(idx, Rho.shape)
    temp = Rho[xc-1, 0:yc-1]
    yc87 = np.sum(temp > 0.75 * np.max(Rho))
    temp = Rho[0:xc-1, yc-1]
    xc87 = np.sum(temp > 0.75 * np.max(Rho))
    Rhopeak = Rho[xc - xc87:xc + xc87 - 1, yc - yc87:yc + yc87 - 1]
    xgrid, ygrid = np.meshgrid(np.arange(-xc87, xc87 - 1), np.arange(-yc87, yc87 - 1))
    xx = np.ravel(xgrid * xgrid)
    yy = np.ravel(ygrid * ygrid)
    xy = np.ravel(xgrid * ygrid)
    x = np.ravel(xgrid)
    y = np.ravel(ygrid)
    c = np.ones(xx.shape)
    r = np.ravel(Rhopeak)
    A = np.column_stack((xx, yy, xy, x, y, c))
    p, _, _, _ = np.linalg.lstsq(A, r)
    return np.real(Rho), np.real(p)


def coomatrix(X):
    Y = MU.checkifRGB(X)
    GLCM = feature.greycomatrix(np.int8(0.25 * Y), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64)
    GLCM_ = np.sum(GLCM, 3)
    GLCM_ = np.sum(GLCM_, 2) / 4.
    for ii in range(GLCM.shape[3]):
        for jj in range(GLCM.shape[2]):
            GLCM[:, :, jj, ii] = GLCM_
    GLCM_ = GLCM_ / np.sum(GLCM_)
    GLCM_[GLCM_ == 0] = np.finfo(np.float32).eps
    xgrid, _ = np.meshgrid(np.arange(0, 64), np.arange(0, 64))
    variance = np.sum(np.power(xgrid - np.mean(GLCM_), 2) * GLCM_)
    Entropy = -np.sum(GLCM_ * np.log2(GLCM_))
    GLCMx = np.sum(GLCM_, 0)
    GLCMy = np.sum(GLCM_, 1)
    Ex = -np.sum(GLCMx * np.log2(GLCMx))
    Ey = -np.sum(GLCMy * np.log2(GLCMy))
    GLCMx = np.tile(GLCMx, (64, 1))
    GLCMy = np.tile(GLCMy, (64, 1))
    Hxy1 = -np.sum(GLCM_ * np.log2(GLCMx * GLCMy))
    info1 = (Entropy - Hxy1) / np.maximum(Ex, Ey)
    Hxy2 = -np.sum((GLCMx * GLCMy) * np.log2(GLCMx * GLCMy))
    info2 = np.sqrt(1 - np.exp(-2 * (Hxy2 - Entropy)))
    c = [feature.greycoprops(GLCM, prop='energy')[0, 0], feature.greycoprops(GLCM, prop='contrast')[0, 0],\
         feature.greycoprops(GLCM, prop='correlation')[0, 0], variance,\
         feature.greycoprops(GLCM, prop='homogeneity')[0, 0], Entropy, info1, info2]
    return GLCM_, np.array(c)


def dwtenergy(X, wname='bior4.4', levels=3):
    Y = MU.checkifRGB(X)
    (cA, (cH, cV, cD)) = pywt.dwt2(Y, wname, 'sym')
    Ea = np.array([np.mean(np.abs(cA)), np.std(np.abs(cA))])
    Eh = np.array([np.mean(np.abs(cH)), np.std(np.abs(cH))])
    Ev = np.array([np.mean(np.abs(cV)), np.std(np.abs(cV))])
    Ed = np.array([np.mean(np.abs(cD)), np.std(np.abs(cD))])
    Ywt = np.concatenate((np.concatenate((cA, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
    for kk in range(levels-1):
        (cA, (cH, cV, cD)) = pywt.dwt2(cA, wname, 'sym')
        Eh = np.hstack((Eh, np.array([np.mean(np.abs(cH)), np.std(np.abs(cH))])))
        Ev = np.hstack((Ev, np.array([np.mean(np.abs(cV)), np.std(np.abs(cV))])))
        Ed = np.hstack((Ed, np.array([np.mean(np.abs(cD)), np.std(np.abs(cD))])))
        Ytemp = np.concatenate((np.concatenate((cA, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
        old_M, old_N = Ytemp.shape
        Ywt[0:old_M, 0:old_N] = Ytemp
    e = np.hstack((Ea,Eh,Ev,Ed))
    return Ywt, e


def eigenfilter(X,r=1.5):
    Y = MU.checkifRGB(X)
    N, _, _ = MU.number_neighbour(r)
    xx = np.arange(0, X.shape[1])
    yy = np.arange(0, X.shape[0])
    fY = interpolate.interp2d(xx, yy, Y, kind='linear')
    X_ii = np.reshape(Y, (Y.size, 1))
    for ii in range(N):
        xi = xx - r * np.sin(2 * np.pi * ii / N)
        yi = yy + r * np.cos(2 * np.pi * ii / N)
        X_ii = np.hstack((X_ii, np.reshape(fY(xi, yi), (Y.size, 1))))
    X_ii = X_ii - np.mean(X_ii, 0)
    S = np.cov(X_ii.T)
    [_, v] = np.linalg.eig(S)
    Xp = np.zeros((Y.shape[0], Y.shape[1], N))
    e = []
    for ii in range(N):
        EY = np.abs(np.reshape(np.dot(X_ii, v[:,ii]), (Y.shape[0], Y.shape[1])))
        Xp[:, :, ii] = EY
        e.extend([np.mean(EY), np.std(EY)])
    return Xp, np.array(e)


def gabor_features(X,Uh=0.4,Ul=0.05,K=6,S=4):
    Y = MU.checkifRGB(X)
    W = Uh
    nstds = 3
    a = (Uh/Ul)**(1./(S-1.))
    sigmau = ((a-1.)*Uh)/((a+1.)*np.sqrt(2.*np.log(2.)))
    fac1 = np.tan(np.pi/(2.*K))
    fac2 = Uh-2.*np.log((2.*sigmau**2.)/Uh)
    fac3 = (2.*np.log(2.)-(((2.*np.log(2.))**2.)*(sigmau**2.)/(Uh**2.)))**(-0.5)
    sigmav = fac1*fac2*fac3
    sigmax = 1/(2*np.pi*sigmau)
    sigmay = 1/(2*np.pi*sigmav)
    xmax = np.maximum(np.abs(nstds*sigmax),np.abs(nstds*sigmay))
    xmax = np.ceil(np.maximum(1,xmax))
    ymax = np.maximum(np.abs(nstds*sigmax),np.abs(nstds*sigmay))
    ymax = np.ceil(np.maximum(1,ymax))
    xmin = -xmax
    ymin = -ymax
    xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    YG = {}
    fac1 = 1./(2.*np.pi*sigmax*sigmay)
    flag = False
    for ii in range(K):
        theta = (ii*np.pi)/K
        for jj in range(S):
            xprime = (a**(-jj))*(xx*np.cos(theta)+yy*np.sin(theta))
            yprime = (a**(-jj))*(-xx*np.sin(theta)+yy*np.cos(theta))
            fac2 = (np.power(xprime,2)/(sigmax**2))+(np.power(yprime,2)/(sigmay**2))
            F = (a**(-jj))*fac1*np.exp(-0.5*fac2+2*np.pi*1j*W*xprime)
            F = F - np.mean(F)
            Fi = np.imag(F)
            Fr = np.real(F)
            Yi = signal.convolve2d(Y, np.rot90(Fi, 2), mode='same')
            Yr = signal.convolve2d(Y, np.rot90(Fr, 2), mode='same')
            if not YG.has_key(ii):
                YG[ii] = {}
            YG[ii][jj] = np.sqrt(Yi * Yi + Yr * Yr)
            H, _ = np.histogram(YG[ii][jj], 256)
            if not flag:
                hist = H
                flag = True
            else:
                hist = np.hstack((hist,H))
    return YG, hist


def gmrf(X):
    Y = MU.checkifRGB(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    Sxy = {}
    H = np.array([[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0]])
    Sxy[0] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    H = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,1,0,1,0], [0,0,0,0,0], [0,0,0,0,0]])
    Sxy[1] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    H = np.array([[0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,0,0]])
    Sxy[2] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    H = np.array([[0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1], [0,0,0,0,0], [0,0,0,0,0]])
    Sxy[3] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    H = np.array([[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0]])
    Sxy[4] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    H = np.array([[0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0]])
    Sxy[5] = signal.convolve2d(Y, np.rot90(H, 2), mode='same')
    S = np.zeros((6, 6))
    G = np.zeros((6, 1))
    for ii in range(6):
        for jj in range(6):
            S[ii,jj] = np.sum(Sxy[ii]*Sxy[jj])
        G[ii] = np.sum(Y*Sxy[ii])
    alpha, _, _, _ = np.linalg.lstsq(S, G)
    Xhat = np.zeros_like(Y)
    for ii in range(6):
        Xhat += alpha[ii]*Sxy[ii]
    alpha = np.vstack((alpha,(1. / ((Y.shape[0] - 2) * (Y.shape[1] - 2))) * np.sum(np.power(Y-Xhat,2))))
    return Xhat, alpha


def granulometrymoments(X, lamb=20):
    Y = MU.checkifRGB(X)
    ave = np.mean(Y)
    var = np.var(Y)
    ske = stats.skew(np.ravel(Y))
    kur = stats.kurtosis(np.ravel(Y))
    ave_open = []
    var_open = []
    ske_open = []
    kur_open = []
    ave_clos = []
    var_clos = []
    ske_clos = []
    kur_clos = []
    Yout = {}
    for ii in range(lamb):
        SE = cv2.getStructuringElement(cv2.MORPH_RECT, (ii+2, ii+2))
        Yopen = cv2.morphologyEx(Y, cv2.MORPH_OPEN, SE)
        ave_open.append(1.-np.mean(Yopen)/ave)
        var_open.append(1.-np.var(Yopen)/var)
        ske_open.append(1.-stats.skew(np.ravel(Yopen))/ske)
        kur_open.append(1.-stats.kurtosis(np.ravel(Yopen))/kur)
        Yclose = cv2.morphologyEx(Y, cv2.MORPH_CLOSE, SE)
        ave_clos.append(1.-np.mean(Yclose)/ave)
        var_clos.append(1.-np.var(Yclose)/var)
        ske_clos.append(1.-stats.skew(np.ravel(Yopen))/ske)
        kur_clos.append(1.-stats.kurtosis(np.ravel(Yopen))/kur)
        Yout[ii] = np.dstack((Yopen,Yclose))
    feat_open = np.hstack((np.array(ave_open), np.array(var_open), np.array(ske_open), np.array(kur_open)))
    feat_clos = np.hstack((np.array(ave_clos), np.array(var_clos), np.array(ske_clos), np.array(kur_clos)))
    return Yout, np.hstack((feat_open,feat_clos))


def laplacianpyramid(X,s=3,sigma=1,wsize=7):
    Y = MU.checkifRGB(X)
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2 + 1), np.arange(-wsize / 2, wsize / 2 + 1))
    window = MU.gaussian(xx, yy, sigma)
    e = []
    Xpyr = {}
    for ii in range(s):
        temp = Y
        Y = signal.convolve2d(Y, np.rot90(window, 2), mode='same')
        temp = np.abs(temp - Y)
        Xpyr[ii] = temp
        e.extend([np.mean(temp), np.std(temp)])
        Y = Y[0::2, 0::2]
    return Xpyr, np.array(e)


def lawsoperators(X):
    Y = MU.checkifRGB(X)
    L = [np.array([1,4,6,4,1]), np.array([-1,-2,0,2,1]), np.array([-1,0,2,0,-1]), np.array([-1,2,0,-2,1]), np.array([1,-4,6,-4,1])]
    e = []
    Laws = {}
    H = np.ones((11,11))
    for ii in range(5):
        for jj in range(5):
            G = np.dot(np.reshape(L[ii], (L[ii].size, 1)), np.reshape(L[jj], (1, L[jj].size)))
            Laws[ii,jj] = signal.convolve2d(Y, np.rot90(G, 2), mode='same')
            Laws[ii,jj] = Laws[ii,jj]*Laws[ii,jj]
            Laws[ii,jj] = np.sqrt(signal.convolve2d(Laws[ii,jj], np.rot90(H, 2), mode='same'))
            if not(ii==0 and jj==0):
                Laws[ii,jj] = Laws[ii,jj]/Laws[0,0]
    M, N = Y.shape
    Xp = np.zeros((M, N, 10))
    c = 0
    for ii in range(5):
        for jj in range(ii+1,5):
            LE = np.abs(Laws[ii,jj] + Laws[jj,ii])
            Xp[:,:,c] = LE
            c += 1
            if not (ii == 0 and jj == 0):
                e.extend([np.mean(LE), np.std(LE)])
    return Xp, np.array(e)


def lbp(X,r=1.5,color_=False,th=0):
    N, Table, Scale = MU.number_neighbour(r)
    xx = np.arange(0, X.shape[0])
    yy = np.arange(0, X.shape[1])
    X_ii = {}
    if color_:
        XYZ = CAM.rgbToxyz(X / 255.)
        O123 = CAM.xyzToo123(XYZ)
        fY = {}
        for ii in range(3):
            fY[ii] = interpolate.RectBivariateSpline(xx, yy, O123[:,:,ii])#interpolate.interp2d(xx, yy, O123[:,:,ii], kind='linear')
    else:
        Y = MU.checkifRGB(X)
        fY = interpolate.RectBivariateSpline(xx, yy, Y/255.)#interpolate.interp2d(xx, yy, Y/255., kind='linear')
    LBP = np.zeros((X.shape[0],X.shape[1]))
    for ii in range(N):
        xi = xx - r * np.sin(2. * np.pi * ii / N)
        yi = yy + r * np.cos(2. * np.pi * ii / N)
        xigrid, yigrid = np.meshgrid(yi, xi)
        if color_:
            Temp = np.zeros_like(XYZ)
            for jj in range(3):
                Temp[:,:,jj] = fY[jj].ev(yigrid, xigrid)#fY[jj](xi, yi)
            Temp = CAM.o123Toxyz(Temp)
            Temp = CAM.xyzTorgb(Temp)
            X_ii[ii], _ = CDMP.cd00_deltaE2000(X,np.uint8(255.*Temp))
        else:
            Temp = np.zeros_like(Y)
            Temp += fY.ev(yigrid, xigrid)#fY(xi, yi)
            X_ii[ii] = np.fabs((Y/255.)-Temp)
        X_ii[ii] = X_ii[ii] > th
        LBP += np.double(X_ii[ii])*np.power(2.,ii)
    LBP = Table[0,np.int_(LBP)]
    H, _ = np.histogram(LBP, N + 1)
    return np.double(LBP), np.double(H)


def steerablepyramid(X,s=3,sigma=1,wsize=7):
    Y = MU.checkifRGB(X)
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2 + 1), np.arange(-wsize / 2, wsize / 2 + 1))
    window = MU.gaussian(xx, yy, sigma)
    Gx, Gy = np.gradient(window)
    theta = [0,45,90,135]
    e = []
    Xout = {}
    for ii in range(s):
        Xx = signal.convolve2d(Y, np.rot90(Gx, 2), mode='same')
        Xy = signal.convolve2d(Y, np.rot90(Gy, 2), mode='same')
        Xout[ii] = {}
        for jj in range(len(theta)):
            Xt = np.cos(np.pi*theta[jj]/180.) * Xx + np.sin(np.pi*theta[jj]/180.) * Xy
            e.extend([np.mean(Xt), np.std(Xt)])
            Xout[ii][jj] = Xt
        Y = signal.convolve2d(Y, np.rot90(window, 2), mode='same')
        Y = Y[0::2,0::2]
    return Xout, np.array(e)


def power_spectrum_fft(X):
    Y = MU.checkifRGB(X)
    F = np.zeros((1, 10))
    XFFT = np.fft.fft2(np.fft.ifftshift(Y))
    XFFT_full = XFFT
    XFFT = np.power(np.absolute(np.fft.fftshift(XFFT)),2)
    xc = np.int_(XFFT.shape[0]/2)
    yc = np.int_(XFFT.shape[1]/2)
    XFFT /= np.sqrt(np.sum(np.power(XFFT,2))-np.power(XFFT[xc,yc],2))
    r = np.array([[2,4,8,16,32,64],[4,8,16,32,64,128]])
    theta1 = np.array([[112.5,67.5,22.5,157.5], [67.5,22.5,337.5,112.5]])
    theta2 = np.array([[247.5,247.5,202.5,292.5], [292.5,202.5,157.5,337.5]])
    phi = np.arange(0,360)
    power_images = {}
    for ii in range(r.shape[1]):
        BWi = np.zeros(XFFT.shape)
        BWe = np.zeros(XFFT.shape)
        xi = xc + r[0, ii] * np.sin(phi*np.pi/180.)
        yi = yc + r[0, ii] * np.cos(phi*np.pi/180.)
        rr, cc = draw.polygon(xi, yi, shape=XFFT.shape)
        BWi[rr,cc] = 1.
        xi = xc + r[1, ii] * np.sin(phi*np.pi/180.)
        yi = yc + r[1, ii] * np.cos(phi*np.pi/180.)
        rr, cc = draw.polygon(xi, yi, shape=XFFT.shape)
        BWe[rr, cc] = 1.
        BW = BWe - BWi
        F[0,ii] = np.sum(BW*XFFT)
        power_images[ii] = np.absolute(np.fft.ifftshift(np.fft.ifft2(BW*XFFT_full)))
    rho = 128
    for ii in range(theta1.shape[1]):
        BW1 = np.zeros(XFFT.shape)
        BW2 = np.zeros(XFFT.shape)
        xi = xc + rho * np.sin(theta1[0, ii]*np.pi/180.)
        yi = yc + rho * np.cos(theta1[0, ii]*np.pi/180.)
        xi = np.vstack((np.vstack((xi,xc+rho*np.sin(theta1[1, ii]*np.pi/180.))), xc))
        yi = np.vstack((np.vstack((yi,yc+rho*np.cos(theta1[1, ii]*np.pi/180.))), yc))
        rr, cc = draw.polygon(xi, yi, shape=XFFT.shape)
        BW1[rr, cc] = 1.
        xi = xc + rho * np.sin(theta2[0, ii]*np.pi/180.)
        yi = yc + rho * np.cos(theta2[0, ii]*np.pi/180.)
        xi = np.vstack((np.vstack((xi, xc + rho*np.sin(theta2[1, ii] * np.pi / 180.))), xc))
        yi = np.vstack((np.vstack((yi, yc + rho*np.cos(theta2[1, ii] * np.pi / 180.))), yc))
        rr, cc = draw.polygon(xi, yi, shape=XFFT.shape)
        BW2[rr, cc] = 1.
        BW = BW1 + BW2
        F[0,r.shape[1]+ii] = np.sum(BW*XFFT)
        power_images[r.shape[1]+ii] = np.absolute(np.fft.ifftshift(np.fft.ifft2(BW * XFFT_full)))
    return power_images, F


def wigner_distribution(X,W=3):
    Y = MU.checkifRGB(X)
    xx, yy = np.meshgrid(np.arange(-W / 2, W / 2 + 1), np.arange(-W / 2, W / 2 + 1))
    hs = MU.gaussian(xx, yy, 1.)
    hf = MU.gaussian(xx, yy, W / 6.)
    PWD = cython_functions.pseudo_wigner(Y, hs, hf, W)
    PWD *= (np.sum(Y*Y) / np.sum(PWD))
    flag = False
    for ii in range(PWD.shape[2]):
        for jj in range(PWD.shape[3]):
            T = PWD[:,:,ii,jj]
            H, _ = np.histogram(T[np.logical_not(np.isnan(T))], 256)
            if not flag:
                F = H
                flag = True
            else:
                F = np.hstack((F,H))
    return PWD, F





import matplotlib.pyplot as plt
if __name__ == "__main__":
    MUltimedia_file_ref = './sample_images/test_ref_0.bmp'
    image_ref = ndimage.imread(MUltimedia_file_ref)
    Yar, alpha = granulometrymoments(image_ref,30)
    print alpha
    plt.imshow(Yar[29][:,:,0])
    plt.colorbar()
    plt.show()
