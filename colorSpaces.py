from scipy import interpolate
from scipy import signal
from scipy import misc
from scipy import io
import numpy as np
import myUtilities


def RGB2XYZ(RGB):
    T = np.array([[42.62846,  38.29084,  13.67019],\
                  [21.64618,  72.06528,   5.83799],\
                  [1.77295,  12.93408,  92.75945]])
    X = T[0, 0] * RGB[:, :, 0] + T[0, 1] * RGB[:, :, 1] + T[0, 2] * RGB[:, :, 2]
    Y = T[1, 0] * RGB[:, :, 0] + T[1, 1] * RGB[:, :, 1] + T[1, 2] * RGB[:, :, 2]
    Z = T[2, 0] * RGB[:, :, 0] + T[2, 1] * RGB[:, :, 1] + T[2, 2] * RGB[:, :, 2]
    return np.dstack((X, Y, Z))


def XYZ2RGB(XYZ):
    A = np.array([[42.62846,  38.29084,  13.67019],\
                  [21.64618,  72.06528,   5.83799],\
                  [1.77295,  12.93408,  92.75945]])
    A = np.linalg.inv(A)
    R = A[0, 0] * XYZ[:, :, 0] + A[0, 1] * XYZ[:, :, 1] + A[0, 2] * XYZ[:, :, 2]
    G = A[1, 0] * XYZ[:, :, 0] + A[1, 1] * XYZ[:, :, 1] + A[1, 2] * XYZ[:, :, 2]
    B = A[2, 0] * XYZ[:, :, 0] + A[2, 1] * XYZ[:, :, 1] + A[2, 2] * XYZ[:, :, 2]
    return np.dstack((R, G, B))


def XYZ2O1O2O3(XYZ):
    A = np.array([[0.2787,  0.7218, -0.1066],\
                  [-0.4488,  0.2898, -0.0772],\
                  [0.0860, -0.5900,  0.5011]])
    O1 = A[0, 0] * XYZ[:, :, 0] + A[0, 1] * XYZ[:, :, 1] + A[0, 2] * XYZ[:, :, 2]
    O2 = A[1, 0] * XYZ[:, :, 0] + A[1, 1] * XYZ[:, :, 1] + A[1, 2] * XYZ[:, :, 2]
    O3 = A[2, 0] * XYZ[:, :, 0] + A[2, 1] * XYZ[:, :, 1] + A[2, 2] * XYZ[:, :, 2]
    return np.dstack((O1, O2, O3))


def O1O2O32XYZ(O1O2O3):
    A = np.array([[0.2787,  0.7218, -0.1066],\
                  [-0.4488,  0.2898, -0.0772],\
                  [0.0860, -0.5900,  0.5011]])
    A = np.linalg.inv(A)
    X = A[0, 0] * O1O2O3[:, :, 0] + A[0, 1] * O1O2O3[:, :, 1] + A[0, 2] * O1O2O3[:, :, 2]
    Y = A[1, 0] * O1O2O3[:, :, 0] + A[1, 1] * O1O2O3[:, :, 1] + A[1, 2] * O1O2O3[:, :, 2]
    Z = A[2, 0] * O1O2O3[:, :, 0] + A[2, 1] * O1O2O3[:, :, 1] + A[2, 2] * O1O2O3[:, :, 2]
    return np.dstack((X, Y, Z))


def RGB2LAlphaBeta(RGB):
    A = np.array([[0.3811, 0.5783, 0.0402],\
                  [0.1967, 0.7244, 0.0782],\
                  [0.0241, 0.1288, 0.8444]])
    L = np.log(A[0, 0] * RGB[:, :, 0] + A[0, 1] * RGB[:, :, 1] + A[0, 2] * RGB[:, :, 2] + 1)
    M = np.log(A[1, 0] * RGB[:, :, 0] + A[1, 1] * RGB[:, :, 1] + A[1, 2] * RGB[:, :, 2] + 1)
    S = np.log(A[2, 0] * RGB[:, :, 0] + A[2, 1] * RGB[:, :, 1] + A[2, 2] * RGB[:, :, 2] + 1)
    A = np.dot(np.array([[1. / np.sqrt(3.), 0., 0.],\
                         [0., 1. / np.sqrt(6.), 0.],\
                         [0., 0., 1. / np.sqrt(2.)]]),\
               np.array([[1.,  1.,  1.],\
                         [1.,  1., -2.],\
                         [1., -1.,  0.]]))
    l = A[0, 0] * L + A[0, 1] * M + A[0, 2] * S
    alpha = A[1, 0] * L + A[1, 1] * M + A[1, 2] * S
    beta = A[2, 0] * L + A[2, 1] * M + A[2, 2] * S
    return np.dstack((l, alpha, beta))


def RGB2YCbCr(RGB):
    a0 = np.array([16.,128.,128.])
    a1 = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]])/255.
    Y  = a0[0]+a1[0,0]*RGB[:,:,0]+a1[0,1]*RGB[:,:,1]+a1[0,2]*RGB[:,:,2]
    Cb = a0[1]+a1[1,0]*RGB[:,:,0]+a1[1,1]*RGB[:,:,1]+a1[1,2]*RGB[:,:,2]
    Cr = a0[2]+a1[2,0]*RGB[:,:,0]+a1[2,1]*RGB[:,:,1]+a1[2,2]*RGB[:,:,2]
    Cb = Cb-128
    Cr = Cr-128
    return np.dstack((Y,Cb,Cr))


def RGB2iCAM(RGB):
    max_L = 20000.
    p = 0.7
    gamma_value = 1.
    M = np.array([[0.412424, 0.212656, 0.0193324], [0.357579, 0.715158, 0.119193],\
                  [0.180464, 0.0721856, 0.950444]])
    X = M[0, 0] * RGB[:, :, 0] + M[0, 1] * RGB[:, :, 1] + M[0, 2] * RGB[:, :, 2]
    Y = M[1, 0] * RGB[:, :, 0] + M[1, 1] * RGB[:, :, 1] + M[1, 2] * RGB[:, :, 2]
    Z = M[2, 0] * RGB[:, :, 0] + M[2, 1] * RGB[:, :, 1] + M[2, 2] * RGB[:, :, 2]
    XYZimg = np.dstack((X,Y,Z))
    XYZimg = XYZimg / np.max(XYZimg[:,:, 1]) * max_L
    XYZimg[np.where(XYZimg < 0.00000001)] = 0.00000001
    base_imgX, detail_imgX = myUtilities.fastbilateralfilter(XYZimg[:,:, 0])
    base_imgY, detail_imgY = myUtilities.fastbilateralfilter(XYZimg[:,:, 1])
    base_imgZ, detail_imgZ = myUtilities.fastbilateralfilter(XYZimg[:,:, 2])
    base_img = np.dstack((base_imgX,base_imgY,base_imgZ))
    detail_img = np.dstack((detail_imgX, detail_imgY, detail_imgZ))
    white = iCAM06_blur(XYZimg, 2)
    XYZ_adapt = iCAM06_CAT(base_img, white)
    white = iCAM06_blur(XYZimg, 3)
    XYZ_tc = iCAM06_TC(XYZ_adapt, white, p)
    XYZ_d = XYZ_tc * iCAM06_LocalContrast(detail_img, base_img)
    return iCAM06_IPT(XYZ_d, base_img, gamma_value)


def RGB2Ljg(RGB):
    M_XYZToRGB = np.array([[0.799, 0.4194, -0.1648],\
                           [-0.4493, 1.3265, 0.0927],\
                           [-0.1149, 0.3394, 0.7170]])
    M_RGBToXYZ = np.linalg.inv(M_XYZToRGB)
    X = M_RGBToXYZ[0, 0] * RGB[:, :, 0] + M_RGBToXYZ[0, 1] * RGB[:, :, 1] + M_RGBToXYZ[0, 2] * RGB[:, :, 2]
    Y = M_RGBToXYZ[1, 0] * RGB[:, :, 0] + M_RGBToXYZ[1, 1] * RGB[:, :, 1] + M_RGBToXYZ[1, 2] * RGB[:, :, 2]
    Z = M_RGBToXYZ[2, 0] * RGB[:, :, 0] + M_RGBToXYZ[2, 1] * RGB[:, :, 1] + M_RGBToXYZ[2, 2] * RGB[:, :, 2]
    XYZ = np.dstack((X, Y, Z))
    RGB3 = np.power(RGB, 1./3.)
    xyY = XYZToxyY(XYZ)
    x= xyY[:, :, 0]
    y= xyY[:, :, 1]
    Y= xyY[:, :, 2]
    Y0 = Y * (4.4934 * np.power(x, 2) + 4.3034 * np.power(y, 2) - 4.276 * (x * y) - 1.3744 * x - 2.5643 * y + 1.8103)
    scriptL = np.zeros(Y0.shape)
    index = np.where(Y0 > 30)
    if index[0].any():
        scriptL[index] = 5.9 * (np.power(Y0[index], 1./3.) - (2. / 3.) +\
                                0.042 * (np.power(np.abs(Y0[index] - 30), 1./3.)))
    index = np.where(Y0 <= 30)
    if index[0].any():
        scriptL[index] = 5.9 * (np.power(Y0[index], 1./3.) - (2. / 3.) -\
                                0.042 * (np.power(np.abs(Y0[index] - 30), 1./3.)))
    C = scriptL / (5.9 * (np.power(Y0, 1. / 3.)) - (2. / 3.))
    L = (scriptL - 14.4) / np.sqrt(2.)
    j = C * (1.7 * RGB3[:, :, 0] + 8 * RGB3[:, :, 1] - 9.7 * RGB3[:, :, 2])
    g = C * (-13.7 * RGB3[:, :, 0] + 17.7 * RGB3[:, :, 1] - 4 * RGB3[:, :, 2])
    return np.real(np.dstack((L, j, g)))


# Extra functions necessary to compute color spaces, e.g., intermediate color spaces
def iCAM06_blur(img, d):
    sy,sx,sz = img.shape
    m = np.minimum(sy,sx)
    if m<64:
        z = 1
    elif m<256:
        z = 2
    elif m<512:
        z = 4
    elif m<1024:
        z = 8
    elif m<2056:
        z = 16
    else:
        z = 32
    img = img[0::z,0::z,:]
    imSize = img.shape
    xDim = imSize[1]
    yDim = imSize[0]
    Y = np.zeros((2*yDim, 2*xDim,3))
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim),:] = img
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),0:int(round(xDim/2)),:] = img[:,0:int(round(xDim/2)),:]
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)+xDim):int(2*xDim),:] = img[:,int(round(xDim/2)):int(xDim),:]
    Y[0:int(round(yDim/2)),int(round(xDim/2)):int(round(xDim/2)+xDim),:] = img[0:int(round(yDim/2)),:,:]
    Y[int(round(yDim/2)+yDim):int(2*yDim),int(round(xDim/2)):int(round(xDim/2)+xDim),:] = img[int(round(yDim/2)):int(yDim),:,:]
    Y[0:int(round(yDim/2)),0:int(round(xDim/2)),:] = img[0:int(round(yDim/2)),0:int(round(xDim/2)),:]
    Y[0:int(round(yDim/2)),int(round(xDim/2)+xDim):int(2*xDim),:] = img[0:int(round(yDim/2)),int(round(xDim/2)):int(xDim),:]
    Y[int(round(yDim/2)+yDim):int(2*yDim), int(round(xDim/2)+xDim):int(2*xDim),:] = img[int(round(yDim/2)):int(yDim),int(round(xDim/2)):int(xDim),:]
    Y[int(round(yDim/2)+yDim):int(2*yDim), 0:int(round(xDim/2)),:] = img[int(round(yDim/2)):int(yDim),0:int(round(xDim/2)),:]
    distMap = myUtilities.idl_dist(Y.shape[0],Y.shape[1])
    Dim = np.maximum(xDim, yDim)
    kernel = np.exp(-1*np.power(distMap/(Dim/d),2))
    filter = np.maximum(np.real(np.fft.fft(kernel)),0)
    filter = filter/filter[0,0]
    whiteX = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,0])*filter)),0)
    whiteY = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,1])*filter)),0)
    whiteZ = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,2])*filter)),0)
    white = np.dstack((whiteX[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)], \
                       whiteY[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)], \
                       whiteZ[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)]))
    white = misc.imresize(white, float(z), 'nearest')
    return white[0:sy,0:sx,:]


def iCAM06_CAT(XYZimg, white):
    M = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6974, 0.0061], [0.0030, 0.0136, 0.9834]])
    Mi = np.linalg.inv(M.T)
    R = M[0, 0] * XYZimg[:, :, 0] + M[0, 1] * XYZimg[:, :, 1] + M[0, 2] * XYZimg[:, :, 2]
    G = M[1, 0] * XYZimg[:, :, 0] + M[1, 1] * XYZimg[:, :, 1] + M[1, 2] * XYZimg[:, :, 2]
    B = M[2, 0] * XYZimg[:, :, 0] + M[2, 1] * XYZimg[:, :, 1] + M[2, 2] * XYZimg[:, :, 2]
    RGB_img = np.dstack((R,G,B))
    R = M[0, 0] * white[:, :, 0] + M[0, 1] * white[:, :, 1] + M[0, 2] * white[:, :, 2]
    G = M[1, 0] * white[:, :, 0] + M[1, 1] * white[:, :, 1] + M[1, 2] * white[:, :, 2]
    B = M[2, 0] * white[:, :, 0] + M[2, 1] * white[:, :, 1] + M[2, 2] * white[:, :, 2]
    RGB_white = np.dstack((R,G,B))
    xyz_d65 = np.array([ 95.05,  100.0, 108.88])
    R = M[0, 0] * xyz_d65[0] + M[0, 1] * xyz_d65[1] + M[0, 2] * xyz_d65[2]
    G = M[1, 0] * xyz_d65[0] + M[1, 1] * xyz_d65[1] + M[1, 2] * xyz_d65[2]
    B = M[2, 0] * xyz_d65[0] + M[2, 1] * xyz_d65[1] + M[2, 2] * xyz_d65[2]
    La = 0.2*white[:,:,1]
    F = 1
    D = 0.3*F*(1-(1/3.6)*np.exp(-1*(La-42)/92))
    RGB_white = RGB_white+0.0000001
    Rc = (D * R/RGB_white[:,:,0] + 1 - D) * RGB_img[:,:,0]
    Gc = (D * G/RGB_white[:,:,1] + 1 - D) * RGB_img[:,:,1]
    Bc = (D * B/RGB_white[:,:,2] + 1 - D) * RGB_img[:,:,2]
    adaptImage = np.dstack((Rc, Gc, Bc))
    X = Mi[0, 0] * adaptImage[:, :, 0] + Mi[0, 1] * adaptImage[:, :, 1] + Mi[0, 2] * adaptImage[:, :, 2]
    Y = Mi[1, 0] * adaptImage[:, :, 0] + Mi[1, 1] * adaptImage[:, :, 1] + Mi[1, 2] * adaptImage[:, :, 2]
    Z = Mi[2, 0] * adaptImage[:, :, 0] + Mi[2, 1] * adaptImage[:, :, 1] + Mi[2, 2] * adaptImage[:, :, 2]
    return np.dstack((X, Y, Z))


def iCAM06_TC(XYZ_adapt, white_img, p):
    M = np.array([ [0.38971, 0.68898, -0.07868],[-0.22981, 1.18340,  0.04641],[ 0.00000, 0.00000,  1.00000]])
    Mi = np.linalg.inv(M.T)
    R = M[0, 0] * XYZ_adapt[:, :, 0] + M[0, 1] * XYZ_adapt[:, :, 1] + M[0, 2] * XYZ_adapt[:, :, 2]
    G = M[1, 0] * XYZ_adapt[:, :, 0] + M[1, 1] * XYZ_adapt[:, :, 1] + M[1, 2] * XYZ_adapt[:, :, 2]
    B = M[2, 0] * XYZ_adapt[:, :, 0] + M[2, 1] * XYZ_adapt[:, :, 1] + M[2, 2] * XYZ_adapt[:, :, 2]
    RGB_img = np.dstack((R, G, B))
    La = 0.2*white_img[:,:,1]
    k = 1./(5.*La+1)
    FL = 0.2*np.power(k,4)*(5*La)+0.1*np.power(1-np.power(k,4),2)*np.power(5*La,1/3)
    FL = np.dstack((FL, FL, FL))
    white_3img = np.dstack((white_img[:,:,1], white_img[:,:,1], white_img[:,:,1]))
    sign_RGB = np.sign(RGB_img)
    RGB_c = sign_RGB*((400 * np.power(FL*np.abs(RGB_img)/white_3img,p))/ \
                      (27.13 + np.power(FL*np.abs(RGB_img)/white_3img,p)) ) + .1
    Las = 2.26*La
    j = 0.00001/(5*Las/2.26+0.00001)
    FLS = 3800*np.power(j,2)*(5*Las/2.26)+0.2*np.power(1-np.power(j,2),4)*np.power(5*Las/2.26,1/6)
    Sw = np.max(5*La)
    S = np.abs(XYZ_adapt[:,:,1])
    Bs = 0.5/(1+.3*np.power((5*Las/2.26)*(S/Sw),3))+0.5/(1+5*(5*Las/2.26))
    As = 3.05*Bs*(((400 * np.power(FLS*(S/Sw),p))/  (27.13 + np.power(FLS*(S/Sw),p)) )) + .03
    As = np.dstack((As, As, As))
    RGB_c = RGB_c + As
    R = Mi[0, 0] * RGB_c[:, :, 0] + Mi[0, 1] * RGB_c[:, :, 1] + Mi[0, 2] * RGB_c[:, :, 2]
    G = Mi[1, 0] * RGB_c[:, :, 0] + Mi[1, 1] * RGB_c[:, :, 1] + Mi[1, 2] * RGB_c[:, :, 2]
    B = Mi[2, 0] * RGB_c[:, :, 0] + Mi[2, 1] * RGB_c[:, :, 1] + Mi[2, 2] * RGB_c[:, :, 2]
    return np.dstack((R, G, B))


def iCAM06_LocalContrast(detail, base_img):
    La = 0.2 * base_img[:,:, 1]
    k = 1. / (5 * La + 1)
    FL = 0.2 * np.power(k, 4) * (5 * La) + 0.1 * np.power(1 - np.power(k, 4),2) * np.power(5 * La,1/3)
    FL = np.dstack((FL, FL, FL))
    return np.power(detail, np.power((FL + 0.8),.25))


def iCAM06_IPT(XYZ_img, base_img, gamma):
    xyz2lms = np.array([[.4002, .7077, -.0807],[-.2280, 1.1500, .0612],[.0, .0, .9184]]).T
    iptMat = np.array([[ 0.4000, 0.4000, 0.2000],[ 4.4550,-4.8510, 0.3960],[ 0.8056, 0.3572,-1.1628] ]).T
    L = xyz2lms[0, 0] * XYZ_img[:, :, 0] + xyz2lms[0, 1] * XYZ_img[:, :, 1] + xyz2lms[0, 2] * XYZ_img[:, :, 2]
    M = xyz2lms[1, 0] * XYZ_img[:, :, 0] + xyz2lms[1, 1] * XYZ_img[:, :, 1] + xyz2lms[1, 2] * XYZ_img[:, :, 2]
    S = xyz2lms[2, 0] * XYZ_img[:, :, 0] + xyz2lms[2, 1] * XYZ_img[:, :, 1] + xyz2lms[2, 2] * XYZ_img[:, :, 2]
    lms_img = np.dstack((L, M, S))
    lms_img = np.power(np.abs(lms_img),.43)
    i = iptMat[0, 0] * lms_img[:, :, 0] + iptMat[0, 1] * lms_img[:, :, 1] + iptMat[0, 2] * lms_img[:, :, 2]
    p = iptMat[1, 0] * lms_img[:, :, 0] + iptMat[1, 1] * lms_img[:, :, 1] + iptMat[1, 2] * lms_img[:, :, 2]
    t = iptMat[2, 0] * lms_img[:, :, 0] + iptMat[2, 1] * lms_img[:, :, 1] + iptMat[2, 2] * lms_img[:, :, 2]
    ipt_img = np.dstack((i, p, t))
    c = np.sqrt(np.power(ipt_img[:,:,1],2)+np.power(ipt_img[:,:,2],2))
    La = 0.2*base_img[:,:,1]
    k = 1/(5*La+1)
    FL = 0.2*np.power(k,4)*(5*La)+0.1*np.power(1-np.power(k,4),2)*np.power(5*La,1/3)
    ipt_img[:,:,1] = ipt_img[:,:,1]*(np.power(FL+1,.15)*((1.29*np.power(c,2)-0.27*c+0.42)/(np.power(c,2)-0.31*c+0.42)))
    ipt_img[:,:,2] = ipt_img[:,:,2]*(np.power(FL+1,.15)*((1.29*np.power(c,2)-0.27*c+0.42)/(np.power(c,2)-0.31*c+0.42)))
    max_i = np.max(ipt_img[:,:,0])
    ipt_img[:,:,0] = ipt_img[:,:,0]/max_i
    ipt_img[:,:,0] = np.power(ipt_img[:,:,0],(gamma))
    ipt_img[:,:,0] = ipt_img[:,:,0]*max_i
    return ipt_img


def ImageSRGB2XYZ(IM_SRGB):
    return SRGB2XYZ(IM_SRGB)


def SRGB2XYZ(SRGB):
    R = invgammacorrection(SRGB[:,:,0]/255.)
    G = invgammacorrection(SRGB[:,:,1]/255.)
    B = invgammacorrection(SRGB[:,:,2]/255.)
    T = np.linalg.inv(np.array([[3.2406, -1.5372, -0.4986],\
                                 [-0.9689, 1.8758, 0.0415],\
                                 [0.0557, -0.2040, 1.057]]))
    X = T[0,0] * R + T[0,1] * G + T[0,2] * B
    Y = T[1,0] * R + T[1,1] * G + T[1,2] * B
    Z = T[2,0] * R + T[2,1] * G + T[2,2] * B
    return np.dstack((X, Y, Z))


def invgammacorrection(Rp):
    R = np.zeros(Rp.shape)
    ii = np.where(Rp <= 0.0404482362771076)
    R[ii] = Rp[ii]/12.92
    ii = np.where(Rp > 0.0404482362771076)
    R[ii] = np.real(np.power((Rp[ii] + 0.055)/1.055,2.4))
    return R


def changeColorSpace(image, T):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    X = T[0, 0] * R + T[0, 1] * G + T[0, 2] * B
    Y = T[1, 0] * R + T[1, 1] * G + T[1, 2] * B
    Z = T[2, 0] * R + T[2, 1] * G + T[2, 2] * B
    return np.dstack((X, Y, Z))


def scielab_simple(sampPerDeg, image):
    T = np.array([[278.7336, 721.8031, -106.5520],\
                  [-448.7736, 289.8056, 77.1569],\
                  [85.9513, -589.9859, 501.1089]])/1000.
    opp = changeColorSpace(image, T)
    [k1, k2, k3] = myUtilities.separableFilters(sampPerDeg, 3)
    p1 = myUtilities.separableConv(opp[:,:,0], k1, np.abs(k1))
    p2 = myUtilities.separableConv(opp[:,:,1], k2, np.abs(k2))
    p3 = myUtilities.separableConv(opp[:,:,2], k3, np.abs(k3))
    opp = np.dstack((p1, p2, p3))
    xyz = changeColorSpace(opp, np.linalg.inv(T))
    return xyz


def ImageXYZ2LAB2000HL(IM_XYZ):
    return XYZ2LAB2000HL(IM_XYZ)


def XYZ2LAB2000HL(XYZ):
    LAB = XYZ2LAB(XYZ)
    return LAB2LAB2000HL(LAB)


def XYZ2LAB(XYZ):
    WhitePoint = np.array([0.950456, 1, 1.088754])
    X = XYZ[:,:, 0] / WhitePoint[0]
    Y = XYZ[:,:, 1] / WhitePoint[1]
    Z = XYZ[:,:, 2] / WhitePoint[2]
    fX = myUtilities.f(X)
    fY = myUtilities.f(Y)
    fZ = myUtilities.f(Z)
    L = 116. * fY - 16.
    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)
    return np.dstack((L, a, b))


def LAB2LAB2000HL(LAB):
    L = LAB[:, :, 0]
    a = LAB[:, :, 1]
    b = LAB[:, :, 2]
    L[L < 0] = 0
    L[L > 100] = 100
    a[a < -128] = -128
    a[a > 128] = 128
    b[b < -128] = -128
    b[b > 128] = 128
    mat_contents = io.loadmat('LAB2000HL.mat')
    RegularGrid = mat_contents['RegularGrid']
    Lgrid = mat_contents['L']
    fL = interpolate.interp1d(np.arange(0,100+0.001,0.001), Lgrid)
    L2000HL = fL(L).reshape(L.shape)
    x = np.arange(-128,129)
    y = np.arange(-128,129)
    fa = interpolate.RectBivariateSpline(x, y, RegularGrid[:,:,0])
    a2000HL = fa.ev(a,b)
    fb = interpolate.RectBivariateSpline(x, y, RegularGrid[:,:,1])
    b2000HL = fb.ev(a,b)
    return np.dstack((L2000HL, a2000HL, b2000HL))


def XYZToxyY(XYZ):
    denom = np.sum(XYZ, 2)
    denom[denom == 0.] = 1.
    x = XYZ[:, :, 0] / denom
    y = XYZ[:, :, 1] / denom
    return np.dstack((x, y, XYZ[:, :, 1]))
