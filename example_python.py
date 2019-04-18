import numpy as np
from skimage import color
from scipy import ndimage
import scipy.misc
import content_features as CF

img = ndimage.imread('/media/bortiz/Data/PhD_Thesis_files/Experiments/Chapt6-color/Tid2013Color/i04.bmp')
Lab_ref = color.rgb2lab(img)
L_ref = Lab_ref[:, :, 0]
a_ref = Lab_ref[:, :, 1]
b_ref = Lab_ref[:, :, 2]
L = np.mean(L_ref)
a = np.mean(a_ref)
b = np.mean(b_ref)
L_ref = L*np.ones(L_ref.shape)
a_ref = a*np.ones(a_ref.shape)
b_ref = b*np.ones(b_ref.shape)
RGB_ref = color.lab2rgb(np.dstack((L_ref, a_ref, b_ref)))
scipy.misc.imsave('/media/bortiz/Data/average_color_stw.png', np.uint8(255.*RGB_ref))
scipy.misc.imsave('/media/bortiz/Data/principal_color_stw.png', np.uint8(255.*CF.principal_color(img)*np.ones(img.shape)))
