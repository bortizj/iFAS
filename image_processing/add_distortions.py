"""
Copyleft 2021
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz Jaramillo
"""

# Format for adding a new distortion 
# in_img is a folder or bgr image 
# lvl is a float 
# out_folder is optional to write the resulting image -> Necessary to create database
# return image is also optional  
# Store in out_folder -> Necessary to create database

import numpy as np
import cv2

# Adds a constant to the brightness of the image when image is in bgr
def brightness(in_img, lvl=0.25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # converting image to YUV in order to modify only the Y component
    img_hsv = cv2.cvtColor(in_img.astype('float32') / 255., cv2.COLOR_BGR2HSV)
    # Adding the constant to the Y component
    img_hsv[::, ::, 2] += lvl
    # Cliping the results and converting back to bgr
    img_hsv[::, ::, 2] = np.clip(img_hsv[::, ::, 2], 0, 1)
    img_out = cv2.cvtColor(img_hsv.astype('float32'), cv2.COLOR_HSV2BGR)
    img_out = (255 * img_out).astype('uint8')

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out
