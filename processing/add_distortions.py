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

import tempfile
import numpy as np
import cv2

# Adds a constant to the brightness of the image when image is in bgr
def brightness(in_img, lvl=0.25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # converting image to YUV in order to modify only the Y component
    img_hsv = cv2.cvtColor(in_img.astype("float32") / 255., cv2.COLOR_BGR2HSV)
    # Adding the constant to the Y component
    img_hsv[::, ::, 2] += lvl
    # Cliping the results and converting back to bgr
    img_hsv[::, ::, 2] = np.clip(img_hsv[::, ::, 2], 0, 1)
    img_out = cv2.cvtColor(img_hsv.astype("float32"), cv2.COLOR_HSV2BGR)
    img_out = (255 * img_out).astype("uint8")

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out

# Adds blur to an image when is in bgr
def blur(in_img, lvl=0.25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # converting image to HSV in order to modify only the V component
    img_hsv = cv2.cvtColor(in_img.astype("float32") / 255., cv2.COLOR_BGR2HSV)
    # Adding blur to the v component
    size = int(7 * lvl + 1) if int(7 * lvl) % 2 == 0 else int(7 * lvl)
    size = np.max((size, 5))
    img_hsv[::, ::, 2] = cv2.GaussianBlur(img_hsv[::, ::, 2], (size, size), lvl)
    # Cliping the results and converting back to bgr
    img_hsv[::, ::, 2] = np.clip(img_hsv[::, ::, 2], 0, 1)
    img_out = cv2.cvtColor(img_hsv.astype("float32"), cv2.COLOR_HSV2BGR)
    img_out = (255 * img_out).astype("uint8")

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out

# Modify contrast image when is in bgr
def contrast(in_img, lvl=0.25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # converting image to HSV in order to modify only the V component
    img_hsv = cv2.cvtColor(in_img.astype("float32") / 255., cv2.COLOR_BGR2HSV)
    # Applying the contrast to the image
    img_hsv[::, ::, 2] *= lvl
    # Cliping the results and converting back to bgr
    img_hsv[::, ::, 2] = np.clip(img_hsv[::, ::, 2], 0, 1)
    img_out = cv2.cvtColor(img_hsv.astype("float32"), cv2.COLOR_HSV2BGR)
    img_out = (255 * img_out).astype("uint8")

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out

# Add Gaussian noise to the image when is in bgr
def gaussian_noise(in_img, lvl=0.25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # converting image to HSV in order to modify only the V component
    img_hsv = cv2.cvtColor(in_img.astype("float32") / 255., cv2.COLOR_BGR2HSV)
    # Applying the contrast to the image
    gauss = np.random.normal(0, lvl, img_hsv[::, ::, 2].shape)
    img_hsv[::, ::, 2] = img_hsv[::, ::, 2] + gauss
    # Cliping the results and converting back to bgr
    img_hsv[::, ::, 2] = np.clip(img_hsv[::, ::, 2], 0, 1)
    img_out = cv2.cvtColor(img_hsv.astype("float32"), cv2.COLOR_HSV2BGR)
    img_out = (255 * img_out).astype("uint8")

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out

# Add compression to the image when is in bgr
def jpeg_compression(in_img, lvl=25, out_folder=None):
    if isinstance(in_img, str):
        in_img_fol = in_img
        in_img = cv2.imread(in_img_fol)
    # Using the opencv write method to compress in jpeg then reading back and storing as png
    file_name = tempfile.gettempdir() + r"\temp_img.jpg"
    cv2.imwrite(file_name, in_img, [cv2.IMWRITE_JPEG_QUALITY, int(lvl)])
    img_out = cv2.imread(file_name).astype("uint8")

    if out_folder is not None:
        cv2.imwrite(out_folder, img_out)

    return img_out
