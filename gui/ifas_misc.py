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

# This file contains helper functions which do not belong to any class

import numpy as np
import cv2

# Creates iFas logo image
def logo_image(size_in):
    # Create a black image
    img = np.zeros((int(size_in[1] - 50), int(size_in[0] / 2 - 25), 3), np.uint8)

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0,255,255)
    font_thickness = 2

    (label_width, label_height), __ = cv2.getTextSize(
        'iFAS: Image fidelity assessment software', font, font_scale, font_thickness
        )
    pos = (int(img.shape[1] / 2 - label_width / 2), int(img.shape[0] / 2 - label_height / 2))

    cv2.putText(
        img, 'iFAS: Image fidelity assessment software', pos, font, font_scale, font_color, 
        font_thickness
        )
    return img