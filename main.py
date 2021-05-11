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

import pathlib
import sys

# Appending the relevant paths to functionalities 

PATH_FILE = pathlib.Path(__file__).parent.absolute()
PATH_MEASURES = PATH_FILE.joinpath("fidelity_measures")
PATH_GUI = PATH_FILE.joinpath("gui")
PATH_IMG_PROC = PATH_FILE.joinpath("processing")
sys.path.append(str(PATH_MEASURES))
sys.path.append(str(PATH_GUI))

from gui.ifas_main_window import AppIFAS

if __name__ == "__main__":
    AppIFAS()
