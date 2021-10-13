![This is an image](/iFAS/docs/ifas_cover.png)

# Software description
--------------------
The image Fidelity Assessment Software (iFAS) is a software tool designed 
to assist researchers, engineers and other users in the process of image 
fidelity assessment.
 
iFAS provides easy access to a range of state-of-the-art methods as well 
as intuitive visualizations that aid data analysis. Also, iFAS offers a 
wide variety of image fidelity measures: five miscellaneous measures, 
twelve texture-based measures, five fidelity measures based on contrast 
and twenty-three image color difference measures. The software is freely 
available to all for non-commercial use. 

iFAS includes a range of common mechanisms for image fidelity assessment 
including computation of fidelity measures on a single pair of images and/or 
in a full database, visualization of pixel-wise image differences and 
histogram of the image differences, scatter plots and correlation analysis 
between human scores and objective measures.

More information can be found in the [documentation](/iFAS/docs/iFAS_user_guide.pdf)

# Configuration
Python virtual environment configuration including pip, scipy, pandas, tk, matplotlib, opencv, pywavelets
Install [Anaconda](https://www.anaconda.com/) using the default settings
To create environment, open anaconda prompt and execute
```
conda create --name ifas_venv python=3.8 pip scipy pandas tk matplotlib pywavelets
conda activate ifas_venv
conda install -c anaconda opencv
pip install -e /path/to/iFAS/folder
cd /path/to/iFAS/folder
python main.py
```

# Measures

Currently iFAS includes the following list of fidelity measures: 
1.	Five miscellaneous measures: (1) Peak signal to noise ratio (PSNR), (2) Structural similarity index measure (SSIM), (3) Wavelet domain noise difference, (4) variance of Laplacian blur difference, (5) Edge Peak signal to noise ratio (EPSNR)
2.	Twelve texture-based measures: (1) autoregressive model difference, (2) Gaussian Markov random fields difference, (3) grey level cooccurrence matrix difference, (4) acf2d_dif, (5) local binary patterns difference, (6) Laws filter bank difference, (7) Eigen-filter bank difference, (8) wedge and ring filter bank based difference, (9) Gabor filter bank based difference, (10) Laplacian filter bank based differences, (11) steerable filter bank based differences, (12) Granulometry moments difference
3.	Five fidelity measures based on contrast: (1) Simple measure of enhancement, (2) Weber’s measure of enhancement, (3) Michelson’s measure of enhancement, (4) Root mean squared measure of enhancement, (5) Peli’s measure of enhancement
4.	Twenty-three image color difference measures: (1) adaptive image difference, (2) chroma spread and extreme, (3) color image appearance difference, (4) circular hue, (5) color image difference, (6)  color Mahalanobis distance, (7) color SSIM, (8) colorfulness difference, (9) color histogram difference, (10) cpsnrha, (11) delta e, (12) delta e2000, (13) hue and saturation difference, (14) just noticeable delta e, (15) local delta e2000, (16) osa ucs delta e, (17) osa ucs spatial delta e, (18) shame cielab, (19) spatial delta e2000, (20) SSIM ipt, (21) texture patched color difference, (22) visual saliency index, (23) weighted delta e.
The fidelity measures are Python functions which have two inputs and one output. The inputs are two BGR images of the same size (reference and test image) to be compared by the fidelity measure and the output is a float representing the differences between the two images. It is possible to add new fidelity measures to iFAS. To do so it is necessary to add a new Python script file `.py` under the folder `idelity_measures`. 

# Statistics

iFAS currently includes four correlation matrices: Pearson, Spearman, tau and correlation of distances.

# Plotting 

iFAS has the following functionalities: 
1. Scatter plot of the data with one of the measures in the x-axis and another measure in the y-axis (the axes are labelled correspondingly to the measures). The plot can be controlled using the left, right, up and down keys in order to change between the different measures.
2. Bar plot of the correlations of each measure against the target variable selected in the Modeling pane.
3. Box plot of the correlation against the target variable for each fidelity measure. Each box represents 
the correlation distribution for each measure computed for each reference independently.
4. Scatter plot and regression line between the available fidelity measures and the target variable selected in the Modeling pane. 

# Modeling 

iFAS contains the settings for creating models between a target variable and the given fidelity measures. It is possible to select between six different regression models in the drop box list Model: (1) linear, (2) quadratic, (3) cubic, (4) exponential, (5) logistic, (6) complementary error. The ini parameters expects the following number of float values separated by coma: 
1. Two parameters for the linear model: `y=a_0+a_1 x, e.g., 0.5,-0.5`
2. Three parameters for the quadratic model: `y=a_0+a_1 x+a_1 x^2, e.g., -0.5,1,-0.5`
3. Four parameters for the cubic model: `y=a_0+a_1 x+a_2 x^2+a_3 x^3, e.g., -0.5,1,-1,0.5`
4. Three parameters for the exponential model: `y=a_0+exp⁡(a_1 x+a_2), e.g., 0.5,-0.5,1`
5. Three parameters for the logistic model: `y=a_0/(1+exp⁡(-〖(a〗_1 x+a_2))), e.g., 0.5,-0.5,1`
6. Two parameters for the complementary error model: `y=0.5-0.5erf⁡((x-a_0)/(√2 a_1 )), e.g., 0.5,-0.5`

# Copyleft

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [gnu](https://www.gnu.org/licenses/).

# References

If you use any of this code please remember to reference the related publications:

B. Ortiz-Jaramillo, et.al., "Evaluation of color differences in natural scene 
color images," Signal Processing: Image Communication, 2019
[link](https://www.sciencedirect.com/science/article/abs/pii/S0923596518301863).

B. Ortiz-Jaramillo, et.al., "Content-aware contrast ratio measure for images," 
Signal Processing: Image Communication, 2018
[link](https://www.sciencedirect.com/science/article/abs/pii/S0923596517302606).

B. Ortiz-Jaramillo, et.al., "iFAS: Image Fidelity Assessment," Proc. International 
Workshop on Computational Color Imaging, 2017
[link](https://link.springer.com/chapter/10.1007/978-3-319-56010-6_7).

B. Ortiz-Jaramillo, et.al., "Reviewing, selecting and evaluating features in 
distinguishing fine changes of global texture," Pattern Analysis and Applications, 2014
[link](https://link.springer.com/article/10.1007/s10044-013-0352-8).
