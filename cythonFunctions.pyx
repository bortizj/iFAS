from __future__ import division
import numpy as np
cimport numpy as np # "cimport" is used to import special compile-time information about the numpy module
DTYPE = np.double # "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For every type in the numpy
# module there's a corresponding compile-time type with a _t-suffix.
ctypedef np.double_t DTYPE_t
ITYPE = np.int
ctypedef np.int_t ITYPE_t
cimport cython

cdef extern from "math.h":
	double fabs(double m)


cdef extern from "math.h":
	double sqrt(double m)


cdef extern from "complex.h":
	double complex cexp(double complex)


@cython.boundscheck(False) # turn off bounds-checking for entire function
def cymin(np.ndarray[DTYPE_t, ndim=1] x):
	cdef int ii
	cdef int M = x.shape[0]
	cdef DTYPE_t mi = 0.
	for ii in range(M):
		if x[ii] < mi:
			mi = x[ii]
	return mi

@cython.boundscheck(False) # turn off bounds-checking for entire function
def cymax(np.ndarray[DTYPE_t, ndim=1] x):
	cdef int ii
	cdef int M = x.shape[0]
	cdef DTYPE_t ma = 0.
	for ii in range(M):
		if x[ii] > ma:
			ma = x[ii]
	return ma


@cython.boundscheck(False) # turn off bounds-checking for entire function
def cymean(np.ndarray[DTYPE_t, ndim=1] x):
	cdef int ii
	cdef int M = x.shape[0]
	cdef DTYPE_t mu = 0.
	for ii in range(M):
		mu += x[ii]
	return mu / M


@cython.boundscheck(False) # turn off bounds-checking for entire function
def mean_above_below_threshold(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t thresh):
	cdef int M, ii, Mbelow, Mabove
	cdef DTYPE_t mean_below, mean_avobe
	cdef np.ndarray[DTYPE_t, ndim=1] means = np.zeros(4, dtype=DTYPE)
	mean_below = 0.
	mean_above = 0.
	Mbelow = 0
	Mabove = 0
	M = x.shape[0]
	for ii in range(M):
		if x[ii] >= thresh:
			mean_above += x[ii]
			Mabove += 1
		else:
			mean_below += x[ii]
			Mbelow += 1
	if Mbelow == 0:
		means[0] = cymin(x)
	else:
		means[0] = mean_below/Mbelow
	if Mabove == 0:
		means[1] = cymax(x)       
	else:	
		means[1] = mean_above/Mabove
	means[2] = Mbelow
	means[3] = Mabove
	return means



@cython.boundscheck(False) # turn off bounds-checking for entire function
def harmonicmean_above_below_threshold(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t thresh):
	cdef int M, ii, Mbelow, Mabove
	cdef DTYPE_t mean_below, mean_avobe
	cdef np.ndarray[DTYPE_t, ndim=1] means = np.zeros(4, dtype=DTYPE)
	mean_below = 0.
	mean_above = 0.
	Mbelow = 0
	Mabove = 0
	M = x.shape[0]
	for ii in range(M):
		if x[ii] >= thresh and x[ii] != 0:
			mean_above += 1./x[ii]
			Mabove += 1
		elif x[ii] < thresh and x[ii] != 0:
			mean_below += 1./x[ii]
			Mbelow += 1
	if Mbelow == 0:
		means[0] = cymin(x)
	else:
		means[0] = 1./(mean_below/Mbelow)
	if Mabove == 0:
		means[1] = cymax(x)       
	else:	
		means[1] = 1./(mean_above/Mabove)
	means[2] = Mbelow
	means[3] = Mabove
	return means



@cython.boundscheck(False) # turn off bounds-checking for entire function
def isodata_cython(np.ndarray[DTYPE_t, ndim=1] x, ITYPE_t W):
	cdef DTYPE_t precision=1.e-6	
	cdef DTYPE_t mean_below, mean_avobe
	cdef np.ndarray[DTYPE_t, ndim=1] means = np.zeros(4, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] retmean = np.zeros(2, dtype=DTYPE)
	cdef DTYPE_t T = cymean(x)
	cdef DTYPE_t Tprev = 0.
	while fabs(T - Tprev) > precision:
		means = mean_above_below_threshold(x,T)
		Tprev = T
		T = (means[0]+means[1])/2.
	means = harmonicmean_above_below_threshold(x,T)
	if means[2] < W:
		retmean[0] = -1.0
		retmean[1] = -1.0
		return retmean
	else:
		retmean[0] = means[0]
	if means[3] < W:
		retmean[0] = -1.0
		retmean[1] = -1.0
		return retmean
	else:
		retmean[1] = means[1]
	return retmean

@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_contrast_bcontent_measure_simple(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[ITYPE_t, ndim=1] g, ITYPE_t W, ITYPE_t overlap):
	if g[0] % 2 != 1 or g[1] % 2 != 1:
		raise ValueError("Only odd dimensions on shape supported")
	assert f.dtype == DTYPE and g.dtype == ITYPE
	#declaring C type variables
	cdef int M = f.shape[0]
	cdef int N = f.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef int sizebM = g[0]
	cdef int sizebN = g[1]
	cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(sizebM*sizebN, dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int ii, jj, ss, tt
	cdef int count
	cdef np.ndarray[DTYPE_t, ndim=1] local_means = np.zeros(2, dtype=DTYPE)
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	cdef int ii_from, ii_to, jj_from, jj_to
	for ii in range(bMmid,M_lims,overlap):
		for jj in range(bNmid,N_lims,overlap):
			ii_from = ii-bMmid
			ii_to = ii+bMmid
			jj_from = jj-bNmid
			jj_to = jj+bNmid
			count = 0
			for ss in range(ii_from, ii_to):
				for tt in range(jj_from, jj_to):
					values[count] = f[ss, tt]
					count += 1
			local_means = isodata_cython(values,W)
			if local_means[0] > 0 and local_means[1] > 0 :
				C[ii, jj] = local_means[0] / local_means[1]
			else:
				C[ii, jj] = 0.
	return C


@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_contrast_bcontent_measure_weber(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[ITYPE_t, ndim=1] g, ITYPE_t W, ITYPE_t overlap):
	if g[0] % 2 != 1 or g[1] % 2 != 1:
		raise ValueError("Only odd dimensions on shape supported")
	assert f.dtype == DTYPE and g.dtype == ITYPE
	#declaring C type variables
	cdef int M = f.shape[0]
	cdef int N = f.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef int sizebM = g[0]
	cdef int sizebN = g[1]
	cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(sizebM*sizebN, dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int ii, jj, ss, tt
	cdef int count
	cdef np.ndarray[DTYPE_t, ndim=1] local_means = np.zeros(2, dtype=DTYPE)
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	cdef int ii_from, ii_to, jj_from, jj_to
	for ii in range(bMmid,M_lims,overlap):
		for jj in range(bNmid,N_lims,overlap):
			ii_from = ii-bMmid
			ii_to = ii+bMmid
			jj_from = jj-bNmid
			jj_to = jj+bNmid
			count = 0
			for ss in range(ii_from, ii_to):
				for tt in range(jj_from, jj_to):
					values[count] = f[ss, tt]
					count += 1
			local_means = isodata_cython(values,W)
			if local_means[0] > 0 and local_means[1] > 0:
				C[ii, jj] = 1. - local_means[0] / local_means[1]
			else:
				C[ii, jj] = 0.
	return C


@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_contrast_bcontent_measure_michelson(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[ITYPE_t, ndim=1] g, ITYPE_t W, ITYPE_t overlap):
	if g[0] % 2 != 1 or g[1] % 2 != 1:
		raise ValueError("Only odd dimensions on shape supported")
	assert f.dtype == DTYPE and g.dtype == ITYPE
	#declaring C type variables
	cdef int M = f.shape[0]
	cdef int N = f.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef int sizebM = g[0]
	cdef int sizebN = g[1]
	cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(sizebM*sizebN, dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int ii, jj, ss, tt
	cdef int count
	cdef np.ndarray[DTYPE_t, ndim=1] local_means = np.zeros(2, dtype=DTYPE)
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	cdef int ii_from, ii_to, jj_from, jj_to
	for ii in range(bMmid,M_lims,overlap):
		for jj in range(bNmid,N_lims,overlap):
			ii_from = ii-bMmid
			ii_to = ii+bMmid
			jj_from = jj-bNmid
			jj_to = jj+bNmid
			count = 0
			for ss in range(ii_from, ii_to):
				for tt in range(jj_from, jj_to):
					values[count] = f[ss, tt]
					count += 1
			local_means = isodata_cython(values,W)
			if local_means[0] > 0 and local_means[1] > 0:
				C[ii, jj] = (local_means[1]-local_means[0]) / (local_means[1]+local_means[0])
			else:
				C[ii, jj] = 0.
	return C


@cython.boundscheck(False) # turn off bounds-checking for entire function
def pseudo_wigner(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] hs, np.ndarray[DTYPE_t, ndim=2] hf, ITYPE_t W):
	cdef int M = X.shape[0]
	cdef int N = X.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=4] PWD = np.zeros([M, N, W, W], dtype=DTYPE)
	cdef DTYPE_t current_value, magnitude
	cdef double complex current_value_c
	cdef int sizebM = W
	cdef int sizebN = W
	cdef np.ndarray[DTYPE_t, ndim=2] values = np.zeros([sizebM, sizebN], dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int xx, yy, ii, jj, kk, ll
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	for xx in range(bMmid,M_lims):
		for yy in range(bNmid,N_lims):
			current_value = 0.
			for uu in range(-bMmid,bMmid+1):
				for vv in range(-bMmid,bMmid+1):
					for ii in range(-bMmid,bMmid+1):
						for jj in range(-bNmid,bNmid+1):
							for kk in range(-bMmid,bMmid+1):						
								for ll in range(-bNmid,bNmid+1):
									current_value_c = cexp(-2.*1j*(ii*uu+jj*vv))
									magnitude = sqrt(current_value_c.real*current_value_c.real+current_value_c.imag*current_value_c.imag)
									current_value += hs[ii+bMmid,jj+bNmid]*hf[kk+bMmid,ll+bNmid]*X[xx+kk+ii,yy+ll+jj]*X[xx+kk-ii,yy+ll-jj]*magnitude
					PWD[xx,yy,uu+bMmid,vv+bNmid] = current_value
	return PWD


@cython.boundscheck(False) # turn off bounds-checking for entire function
def crm_convolution(np.ndarray[DTYPE_t, ndim=2] X, ITYPE_t W):
	cdef int M = X.shape[0]
	cdef int N = X.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef DTYPE_t current_value
	cdef int sizebM = W
	cdef int sizebN = W
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - sizebM
	cdef int N_lims = N - sizebN
	cdef int xx, yy, ii, jj
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	for xx in range(bMmid,M_lims):
		for yy in range(bNmid,N_lims):
			current_value = 0.
			for ii in range(-bMmid,bMmid+1):
				for jj in range(-bNmid,bNmid+1):
					current_value += fabs(X[xx,yy]-X[xx+ii,yy+jj])
			C[xx,yy] = current_value / (sizebM * sizebN)
	return C



@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_contrast_bcontent_measure_weber_overlaping(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[ITYPE_t, ndim=1] g, ITYPE_t W):
	if g[0] % 2 != 1 or g[1] % 2 != 1:
		raise ValueError("Only odd dimensions on shape supported")
	assert f.dtype == DTYPE and g.dtype == ITYPE
	#declaring C type variables
	cdef int M = f.shape[0]
	cdef int N = f.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef int sizebM = g[0]
	cdef int sizebN = g[1]
	cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(sizebM*sizebN, dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int ii, jj, ss, tt
	cdef int count
	cdef np.ndarray[DTYPE_t, ndim=1] local_means = np.zeros(2, dtype=DTYPE)
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	cdef int ii_from, ii_to, jj_from, jj_to
	for ii in range(bMmid,M_lims):
		for jj in range(bNmid,N_lims):
			ii_from = ii-bMmid
			ii_to = ii+bMmid
			jj_from = jj-bNmid
			jj_to = jj+bNmid
			count = 0
			for ss in range(ii_from, ii_to):
				for tt in range(jj_from, jj_to):
					values[count] = f[ss, tt]
					count += 1
			local_means = isodata_cython(values,W)
			if local_means[0] > 0 and local_means[1] > 0:
				C[ii, jj] = 1. - local_means[0] / local_means[1]
			else:
				C[ii, jj] = 0.
	return C


@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_contrast_bcontent_measure_michelson_overlaping(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[ITYPE_t, ndim=1] g, ITYPE_t W):
	if g[0] % 2 != 1 or g[1] % 2 != 1:
		raise ValueError("Only odd dimensions on shape supported")
	assert f.dtype == DTYPE and g.dtype == ITYPE
	#declaring C type variables
	cdef int M = f.shape[0]
	cdef int N = f.shape[1]
	cdef np.ndarray[DTYPE_t, ndim=2] C = np.zeros([M, N], dtype=DTYPE)
	cdef int sizebM = g[0]
	cdef int sizebN = g[1]
	cdef np.ndarray[DTYPE_t, ndim=1] values = np.zeros(sizebM*sizebN, dtype=DTYPE)
	cdef int bMmid = sizebM // 2
	cdef int bNmid = sizebN // 2
	cdef int M_lims = M - bMmid
	cdef int N_lims = N - bNmid
	cdef int ii, jj, ss, tt
	cdef int count
	cdef np.ndarray[DTYPE_t, ndim=1] local_means = np.zeros(2, dtype=DTYPE)
	# It is very important to type ALL your variables. You do not get any
	# warnings if not, only much slower code (they are implicitly typed as
	# Python objects).
	cdef int ii_from, ii_to, jj_from, jj_to
	for ii in range(bMmid,M_lims):
		for jj in range(bNmid,N_lims):
			ii_from = ii-bMmid
			ii_to = ii+bMmid
			jj_from = jj-bNmid
			jj_to = jj+bNmid
			count = 0
			for ss in range(ii_from, ii_to):
				for tt in range(jj_from, jj_to):
					values[count] = f[ss, tt]
					count += 1
			local_means = isodata_cython(values,W)
			if local_means[0] > 0 and local_means[1] > 0:
				C[ii, jj] = (local_means[1]-local_means[0]) / (local_means[1]+local_means[0])
			else:
				C[ii, jj] = 0.
	return C
