import numpy as np
from scipy.linalg import dft
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct
import pywt

def generate_gaussian_basis(dimensionality=100):
    basis = np.random.normal(size=(dimensionality, dimensionality))
    
    return basis

def generate_dft_basis(dimensionality=100):
    basis = dft(dimensionality)

    return basis

def multiplicative_basis_analysis(signal, basis):
    analysis_coefficients = np.matmul(signal, basis)

    return analysis_coefficients

def multiplicative_basis_synthesis(analysis_coefficients, basis):
    dual_basis = pinv(basis)
    reconstructed_signal = np.real(np.matmul(basis_coefficients, dual_basis))

    return reconstructed_signal

def stft_basis_analysis(signal):
    _, _, analysis_coefficients = stft(signal)
    return analysis_coefficients.T

def stft_basis_synthesis(analysis_coefficients):
    _, reconstructed_signal = istft(analysis_coefficients.T)
    return reconstructed_signal

def debauchies_1D_basis_analysis(signal):
    analysis_coefficients = pywt.wavedecn(signal, 'db2', 'smooth', level=4)
    array, coeff_slices = pywt.coeffs_to_array(analysis_coefficients)

    return array, coeff_slices

def debauchies_1D_basis_synthesis(analysis_coefficients):
    array, coeff_slices = analysis_coefficients
    analysis_coefficients = pywt.array_to_coeffs(array, coeff_slices, output_format='wavedecn')

    signal = pywt.waverecn(analysis_coefficients, 'db2', 'smooth')
    return signal

def debauchies_2D_basis_analysis(signal):
    analysis_coefficients = pywt.wavedec2(signal, 'db2', 'smooth', level=4)
    array, coeff_slices = pywt.coeffs_to_array(analysis_coefficients)

    return array, coeff_slices

def debauchies_2D_basis_synthesis(analysis_coefficients):
    array, coeff_slices = analysis_coefficients
    analysis_coefficients = pywt.array_to_coeffs(array, coeff_slices, output_format='wavedec2')

    signal = pywt.waverec2(analysis_coefficients, 'db2', 'smooth')
    return signal

# DCT code borrowed from following tutorial: https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
def dct_analysis(signal):
    return dct(dct( signal, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def dct_synthesis(coefficients):
    return idct(idct( coefficients, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def block_dct_analysis(signal, block_size=16):
    coefficients = np.zeros(signal.shape)
    for row in range(0, coefficients.shape[0], block_size):
        for column in range(0, coefficients.shape[1], block_size):
            coefficients[row: row + block_size, column: column + block_size] = \
                dct_analysis(signal[row: row + block_size, column: column + block_size])

    return coefficients

def block_dct_synthesis(coefficients, block_size=16):
    reconstructed_image = np.zeros(coefficients.shape)
    for row in range(0, coefficients.shape[0], block_size):
        for column in range(0, coefficients.shape[1], block_size):
            reconstructed_image[row: row + block_size, column: column + block_size] = \
                dct_synthesis(coefficients[row: row + block_size, column: column + block_size])

    return reconstructed_image

def get_indices_of_k_smallest(arr, k):
    """
    An amazing helper function to get the k smallest indices of an n-dimensional array.

    Pass in -k to get the k largest value indices instead.
    """

    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])


