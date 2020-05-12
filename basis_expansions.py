import numpy as np
from scipy.linalg import dft
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct
import pywt

def generate_gaussian_basis(dimensionality=100, num_frame_vectors=200):
    basis = np.random.normal(size=(dimensionality, dimensionality))
    
    return basis

def generate_dft_basis(dimensionality=100):
    basis = dft(dimensionality)

    return basis

def generate_dft_dropout_frame(dimensionality=100, num_frame_vectors=200):
    """
    Generate a dft_dropout frame using the DFT matrix.
    """

    random_indices = np.random.randint(num_frame_vectors, size=dimensionality)
    frame = dft(num_frame_vectors)[random_indices]

    return frame


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
    (cA, cD) = pywt.dwt(signal, 'db2', 'smooth')
    return np.array((cA, cD))

def debauchies_1D_basis_synthesis(frame_coefficients):
    approximation_coefficients, detail_coefficients = frame_coefficients[0], frame_coefficients[1] 
    signal = pywt.idwt(approximation_coefficients, detail_coefficients, 'db2', 'smooth')
    return signal

def debauchies_2D_basis_analysis(signal):
    (cA, cD) = pywt.dwt2(signal, 'db2', 'smooth')
    return np.array((cA, cD))

def debauchies_2D_basis_synthesis(basis_coefficients):
    signal = pywt.idwt2(basis_coefficients, 'db2', 'smooth')
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


