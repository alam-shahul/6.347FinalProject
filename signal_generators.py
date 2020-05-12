import numpy as np

from scipy.linalg import dft
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct

from numpy.linalg import pinv
from PIL import Image

"""
Signal-generating code below!

Below you will find code for generating random 1D and 2D signals.

"""

def generate_1D_white_noise_signal(dimensionality = 100):
    signal = np.random.uniform(size=dimensionality)

    #plt.plot(signal)

    return signal

def generate_2D_white_noise_signal(dimensionality = 512):
    signal = np.random.uniform(low = 0, high = 255, size=(dimensionality, dimensionality))

    signal *= 255/signal.max()
    signal = signal.astype(np.uint8) 
    #plt.plot(signal)

    return signal

def generate_sinusoidal_signal(dimensionality = 100):
    A = 1
    phi = 0
    n = np.arange(dimensionality)
    y = np.zeros(dimensionality)
    num_sinusoids = 5
    for _ in range(num_sinusoids):
        y += A*np.cos(2*np.pi*np.random.uniform(low=0, high=1)*n + np.random.uniform(low=0, high=2*np.pi))
    #plt.plot(y)
    #plt.show()

    return y

def generate_1D_smoothish_signal(dimensionality=100):
    a1, b1, c1 = np.random.normal(size=3)
    a2, b2, c2 = np.random.normal(size=3)

    first_half=[a1*(x**2) + b1*x+c1 for x in range(0, dimensionality//2)]
    second_half=[a2*(x**2) + b2*x+c2 for x in range(dimensionality//2, dimensionality)]

    return np.concatenate((first_half, second_half))

def generate_2D_smoothish_signal(dimensionality=512):
    first_quadratic = np.poly1d(np.abs(np.random.normal(size=3)))
    second_quadratic = np.poly1d(np.abs(np.random.normal(size=3)))

    first_quadrant = np.zeros((dimensionality//2, dimensionality//2))

    for row in range(dimensionality//2):
        for column in range(dimensionality//2):
            first_quadrant[row, column] = first_quadratic(row) + first_quadratic(column)

    first_quadrant = (255 * first_quadrant / np.max(first_quadrant)).astype(np.uint8)

    smoothish_image = np.zeros((dimensionality, dimensionality))
    for row in range(dimensionality):
        for column in range(dimensionality):
            smoothish_image[row, column] = second_quadratic(row) + second_quadratic(column)
    
    smoothish_image = (255 * smoothish_image / np.max(smoothish_image)).astype(np.uint8)
    smoothish_image[:dimensionality//2, :dimensionality//2] = first_quadrant

    return smoothish_image

def load_bark_image():
    bark_image = Image.open("textures/1.1.02.tiff")

    return np.array(bark_image)
