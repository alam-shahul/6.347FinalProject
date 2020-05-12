# Code for 6.347 Project I
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex = True)
import copy

from scipy.linalg import dft
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct

from numpy.linalg import pinv
import pywt
from signal_generators import *
from basis_expansions import *

def sparse_approximation_for_1D_signals():
    gaussian_basis = generate_gaussian_frame(dimensionality=dimensionality, num_frame_vectors=num_frame_vectors)
    dft_basis = generate_dft_basis(dimensionality=dimensionality)

    signal_generators = {
        "smoothish": generate_smoothish_signal,
        "white_noise": generate_white_noise_signal,
        "sinusoidal": generate_sinusoidal_signal
    }

    multiplicative_frames = {
        "harmonic": h,
        "gaussian": gaussian_basis
    }

def sparse_approximation_for_2D_signals():
    """
    2D experiments for Part I of Project II.
    """

    signal_generators = {
        "actual": load_bark_image,
        "smoothish": generate_2D_smoothish_signal,
        "white_noise": generate_2D_white_noise_signal,
    }

    for signal_type in signal_generators:
        signal = signal_generators[signal_type]()
    
    analysis_functions = {
        "dct": dct_analysis,
        "block_dct": block_dct_analysis,
        "debauchies_wavelet": debauchies_2D_basis_analysis
    }     
    
    synthesis_functions = {
        "dct": dct_synthesis,
        "block_dct": block_dct_synthesis,
        "debauchies_wavelet": debauchies_2D_basis_synthesis
    }     
    
    # Experiments
    for signal_type in signal_generators:
        signal = signal_generators[signal_type]()

        image = Image.fromarray(signal)
        image.save("results/%s_signal.png" % (signal_type))
        
        # Sparse approximation
        print("Sparse approximation for signal type %s" % signal_type)
        for analysis_type in analysis_functions:
            print("Analysis type %s" % analysis_type)
            analysis = analysis_functions[analysis_type]

            coefficients = analysis(signal)
            synthesis = synthesis_functions[analysis_type]
            
            # Sparse approximation error calculation
            proportions = np.linspace(0.1, 1, 9, endpoint=False)
            errors = np.zeros(proportions.shape)
            for index, proportion in enumerate(proportions):
                if analysis_type == "debauchies_wavelet":
                    truncated_coefficients = copy.deepcopy(coefficients)
                    k = int(coefficients[0].size * proportion)
                    approximation_erasure_coordinates = get_indices_of_k_smallest(coefficients[0], k)
                    truncated_coefficients[0][approximation_erasure_coordinates] = 0
                    errors[index] += np.linalg.norm(coefficients[0] - truncated_coefficients[0])
                    for detail_axis in range(len(coefficients[1])):
                        k = int(coefficients[1][detail_axis].size * proportion)
                        detail_erasure_coordinates = get_indices_of_k_smallest(coefficients[1][detail_axis], k)
                        truncated_coefficients[1][detail_axis][detail_erasure_coordinates] = 0
                        errors[index] += np.linalg.norm(coefficients[1][detail_axis] - truncated_coefficients[1][detail_axis])
                else:
                    k = int(coefficients.size * proportion)
                    erasure_coordinates = get_indices_of_k_smallest(coefficients, k)
                    truncated_coefficients = copy.deepcopy(coefficients)
                    truncated_coefficients[erasure_coordinates] = 0
                    errors[index] = np.linalg.norm(coefficients - truncated_coefficients)
                
                # Plot recovered signal for 50% truncation
                if proportion == 0.5:
                    print("Saving reconstructed signal")
                    reconstructed_signal = synthesis(truncated_coefficients)
       
                    reconstructed_signal *= (255/reconstructed_signal.max())

                    reconstructed_image = Image.fromarray(reconstructed_signal.astype(np.uint8))
                    reconstructed_image.save("results/%s_reconstructed_signal_with_%s_analysis.png" % (signal_type, analysis_type))

            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(proportions, errors)
            ax.set_title("%s signal with %s analysis" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_with_%s_analysis.png" % (signal_type, analysis_type))


def sparse_approximation_with_quantization():
    
    signal_generators = {
        "smoothish": generate_2D_smoothish_signal,
        "actual": load_bark_image
    }

    analysis_functions = {
        "block_dct": block_dct_analysis,
        "debauchies_wavelet": debauchies_2D_basis_analysis
    }     
    
    synthesis_functions = {
        "block_dct": block_dct_synthesis,
        "debauchies_wavelet": debauchies_2D_basis_synthesis
    }
     
    # Approach I
    for signal_type in signal_generators:
        signal = signal_generators[signal_type]()

        image = Image.fromarray(signal)
        image.save("results/%s_signal.png" % (signal_type))
        # Sparse approximation
        print("Sparse approximation for signal type %s" % signal_type)
        for analysis_type in analysis_functions:
            print("Analysis type %s" % analysis_type)
            analysis = analysis_functions[analysis_type]

            coefficients = analysis(signal)
            synthesis = synthesis_functions[analysis_type]
            
            bit_rates = np.linspace(1, 20, 20)
            distortions = np.zeros(bit_rates.shape)
            
            # Quantization and distortion calculation
            if analysis_type == "debauchies_wavelet":
                saturation_point = np.max(np.abs(coefficients[0]))
                for detail_axis in range(len(coefficients[1])):
                    saturation_point = max(saturation_point, np.max(np.abs(coefficients[1][detail_axis])))
                for index, bit_rate in enumerate(bit_rates):
                    quantum = saturation_point / 2**(bit_rate - 1)
                    quantized_coefficients = copy.deepcopy(coefficients)
                    
                    quantized_approximation_coefficients = np.round(coefficients[0] / quantum) * quantum
                    quantized_coefficients[0] = quantized_approximation_coefficients
                    distortions[index] = np.max(np.abs(coefficients[0] - quantized_coefficients[0]))
                    
                    quantized_detail_coefficients = []
                    for detail_axis in range(len(coefficients[1])):
                        quantized_detail_coefficients.append(np.round(coefficients[1][detail_axis] / quantum) * quantum)
                        distortions[index] = max(distortions[index], np.max(np.abs(coefficients[1][detail_axis] - quantized_detail_coefficients[detail_axis])))

                    quantized_coefficients[1] = tuple(quantized_detail_coefficients)

                rates = bit_rates * (coefficients[0].size + sum(coefficients[1][detail_axis].size for detail_axis in range(len(coefficients[1]))))
            else:
                saturation_point = np.max(np.abs(coefficients))
                for index, bit_rate in enumerate(bit_rates):
                    quantum = saturation_point / 2**(bit_rate - 1)
                    quantized_coefficients = np.round(coefficients / quantum) * quantum
                    distortion = np.max(np.abs(coefficients - quantized_coefficients))
                    distortions[index] = distortion

                rates = bit_rates * coefficients.size

            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(rates, distortions)
            ax.set_title("D(R) for %s signal with %s analysis" % (signal_type, analysis_type))
            #fig.savefig("results/%s_signal_with_%s_analysis.png" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_distortion_rate_with_%s_analysis.png" % (signal_type, analysis_type))


if __name__ == "__main__":
    #sparse_approximation_for_2D_signals()
    sparse_approximation_with_quantization()
