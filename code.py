# Code for 6.347 Final Project
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex = True)
import copy

from scipy.linalg import dft
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct
from scipy.special import comb
import math

from numpy.linalg import pinv
import pywt
from signal_generators import *
from basis_expansions import *

def sparse_approximation_for_1D_signals():
    gaussian_basis = generate_gaussian_basis(dimensionality=dimensionality)
    dft_basis = generate_dft_basis(dimensionality=dimensionality)

    signal_generators = {
        "smoothish": generate_smoothish_signal,
        "white_noise": generate_white_noise_signal,
        "sinusoidal": generate_sinusoidal_signal
    }

    multiplicative_frames = {
        "dft_basis": dft_basis,
        "gaussian": gaussian_basis
    }
    
    analysis_functions = {
        "gaussian": lambda signal, basis=gaussian_frame: multiplicative_basis_analysis(signal, basis),
        "stft": stft_basis_analysis,
        "dft": lambda signal, basis=dft_basis: multiplicative_basis_analysis(signal, basis),
        "debauchies": debauchies_1D_basis_analysis
    }

    synthesis_functions = {
        "gaussian": lambda basis_coefficients, basis=gaussian_basis: multiplicative_basis_synthesis(basis_coefficients, basis),
        "stft": stft_frame_synthesis,
        "dft": lambda basis_coefficients, basis=dft_basis: multiplicative_basis_synthesis(basis_coefficients, basis),
        "debauchies": debauchies_1D_basis_synthesis
    }

def sparse_approximation_for_2D_signals():
    """
    2D experiments for Part I of Project II.
    """

    signal_generators = {
        "actual": load_image,
        "smoothish": generate_2D_smoothish_signal,
        "white_noise": generate_2D_white_noise_signal,
    }

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
            proportions = np.linspace(0.05, 1, 19, endpoint=False)
            errors = np.zeros(proportions.shape)
            for index, proportion in enumerate(proportions):
                if analysis_type == "debauchies_wavelet":
                    array_coefficients, coeff_slices = coefficients
                    k = int(array_coefficients.size * proportion)
                    erasure_coordinates = get_indices_of_k_smallest(np.abs(array_coefficients), k)
                    truncated_array_coefficients = copy.deepcopy(array_coefficients)
                    truncated_array_coefficients[erasure_coordinates] = 0
                    errors[index] = np.linalg.norm(array_coefficients - truncated_array_coefficients)
                    truncated_coefficients = (truncated_array_coefficients, coeff_slices)
                else:
                    k = int(coefficients.size * proportion)
                    erasure_coordinates = get_indices_of_k_smallest(np.abs(coefficients), k)
                    truncated_coefficients = copy.deepcopy(coefficients)
                    truncated_coefficients[erasure_coordinates] = 0
                    errors[index] = np.linalg.norm(coefficients - truncated_coefficients)
                
                # Plot recovered signal for 50%, 90% truncation
                if proportion in (0.9, 0.95):
                    print("Saving reconstructed signal")
                    reconstructed_signal = synthesis(truncated_coefficients)
       
                    reconstructed_signal *= (255/reconstructed_signal.max())

                    reconstructed_image = Image.fromarray(reconstructed_signal.astype(np.uint8))
                    reconstructed_image.save("results/%s_reconstructed_signal_with_%s_truncation_and_%s_analysis.png" % (signal_type, proportion, analysis_type))

            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(proportions, errors)
            ax.set_title("%s signal with %s analysis" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_with_%s_analysis.png" % (signal_type, analysis_type))


def sparse_approximation_with_quantization():
    """
    2D experiments for Part II of Final Project
    """
    
    signal_generators = {
        "smoothish": generate_2D_smoothish_signal,
        "actual": load_image
    }

    analysis_functions = {
        "block_dct": block_dct_analysis,
        "debauchies_wavelet": debauchies_2D_basis_analysis
    }     
    
    synthesis_functions = {
        "block_dct": block_dct_synthesis,
        "debauchies_wavelet": debauchies_2D_basis_synthesis
    }
     
    # Approach 1
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
            
            bit_rates = np.linspace(6, 20, 15)
            distortions = np.zeros(bit_rates.shape)
            rates = np.zeros(bit_rates.shape)
            
            # Quantization and distortion calculation
            if analysis_type == "debauchies_wavelet":
                array_coefficients, coeff_slices = coefficients
                saturation_point = np.max(np.abs(array_coefficients))
                for index, bit_rate in enumerate(bit_rates):
                    quantum = saturation_point / 2**(bit_rate - 1)
                    quantized_array_coefficients = np.round(array_coefficients / quantum) * quantum
                    distortion = np.max(np.abs(array_coefficients - quantized_array_coefficients))
                    distortions[index] = distortion

                quantized_coefficients = (quantized_array_coefficients, coeff_slices)

                rates = bit_rates * array_coefficients.size
            else:
                saturation_point = np.max(np.abs(coefficients))
                for index, bit_rate in enumerate(bit_rates):
                    quantum = saturation_point / 2**(bit_rate - 1)
                    quantized_coefficients = np.round(coefficients / quantum) * quantum
                    distortion = np.max(np.abs(coefficients - quantized_coefficients))
                    distortions[index] = distortion

                rates = bit_rates * coefficients.size

            # Approach 2 - note that these values would take a while to calculate due to complexity of comb function, so instead
            # we estimate log2(N c k) = O(k*log2(N/k))
            proportions = np.linspace(0.1, 1, 9, endpoint=False)
            alternative_distortions = np.zeros((proportions.size, bit_rates.size))
            alternative_rates = np.zeros((proportions.size, bit_rates.size))
            if analysis_type == "debauchies_wavelet":
                for proportion_index, proportion in enumerate(proportions):
                    array_coefficients, coeff_slices = coefficients

                    # k-term approximation
                    k = int(array_coefficients.size * proportion)
                    nonzero_coordinates = get_indices_of_k_smallest(np.abs(array_coefficients), -k) # -k => k largest
                    truncated_array_coefficients = np.zeros(array_coefficients.shape)
                    truncated_array_coefficients[nonzero_coordinates] = array_coefficients[nonzero_coordinates]
                    saturation_point = np.max(np.abs(truncated_array_coefficients))
                    for bit_index, bit_rate in enumerate(bit_rates):
                        # quantize using uniform scalar quantizer as before
                        quantum = saturation_point / 2**(bit_rate - 1)
                        quantized_array_coefficients = np.round(truncated_array_coefficients / quantum) * quantum # this is inefficient because of all the zeros
                        alternative_distortion = np.max(np.abs(truncated_array_coefficients - quantized_array_coefficients))
                        alternative_distortions[proportion_index, bit_index] = alternative_distortion

                        alternative_rates[proportion_index, bit_index] = bit_rate * k + k * math.log2(array_coefficients.size/k)

                        coefficients = (quantized_array_coefficients, coeff_slices)
            else:
                for proportion_index, proportion in enumerate(proportions):
                    # k-term approximation
                    k = int(coefficients.size * proportion)
                    nonzero_coordinates = get_indices_of_k_smallest(np.abs(coefficients), -k) # -k => k largest
                    truncated_coefficients = np.zeros(coefficients.shape)
                    truncated_coefficients[nonzero_coordinates] = coefficients[nonzero_coordinates]
                    saturation_point = np.max(np.abs(truncated_coefficients))
                    for bit_index, bit_rate in enumerate(bit_rates):
                        # quantize using uniform scalar quantizer as before
                        quantum = saturation_point / 2**(bit_rate - 1)
                        quantized_coefficients = np.round(truncated_coefficients / quantum) * quantum # this is inefficient because of all the zeros
                        alternative_distortion = np.max(np.abs(truncated_coefficients - quantized_coefficients))
                        alternative_distortions[proportion_index, bit_index] = alternative_distortion

                        alternative_rates[proportion_index, bit_index] = bit_rate * k + k * math.log2(coefficients.size/k)

            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(rates, distortions)

            flattened_alternative_rates = alternative_rates.flatten()
            flattened_alternative_distortions = alternative_distortions.flatten()

            # Unique colors for scatterplot points
            rgb_cycle = np.vstack((                          # Three sinusoids
                .5*(1.+np.cos(flattened_alternative_rates          )), # scaled to [0,1]
                .5*(1.+np.cos(flattened_alternative_rates+2*np.pi/3)), # 120° phase shifted.
                .5*(1.+np.cos(flattened_alternative_rates-2*np.pi/3)))).T # Shape = (60,3)
            ax.scatter(alternative_rates.flatten(), alternative_distortions.flatten(), c=rgb_cycle)
            
            ax.set_title("D(R) for %s signal with %s analysis" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_distortion_rate_with_%s_analysis.png" % (signal_type, analysis_type))

def sparse_compressed_sensing_for_2D_signals():
    """
    2D experiments for Part III of Final Project
    """

    signal_generators = {
        #"actual": load_image,
        "smoothish": generate_2D_smoothish_signal,
        "white_noise": generate_2D_white_noise_signal,
    }
    
    analysis_functions = {
        "block_dct": block_dct_analysis,
    }     

    synthesis_functions = {
        "block_dct": block_dct_synthesis,
    }

    for signal_type in signal_generators:
        signal = signal_generators[signal_type]()
        signal = signal[:signal.shape[0]//4, :signal.shape[1]//4] # downsample because we have to

        image = Image.fromarray(signal)
        image.save("results/%s_signal.png" % (signal_type))

        # Compressed sensing
        print("Compressed sensing for signal type %s" % signal_type)
        for analysis_type in analysis_functions:
            print("Analysis type %s" % analysis_type)
            analysis = analysis_functions[analysis_type]
            synthesis = synthesis_functions[analysis_type]

            original_shape = signal.shape # assume signal is N x N
            dimensionality = signal.size

            flattened_image = signal.T.flatten().T

            # Varying M
            num_measurement_choices = list(range(dimensionality//10, dimensionality//2, dimensionality//10))

            sparsity_penalty = 10

            step_size = 1

            def ista(initial_estimate, measurement, measurement_matrix, analysis, synthesis, step_size=1, sparsity_penalty=10, num_iterations=1000):
                """
                ISTA implementation for DCT (although probably works for other bases as well)
                """

                estimate = initial_estimate
                for iteration in range(num_iterations): 
                    gradient = (measurement_matrix.T @ ((measurement_matrix @ estimate.T.flatten().T) - measurement)).T.reshape(original_shape).T

                    gradient_step = step_size * gradient

                    new_estimate = synthesis(pywt.threshold(analysis(estimate - gradient_step), step_size * sparsity_penalty))

                    loss = np.linalg.norm(new_estimate - estimate, ord='fro') # loss is just l2 norm over all entries

                    print("Iteration %d: Loss = %f" % (iteration, loss))

                    estimate = new_estimate

                return estimate, loss

            reconstruction_errors = np.zeros(len(num_measurement_choices))
            for index, num_measurements in enumerate(num_measurement_choices):
                print("Number of measurements: %s" % num_measurements)
                initial_estimate = np.random.normal(size=original_shape) # initial guess (just random analysis)
                
                print("Generating random measurement matrix of shape (%s, %s) ..." % (num_measurements, dimensionality))
                measurement_matrix = np.random.normal(size=(num_measurements, dimensionality))
                
                print("Normalizing measurement matrix (calculating SVD)...")
                measurement_matrix /= np.linalg.norm(measurement_matrix, ord=2) # normalize using 2-norm

                measurement = measurement_matrix @ flattened_image
               
                estimate, reconstruction_error = ista(initial_estimate, measurement, measurement_matrix, analysis, synthesis)

                reconstruction_errors[index] = reconstruction_error

                reconstructed_signal = estimate
                reconstructed_image = Image.fromarray(reconstructed_signal.astype(np.uint8))
                reconstructed_image.save("results/reconstructed_%s_signal_with_%s_analysis_and_%s_measurements.png" % (signal_type, analysis_type, num_measurements))
                
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(num_measurement_choices, reconstruction_errors)
            ax.set_title("M vs. reconstruction_error for %s signal with %s analysis" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_with_%s_analysis_recovery_error" % (signal_type, analysis_type))


         # Effect of sparsity penalty
        print("Effect of sparsity penalty for signal type %s" % signal_type)
        for analysis_type in analysis_functions:
            print("Analysis type %s" % analysis_type)
            analysis = analysis_functions[analysis_type]
            synthesis = synthesis_functions[analysis_type]

            original_shape = signal.shape # assume signal is N x N
            dimensionality = signal.size

            flattened_image = signal.T.flatten().T

            # Varying M
            num_measurement_choices = dimensionality//3

            sparsity_penalties = np.linspace(10, 100, 10)

            step_size = 1

            reconstruction_errors = np.zeros(len(sparsity_penalties))
            print("Number of measurements: %s" % num_measurements)
            initial_estimate = analysis(np.random.normal(size=signal.shape)) # initial guess (just random analysis)
            
            print("Generating random measurement matrix of shape (%s, %s) ..." % (num_measurements, dimensionality))
            measurement_matrix = np.random.normal(size=(num_measurements, dimensionality))
            
            print("Normalizing measurement matrix (calculating SVD)...")
            measurement_matrix /= np.linalg.norm(measurement_matrix, ord=2) # normalize using 2-norm

            measurement = measurement_matrix @ flattened_image
               
            for index, sparsity_penalty in enumerate(sparsity_penalties):
                print("Sparsity penalty = %s" % sparsity_penalty)
                estimate, reconstruction_error = ista(initial_estimate, measurement, measurement_matrix, analysis, synthesis, sparsity_penalty=sparsity_penalty)

                reconstruction_errors[index] = reconstruction_error

                reconstructed_signal = synthesis(estimate)
                reconstructed_image = Image.fromarray(reconstructed_signal.astype(np.uint8))
                reconstructed_image.save("results/reconstructed_%s_signal_with_%s_analysis_and_%s_sparsity_penalty.png" % (signal_type, analysis_type, sparsity_penalty))
                
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.plot(sparsity_penalties, reconstruction_errors)
            ax.set_title("sparsity_penalty vs. reconstruction_error for %s signal with %s analysis" % (signal_type, analysis_type))
            fig.savefig("results/%s_signal_with_%s_analysis_sparsity_effect" % (signal_type, analysis_type))

if __name__ == "__main__":
    #sparse_approximation_for_2D_signals()
    #sparse_approximation_with_quantization()
    sparse_compressed_sensing_for_2D_signals()
