import numpy as np
import soundfile as sf
import os

def process_spatial_audio(input_file, output_file, target_channels):
    """
    Compresses multi-channel spatial audio using Redundancy Analysis (PCA)
    and reconstructs it for subjective A/B testing in Reaper.
    """
    print(f"--- Spatial Audio Compression pipeline ---")
    print(f"Loading '{input_file}'...")
    
    # 1. LOAD SPATIAL AUDIO
    # data shape will be (samples, channels). e.g., 16 channels for 3rd Order.
    try:
        data, samplerate = sf.read(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    original_channels = data.shape[1]
    print(f"Original spatial resolution: {original_channels} channels at {samplerate}Hz")
    
    if target_channels >= original_channels:
        print("Error: Target compressed channels must be less than original channels.")
        return

    # 2. REDUNDANCY ANALYSIS (PCA)
    print("Analyzing spatial redundancy...")
    # Mean-center the audio data (per channel) to prepare for covariance math
    channel_means = np.mean(data, axis=0)
    centered_data = data - channel_means

    # Calculate the covariance matrix between the spatial channels
    # This reveals how much audio channels overlap or are "redundant"
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Calculate eigenvalues and eigenvectors to find the Principal Components
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the vectors by highest energy (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 3. COMPRESSION
    print(f"Compressing down to {target_channels} channels...")
    # Create the compression matrix using only the top 'K' spatial components
    compression_matrix = sorted_eigenvectors[:, :target_channels]
    
    # Multiply the original 16-channel audio by the matrix to compress it
    compressed_data = np.dot(centered_data, compression_matrix)
    print(f"Data compressed. Internal footprint: {compressed_data.shape[1]} channels.")

    # 4. RECONSTRUCTION
    print("Reconstructing soundfield for evaluation...")
    # To play it in Reaper, we must expand it back to the original channel count.
    # We do this by multiplying by the transposed matrix. 
    # (The audio will be 16 channels again, but missing the discarded spatial data)
    reconstruction_matrix = compression_matrix.T
    reconstructed_data = np.dot(compressed_data, reconstruction_matrix) + channel_means

    # 5. EXPORT
    print(f"Saving reconstructed audio to '{output_file}'...")
    # Prevent clipping before saving
    reconstructed_data = np.clip(reconstructed_data, -1.0, 1.0)
    sf.write(output_file, reconstructed_data, samplerate)
    print("Done! Ready for subjective evaluation in Reaper.")


if __name__ == "__main__":
    # --- PROJECT CONFIGURATION ---
    
    # Put the name of your raw Ambisonic test file here
    INPUT_WAV = "original_ambisonics.wav"  
    
    # This is the file you will drag into Track 2 in Reaper
    OUTPUT_WAV = "reconstructed_ambisonics.wav" 
    
    # How heavily do you want to compress it?
    # e.g., If original is 3rd Order (16 channels), try compressing to 6.
    COMPRESSED_CHANNELS = 6 
    
    # --- Auto-Generate Test File (For your first run) ---
    if not os.path.exists(INPUT_WAV):
        print(f"'{INPUT_WAV}' not found. Generating a 5-second 16-channel test sweep...")
        # Generates 5 seconds of random white noise across 16 channels to test the math
        dummy_data = np.random.randn(48000 * 5, 16) * 0.1 
        sf.write(INPUT_WAV, dummy_data, 48000)
        
    process_spatial_audio(INPUT_WAV, OUTPUT_WAV, COMPRESSED_CHANNELS)