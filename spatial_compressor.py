#First cell
import numpy as np
import soundfile as sf

# Set up time and sample rate
sr = 48000
duration = 5.0 # 5 seconds of audio
t = np.linspace(0, duration, int(sr * duration))

# 1. Create a sound source (a 440Hz tone)
source = np.sin(2 * np.pi * 440 * t)

# 2. Make it move! Angle theta spins from 0 to 360 degrees (2*pi)
theta = np.linspace(0, 2 * np.pi, len(t))

# 3. Encode into First-Order Ambisonics (B-Format)
W = source * 0.707
X = source * np.cos(theta)
Y = source * np.sin(theta)
Z = np.zeros_like(t) # Keeping it flat on the horizontal plane

# 4. Stack into a 4-channel matrix and save
foa_audio = np.vstack((W, X, Y, Z))
sf.write('spinning_test_tone.wav', foa_audio.T, sr)

print("Created 'spinning_test_tone.wav' successfully!")

#Second cell
# Install necessary audio libraries
!pip install librosa soundfile scikit-learn

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1. SPATIAL REPRESENTATION ---
# Upload your multi-channel WAV file to Colab's file explorer first
audio_path = 'your_spatial_audio_file.wav'

# Load audio (mono=False ensures we keep all spatial channels)
try:
    y_original, sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"Original audio shape: {y_original.shape} (Channels, Samples)")

    # --- 2. REDUNDANCY ANALYSIS & COMPRESSION ---
    # Transpose so samples are rows and channels are columns for PCA
    y_transposed = y_original.T

    # Initialize PCA (e.g., compress down to 2 principal components)
    pca = PCA(n_components=2)

    # Fit and transform the data (Compression)
    y_compressed = pca.fit_transform(y_transposed)
    print(f"Compressed audio shape: {y_compressed.shape}")

    # --- 3. RECONSTRUCTION ---
    # Reconstruct the audio back to the original number of channels
    y_reconstructed = pca.inverse_transform(y_compressed).T

    # --- 4. VISUALIZATION ---
    # Plotting channel 1 original vs reconstructed to visualize loss
    plt.figure(figsize=(14, 5))
    plt.plot(y_original, label='Original Channel 1', alpha=0.7)
    plt.plot(y_reconstructed, label='Reconstructed Channel 1', alpha=0.7)
    plt.title('Spatial Audio: Original vs. Reconstructed Waveform')
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Please upload an audio file and update the 'audio_path' variable.")

#Third cell
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# --- 1. SETUP & LOADING ---
# Upload your 4-channel B-format file to Colab's file explorer
# After uploading, update this variable with the correct filename.
audio_path = 'spinning_test_tone.wav' # <--- Please change this to your uploaded file name, e.g., 'my_foa_audio.wav'

try:
    y_original, sr = librosa.load(audio_path, sr=None, mono=False)

    # Verify it is an FOA file
    if y_original.shape[0] != 4:
        print(f"Warning: Expected 4 channels for FOA B-Format, but got {y_original.shape[0]}.")
    else:
        print(f"Loaded Audio Shape: {y_original.shape} (Channels, Samples)")

    # --- 2. COMPRESSION (PCA) ---
    y_transposed = y_original.T

    # Compress from 4 channels down to 2 (a 50% data reduction)
    pca = PCA(n_components=2)
    y_compressed = pca.fit_transform(y_transposed)

    # Calculate how much spatial energy we kept
    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Variance Retained with 2 components: {variance_retained:.2f}%\n")

    # --- 3. RECONSTRUCTION ---
    y_reconstructed = pca.inverse_transform(y_compressed).T

    # --- 4. EVALUATION: MATHEMATICAL ERROR ---
    # Calculate Mean Squared Error for each specific FOA channel
    channels = ['W (Omni)', 'X (Front/Back)', 'Y (Left/Right)', 'Z (Up/Down)']
    print("--- Mean Squared Error (MSE) per Channel ---")
    for i in range(4):
        mse = mean_squared_error(y_original[i], y_reconstructed[i])
        print(f"{channels[i]}: {mse:.6f}")

    # --- 5. EVALUATION: VISUALIZE SPATIAL CUES ---
    # To evaluate horizontal localization, we plot X vs Y for a short time frame.
    # This creates a visual "map" of where the sound energy is pointing.

    frame_start = 10000 # Adjust these frame indices to find an active part of your audio
    frame_end = 12000

    # Extract X (index 1) and Y (index 2) channels
    X_orig, Y_orig = y_original[1, frame_start:frame_end], y_original[2, frame_start:frame_end]
    X_recon, Y_recon = y_reconstructed[1, frame_start:frame_end], y_reconstructed[2, frame_start:frame_end]

    # Normalize limits for the plot based on the max amplitude
    limit = max(np.max(np.abs(X_orig)), np.max(np.abs(Y_orig))) * 1.1

    plt.figure(figsize=(12, 5))

    # Plot Original Spatial Position
    plt.subplot(1, 2, 1)
    plt.scatter(Y_orig, X_orig, alpha=0.5, c='blue', s=5)
    plt.title("Original Localization (Horizontal Plane X vs Y)")
    plt.xlabel("Y (Left/Right)")
    plt.ylabel("X (Front/Back)")
    plt.grid(True)
    plt.xlim([-limit, limit])
    plt.ylim([-limit, limit])

    # Plot Reconstructed Spatial Position
    plt.subplot(1, 2, 2)
    plt.scatter(Y_recon, X_recon, alpha=0.5, c='red', s=5)
    plt.title("Reconstructed Localization (After PCA)")
    plt.xlabel("Y (Left/Right)")
    plt.ylabel("X (Front/Back)")
    plt.grid(True)
    plt.xlim([-limit, limit])
    plt.ylim([-limit, limit])

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("Please upload your FOA .wav file to Colab's file explorer and update the 'audio_path' variable in this cell.")

#Fourth cell
import IPython.display as ipd
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

# --- 1. DECODE FOA TO STEREO FOR LISTENING ---
def decode_foa_to_stereo(foa_signal):
    # np.squeeze ensures we don't have hidden extra dimensions
    W = np.squeeze(foa_signal)
    Y = np.squeeze(foa_signal)

    L = (W + Y) * 0.707
    R = (W - Y) * 0.707

    # vstack guarantees a strict (2, Samples) 2D array
    return np.vstack((L, R))

# Decode the matrices
stereo_original = decode_foa_to_stereo(y_original)
stereo_reconstructed = decode_foa_to_stereo(y_reconstructed)

# --- 2. CREATE COLAB AUDIO WIDGETS ---
# Notice: No .T is used here!
audio_orig = ipd.Audio(stereo_original, rate=sr)
audio_recon = ipd.Audio(stereo_reconstructed, rate=sr)

# --- 3. BUILD THE EVALUATION DASHBOARD ---
out = widgets.Output()

def on_button_clicked(b):
    with out:
        out.clear_output()

        print("🎧 ORIGINAL AUDIO (Uncompressed)")
        display(audio_orig)

        print("\n🎧 RECONSTRUCTED AUDIO (PCA Compressed)")
        display(audio_recon)

        print("\n📊 SPATIAL CUE VISUALIZATION (Goniometer)")

        # Visualize a specific timeframe
        frame_start, frame_end = 10000, 12000
        X_orig, Y_orig = y_original[1, frame_start:frame_end], y_original[2, frame_start:frame_end]
        X_recon, Y_recon = y_reconstructed[1, frame_start:frame_end], y_reconstructed[2, frame_start:frame_end]
        limit = max(np.max(np.abs(X_orig)), np.max(np.abs(Y_orig))) * 1.1

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(Y_orig, X_orig, alpha=0.3, c='blue', s=2)
        plt.title("Original Spatial Spread")
        plt.xlim([-limit, limit]); plt.ylim([-limit, limit]); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(Y_recon, X_recon, alpha=0.3, c='red', s=2)
        plt.title("Reconstructed Spatial Spread")
        plt.xlim([-limit, limit]); plt.ylim([-limit, limit]); plt.grid(True)

        plt.tight_layout()
        plt.show()

# Run the UI
btn = widgets.Button(description="Generate A/B Comparison", button_style='info')
btn.on_click(on_button_clicked)
display(btn, out)

#Fifth cell
from google.colab import files
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import IPython.display as ipd
import ipywidgets as widgets

# --- 1. INTERACTIVE UPLOAD WIDGET ---
print("Please upload your 4-channel FOA .wav file:")
# This creates the "Choose Files" button
uploaded = files.upload()

if not uploaded:
    print("No file was uploaded. Please run the cell again.")
else:
    # Get the name of the file you just uploaded
    filename_list = list(uploaded.keys())
    if not filename_list:
        print("No file was uploaded or filename could not be retrieved.")
    else:
        filename = filename_list[0] # Correctly get the string filename
        print(f"\n✅ Successfully uploaded '{filename}'. Processing...")

        # --- 2. LOAD & VERIFY AUDIO (THE HACK) ---
        y_original, sr = librosa.load(filename, sr=None, mono=False, duration=30.0)

        # If it's stereo, let's pad it with two empty channels to reach 4
        if y_original.ndim == 2 and y_original.shape[0] == 2:
            print("⚠️ Detected Stereo (2ch). Padding to 4 channels to allow PCA testing.")
            # Create 2 silent channels of the same length
            silence = np.zeros((2, y_original.shape[1]))
            # Stack them: [L, R, 0, 0]
            y_original = np.vstack((y_original, silence))

        # Check if we have 4 channels now
        if y_original.ndim != 2 or y_original.shape[0] != 4:
            print(f"❌ Error: Still don't have 4 channels. Found {y_original.shape}.\nPlease ensure you are uploading a First-Order Ambisonics (B-Format) file or a 2-channel file that can be padded.")
        else:
            print(f"✅ Success! Processing 4-channel matrix: {y_original.shape}")

            # --- 3. COMPRESSION (PCA) ---
            pca = PCA(n_components=2)
            y_compressed = pca.fit_transform(y_original.T)
            y_reconstructed = pca.inverse_transform(y_compressed).T

            variance = np.sum(pca.explained_variance_ratio_) * 100
            print(f"Spatial Variance Retained: {variance:.2f}%")

            # --- 4. DECODE TO PLAYABLE STEREO ---
            def decode_foa_to_stereo(foa_signal):
                # W is the omnidirectional channel (index 0)
                W = foa_signal[0]
                # Y is the left-right spatial axis channel (index 2)
                Y = foa_signal[2]

                L = (W + Y) * 0.707
                R = (W - Y) * 0.707
                return np.vstack((L, R)) # Strict 2D array for Colab Audio

            stereo_original = decode_foa_to_stereo(y_original)
            stereo_reconstructed = decode_foa_to_stereo(y_reconstructed)

            audio_orig = ipd.Audio(stereo_original, rate=sr)
            audio_recon = ipd.Audio(stereo_reconstructed, rate=sr)

            # --- 5. BUILD THE EVALUATION DASHBOARD ---
            out = widgets.Output()

            def on_button_clicked(b):
                with out:
                    out.clear_output()

                    print("🎧 ORIGINAL AUDIO (Uncompressed)")
                    display(audio_orig)

                    print("\n🎧 RECONSTRUCTED AUDIO (2-Component PCA)")
                    display(audio_recon)

                    print("\n📊 SPATIAL CUE VISUALIZATION (Mid-file sample)")

                    # Dynamically grab 2000 samples from the exact middle of the file
                    # Ensure mid_point is an integer index
                    mid_point = y_original.shape[1] // 2
                    frame_start = max(0, mid_point - 1000)
                    frame_end = min(y_original.shape[1], mid_point + 1000)

                    X_orig = y_original[1, frame_start:frame_end]
                    Y_orig = y_original[2, frame_start:frame_end]
                    X_recon = y_reconstructed[1, frame_start:frame_end]
                    Y_recon = y_reconstructed[2, frame_start:frame_end]

                    limit = max(np.max(np.abs(X_orig)), np.max(np.abs(Y_orig))) * 1.1

                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.scatter(Y_orig, X_orig, alpha=0.3, c='blue', s=2)
                    plt.title("Original Spatial Spread")
                    plt.xlim([-limit, limit]); plt.ylim([-limit, limit]); plt.grid(True)

                    plt.subplot(1, 2, 2)
                    plt.scatter(Y_recon, X_recon, alpha=0.3, c='red', s=2)
                    plt.title("Reconstructed Spatial Spread")
                    plt.xlim([-limit, limit]); plt.ylim([-limit, limit]); plt.grid(True)

                    plt.tight_layout()
                    plt.show()

            btn = widgets.Button(description="Generate A/B Comparison", button_style='info')
            btn.on_click(on_button_clicked)
            display(btn, out)
