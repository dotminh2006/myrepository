Compressing Spatial Audio for Immersive Playback

This project implements a pipeline to compress First-Order Ambisonics (FOA) audio using Principal Component Analysis (PCA). The goal is to analyze spatial redundancy, compress the 4-channel B-format signal, and evaluate the reconstructed audio through both mathematical error analysis and subjective visual cues (Goniometer/Lissajous figures).

Prerequisites
- Environment: Google Colab
- Libraries: `librosa`, `scikit-learn`, `numpy`, `matplotlib`, `ipywidgets`

Project Workflow

Step 1: Install Dependencies
Run this cell once at the start of your notebook to ensure all required libraries are installed.
```python
!pip install librosa soundfile scikit-learn

Step 2: Generate a Test File (Optional)
If you do not have a 4-channel B-Format .wav file, run this cell to generate a "spinning" test tone. This allows you to verify the pipeline works before testing real-world audio.
import numpy as np
import soundfile as sf

# Set up time and sample rate
sr = 48000
duration = 5.0 
t = np.linspace(0, duration, int(sr * duration))

# Create a source signal and a rotating movement pattern
source = np.sin(2 * np.pi * 440 * t) 
theta = np.linspace(0, 2 * np.pi, len(t))

# Encode into FOA (B-Format)
W = source * 0.707
X = source * np.cos(theta)
Y = source * np.sin(theta)
Z = np.zeros_like(t) 

# Save as 4-channel file
foa_audio = np.vstack((W, X, Y, Z))
sf.write('spinning_test_tone.wav', foa_audio.T, sr)
print("Created 'spinning_test_tone.wav' successfully!")

Step 3: The Master Processing Pipeline
This cell handles everything: file upload, PCA compression, reconstruction, and the evaluation dashboard.
Important: This script enforces a strict 4-channel check. Ensure your input file is a true FOA B-Format .wav file.
from google.colab import files
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import IPython.display as ipd
import ipywidgets as widgets

# Upload file
print("Please upload your 4-channel FOA .wav file:")
uploaded = files.upload() 

if uploaded:
    filename = list(uploaded.keys())[0]
    
    # Load audio
    y_original, sr = librosa.load(filename, sr=None, mono=False, duration=30.0)

    # Validate channel count
    if y_original.shape[0] != 4:
        print(f"❌ Error: Expected 4 channels for FOA, but got {y_original.shape[0]}.")
    else:
        # PCA Compression
        pca = PCA(n_components=2)
        y_compressed = pca.fit_transform(y_original.T)
        y_reconstructed = pca.inverse_transform(y_compressed).T
        
        # UI Setup
        def decode_foa_to_stereo(foa_signal):
            W, Y = np.squeeze(foa_signal[0]), np.squeeze(foa_signal[2])
            L, R = (W + Y) * 0.707, (W - Y) * 0.707
            return np.vstack((L, R))

        audio_orig = ipd.Audio(decode_foa_to_stereo(y_original), rate=sr)
        audio_recon = ipd.Audio(decode_foa_to_stereo(y_reconstructed), rate=sr)
        
        out = widgets.Output()
        def on_button_clicked(b):
            with out:
                out.clear_output()
                display(audio_orig, audio_recon)
                
                # Plotting
                mid = y_original.shape[1] // 2
                X_orig, Y_orig = y_original[1, mid-1000:mid+1000], y_original[2, mid-1000:mid+1000]
                X_rec, Y_rec = y_reconstructed[1, mid-1000:mid+1000], y_reconstructed[2, mid-1000:mid+1000]
                
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1); plt.scatter(Y_orig, X_orig, s=2); plt.title("Original Spread")
                plt.subplot(1, 2, 2); plt.scatter(Y_rec, X_rec, s=2, c='red'); plt.title("Reconstructed Spread")
                plt.show()

        btn = widgets.Button(description="Generate Dashboard", button_style='info')
        btn.on_click(on_button_clicked)
        display(btn, out)

***Note: Troubleshooting
Troubleshooting

    TypeError: Invalid file [...]: This usually happens if the file selection is empty. Ensure you actually selected a file in the "Choose Files" prompt.

    Error: Expected 4 channels...: Your input file is likely standard Stereo or Mono. You must provide a true First-Order Ambisonic file. Use the generator in Step 2 to verify your pipeline is working with known-good data.

    Colab Freezing: If the cell hangs, click the 'Stop' button (square in the circle) and re-run. Ensure you only process the first 30 seconds of large files to avoid memory limits.

