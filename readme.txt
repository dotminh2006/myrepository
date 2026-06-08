markdown_content = """# Spatial Audio Compression for Immersive Playback

## Project Overview
This project implements a complete pipeline for analyzing, compressing, and subjectively evaluating spatial audio (Ambisonics). Using Python and Principal Component Analysis (PCA), the system identifies spatial redundancy in high-order ambisonic soundfields (e.g., 3rd-order, 16 channels), compresses the data footprint, and reconstructs it. The reconstructed audio is then evaluated against the original uncompressed audio using a binaural rendering engine in the REAPER Digital Audio Workstation (DAW).

---

## 1. System Prerequisites

### Operating System
* **Windows 10/11 (Highly Recommended):** Native support for VST3 plugins and pre-compiled Python audio wheels.
* **Ubuntu 22.04 (Alternative):** Requires manual VST directory management and specific Linux VST packages (IEM Plugin Suite is recommended over SPARTA for compatibility).

### Required Software
1.  **Python 3.8+** (with `pip` and `venv`)
2.  **REAPER DAW:** [Download REAPER](https://www.reaper.fm/download.php)
3.  **IEM Plugin Suite:** [Download IEM Suite](https://plugins.iem.at/) (Contains the critical `BinauralDecoder` VST3 plugin).

---

## 2. REAPER Setup & Configuration

### A. Installing the Spatial Plugins (IEM Suite)
The IEM Binaural Decoder is required to translate the multi-channel mathematics of Ambisonics into a 3D acoustic space you can hear over standard stereo headphones.
* **Windows:** Run the IEM installer `.exe` and ensure the **VST3** option is checked. It will install to `C:\\Program Files\\Common Files\\VST3`.
* **Linux:** Extract the downloaded archive and copy the `.vst3` folders (specifically `BinauralDecoder.vst3`) into `~/.vst3`.

### B. Configuring the Audio Backend
1. Open REAPER and press `Ctrl + P` to open Preferences.
2. Navigate to **Audio -> Device**.
3. **Windows:** Set the Audio system to **WASAPI** (Shared or Exclusive mode). If using an external audio interface, select **ASIO**.
4. **Linux:** Select **ALSA** or **JACK/PipeWire** depending on your distribution's audio server.

### C. Scanning the Plugins
1. In Preferences, go to **Plug-ins -> VST**.
2. Click **Re-scan -> Scan for new/modified plug-ins**. Ensure REAPER detects "IEM: BinauralDecoder".

---

## 3. Creating the Evaluation Template

To compare the compressed audio against the original, you must set up a multi-channel project template.

1.  **Create Tracks:** Create Track 1 (Name: "Original") and Track 2 (Name: "Reconstructed").
2.  **Configure Track Channels:**
    * Click the **Route** button on Track 1.
    * Change **Track channels** to **16** (for 3rd-Order Ambisonics) or **4** (for 1st-Order).
    * Repeat this exact step for Track 2.
3.  **Configure the Master Track:**
    * Go to **View -> Master Track** to make it visible.
    * Click **Route** on the Master Track and change its channels to **16** (or 4).
    * Click the **FX** button on the Master Track.
    * Add **VST3: BinauralDecoder (IEM)**.
4.  **Configure the Decoder:**
    * In the plugin window, set **Input Order** to match your project (e.g., 3rd).
    * Select your headphone EQ profile if desired.

*Tip: Save this empty project as a REAPER Template (`File > Project templates > Save project as template...`) for easy access.*

---

## 4. Python Environment Setup

The Python pipeline handles the matrix math required to identify spatial redundancy and perform the compression.

1.  **Create a Virtual Environment:** Open your terminal/command prompt in your project folder.
    ```
```text?code_stdout&code_event_index=2
File saved successfully with tag: /mnt/data/Spatial_Audio_Compression_README.md

```bash
    # Windows
    python -m venv spatial_audio_env
    spatial_audio_env\\Scripts\\activate

    # Linux/Mac
    python3 -m venv spatial_audio_env
    source spatial_audio_env/bin/activate
    ```
2.  **Install Dependencies:**
    ```bash
    pip install numpy scipy soundfile
    ```
    *(Optional: Install `librosa`, `matplotlib`, and `spaudiopy` for advanced plotting and feature extraction).*

---

## 5. Running the Python Compression Pipeline

Save the following code as `spatial_compressor.py` in your project folder.

```python
import numpy as np
import soundfile as sf
import os

def process_spatial_audio(input_file, output_file, target_channels):
    print(f"--- Spatial Audio Compression pipeline ---")
    print(f"Loading '{input_file}'...")
    
    try:
        data, samplerate = sf.read(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    original_channels = data.shape[1]
    print(f"Original spatial resolution: {original_channels} channels at {samplerate}Hz")
    
    # REDUNDANCY ANALYSIS (PCA)
    print("Analyzing spatial redundancy...")
    channel_means = np.mean(data, axis=0)
    centered_data = data - channel_means
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # COMPRESSION
    print(f"Compressing down to {target_channels} channels...")
    compression_matrix = sorted_eigenvectors[:, :target_channels]
    compressed_data = np.dot(centered_data, compression_matrix)

    # RECONSTRUCTION
    print("Reconstructing soundfield for evaluation...")
    reconstruction_matrix = compression_matrix.T
    reconstructed_data = np.dot(compressed_data, reconstruction_matrix) + channel_means

    # EXPORT
    reconstructed_data = np.clip(reconstructed_data, -1.0, 1.0)
    sf.write(output_file, reconstructed_data, samplerate)
    print(f"Saved reconstructed audio to '{output_file}'. Ready for Reaper!")

if __name__ == "__main__":
    INPUT_WAV = "original_ambisonics.wav"  
    OUTPUT_WAV = "reconstructed_ambisonics.wav" 
    COMPRESSED_CHANNELS = 6 
    
    # Auto-generate a dummy file for testing if one doesn't exist
    if not os.path.exists(INPUT_WAV):
        print(f"'{INPUT_WAV}' not found. Generating a 5-second 16-channel test noise...")
        dummy_data = np.random.randn(48000 * 5, 16) * 0.1 
        sf.write(INPUT_WAV, dummy_data, 48000)
        
    process_spatial_audio(INPUT_WAV, OUTPUT_WAV, COMPRESSED_CHANNELS)