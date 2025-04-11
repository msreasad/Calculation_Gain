# --------------------------------------------------------
# Script: Echosounder Peak Detection from .mat Data (0–1.3 m)
# Author: Mustahsin Reasad
# Description:
#   - Loads acoustic intensity data from a .mat file
#   - Computes mean echo intensity per cell
#   - Identifies peaks in the 0–1.3 m region using moving average filtering
#   - Visualizes and labels detected peaks
# --------------------------------------------------------

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# === Load .mat file ===
file_path = "/content/drive/MyDrive/Echosounder/Echosounder/Calibration_2/t421.mat"
data = scipy.io.loadmat(file_path)

# === Extract configuration parameters ===
config = data['Config']
n_cells = int(config['EchoSounder_NCells'][0, 0])
cell_size = float(config['EchoSounder_CellSize'][0, 0])
blanking_distance = float(config['EchoSounder_BlankingDistance'][0, 0])

# Calculate physical distance to each cell from transducer
distance_to_sensor = np.arange(1, n_cells + 1) * cell_size + blanking_distance

# === Load echo intensity data and compute mean intensity ===
echo = data['Data']['Echo1Bin1_1000kHz_Echo'][0, 0].T  # Transpose for column-wise mean
mean_echo = np.mean(echo, axis=1)

# === Restrict analysis to 0–1.3 m range ===
mask = (distance_to_sensor >= 0) & (distance_to_sensor <= 1.3)
distance_subset = distance_to_sensor[mask]
mean_echo_subset = mean_echo[mask]

# === Compute moving average for background comparison ===
window_size = 10  # Adjust for smoothing strength
moving_avg_subset = np.convolve(mean_echo_subset, np.ones(window_size) / window_size, mode='same')

# === Peak detection parameters ===
threshold = 4  # Peaks must be 4 dB above local moving average

# Find peaks above moving average threshold
peaks, properties = find_peaks(mean_echo_subset, height=moving_avg_subset + threshold)

# Filter peaks near edges (avoid false positives due to windowing)
valid_peaks = [p for p in peaks if window_size // 2 < p < len(mean_echo_subset) - window_size // 2]

# Extract distance and intensity values of valid peaks
peak_distances = distance_subset[valid_peaks]
peak_intensities = mean_echo_subset[valid_peaks]

# === Print identified peak data ===
for i, (dist, inten) in enumerate(zip(peak_distances, peak_intensities), start=1):
    print(f"Peak {i}: Intensity = {inten:.2f} dB at Distance = {dist:.2f} m")

# === Store peak info in DataFrame ===
df_peaks = pd.DataFrame({
    'Peak Number': [f'P{i}' for i in range(1, len(valid_peaks) + 1)],
    'Distance (m)': peak_distances,
    'Intensity (dB)': peak_intensities
})

print("\nPeaks DataFrame:")
print(df_peaks)

# === Visualization ===
plt.figure(figsize=(12, 6))
plt.plot(distance_subset, mean_echo_subset, label='Mean Intensity', color='blue', linewidth=1.5)
plt.plot(distance_subset, moving_avg_subset, label='Moving Average', color='red', linestyle='--', linewidth=1.5)

# Mark peaks
plt.scatter(peak_distances, peak_intensities, color='orange', label=f'Peaks > {threshold} dB', s=80)

# Annotate each peak
for i, (x, y) in enumerate(zip(peak_distances, peak_intensities), start=1):
    plt.text(x, y + 0.5, f'P{i}', fontsize=12, color='black', ha='center', va='bottom')

# Final plot setup
plt.title('Echo Intensity vs. Distance (0–1.3 m) with Detected Peaks', fontsize=16)
plt.xlabel('Distance to Sensor (m)', fontsize=14)
plt.ylabel('Mean Intensity (dB)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 1.3)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
