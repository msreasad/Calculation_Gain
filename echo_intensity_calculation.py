# --------------------------------------------------------
# Script: Time-Resolved Peak Echo Intensity Detection
# Author: Mustahsin Reasad
# Description:
#   - Loads .mat echosounder data and extracts time series intensity
#   - Focuses on a target distance band (± tolerance)
#   - Computes average intensity across selected range per ping
#   - Detects peaks exceeding moving average + threshold
#   - Plots intensity vs. time and highlights peaks
# --------------------------------------------------------

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime, timedelta

# === Helper Function ===
def matlab_datenum_to_datetime(matlab_datenum):
    """Convert MATLAB datenum to Python datetime."""
    return (datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=matlab_datenum % 1)
            - timedelta(days=366))

# === USER SETTINGS ===
# Optional: Override full time range by setting start/end (e.g., "15:27:00")
start_time_str = None
end_time_str = None

# === File and Configuration ===
file_path = "/content/drive/MyDrive/Echosounder/Echosounder/Calibration_2/t421.mat"
matlab_data = scipy.io.loadmat(file_path)
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Extract configuration
config = matlab_data['Config']
n_cells = int(config['EchoSounder_NCells'][0, 0])
cell_size = float(config['EchoSounder_CellSize'][0, 0])
blanking_distance = float(config['EchoSounder_BlankingDistance'][0, 0])

# Compute distance-to-sensor for each cell
distance_to_sensor = np.arange(1, n_cells + 1) * cell_size + blanking_distance

# Extract time and echo data
data_structure = matlab_data['Data']
time_data = data_structure['Echo1Bin1_1000kHz_Time'][0, 0].flatten()
intensity_data = data_structure['Echo1Bin1_1000kHz_Echo'][0, 0]

# Convert MATLAB times to Python datetime
time_dates = np.array([matlab_datenum_to_datetime(t) for t in time_data])

# === Distance Range Selection ===
target_distance = 0.952  # m
tolerance = 0.09         # ± tolerance
lower_bound = target_distance - tolerance
upper_bound = target_distance + tolerance

# Select cells within range
range_mask = (distance_to_sensor >= lower_bound) & (distance_to_sensor <= upper_bound)
selected_indices = np.where(range_mask)[0]

if selected_indices.size == 0:
    print(" No data found in the specified range.")
    exit()

# === Compute Avg Intensity Over Range for Each Ping ===
intensity_slice = intensity_data[:, selected_indices]
avg_intensity = np.mean(intensity_slice, axis=1)

# === Moving Average and Peak Detection ===
window_size = 10
threshold = 5  # dB above moving average

moving_avg = np.convolve(avg_intensity, np.ones(window_size) / window_size, mode='same')
peaks, _ = find_peaks(avg_intensity, height=moving_avg + threshold)

# Filter out edge peaks
valid_peaks = [p for p in peaks if window_size // 2 < p < len(avg_intensity) - window_size // 2]
peak_times = time_dates[valid_peaks]
peak_values = avg_intensity[valid_peaks]

# === Time Window (optional override) ===
start_time_auto, end_time_auto = time_dates[0], time_dates[-1]
start_time = datetime.strptime(start_time_str, "%H:%M:%S").replace(
    year=time_dates[0].year, month=time_dates[0].month, day=time_dates[0].day
) if start_time_str else start_time_auto

end_time = datetime.strptime(end_time_str, "%H:%M:%S").replace(
    year=time_dates[0].year, month=time_dates[0].month, day=time_dates[0].day
) if end_time_str else end_time_auto

# === Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(time_dates, avg_intensity,
         label=f'Avg Intensity ({lower_bound:.3f}-{upper_bound:.3f} m)',
         color='teal', linewidth=1.5)
plt.plot(time_dates, moving_avg,
         label='Moving Average', linestyle='--', color='red', linewidth=1.5)
plt.scatter(peak_times, peak_values, color='orange',
            label=f'Peaks > {threshold} dB above moving avg', zorder=5)

plt.title(f'Peak Echo Intensity vs Time for {file_name}', fontsize=16)
plt.xlabel('Time (HH:MM:SS)', fontsize=14)
plt.ylabel('Intensity (dB)', fontsize=14)
plt.xlim(start_time, end_time)
plt.ylim(min(avg_intensity)-5, max(avg_intensity)+5)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.tight_layout()
plt.show()

# === Export to DataFrames ===
df_intensity = pd.DataFrame({
    'Timestamp': time_dates,
    'Intensity (dB)': avg_intensity
})
df_peaks = pd.DataFrame({
    'Timestamp': peak_times,
    'Peak Intensity (dB)': peak_values
})

# === Print Results ===
print("\n Intensity vs Time (first few rows):")
print(df_intensity.head())

print("\n Detected Peaks:")
with pd.option_context('display.max_rows', None):
    print(df_peaks)
