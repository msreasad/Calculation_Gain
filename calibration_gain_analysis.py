"""
Thesis Code Repository: Calculation_Gain
Author: Mustahsin Reasad
Affiliation: University of Missouri – Columbia
Program: M.S. in Civil and Environmental Engineering

Description:
-------------
This script contains modular code used for the calibration of an echosounder using a tungsten carbide (WC) ball
and for estimating the target strength (TS) of bubbles in freshwater. The calculations are performed using 
the SONAR equation and include corrections for sound attenuation, spherical spreading, and noise threshold (NT).

The code supports:
- Freshwater sound attenuation modeling
- Processing echosounder .mat data for received power
- Estimating noise threshold (NT)
- Calibration gain (G) computation with/without NT
- Bubble target strength (TS) estimation

Dependencies:
-------------
- numpy
- scipy
- matplotlib
- pandas

Example Use Cases:
------------------
- Lab calibration of acoustic instruments
- Bubble size and TS measurement from echo intensity
- Underwater acoustics and fluid dynamics research
"""

# ---------------------------------------------
# 1. Mount Google Drive (for Google Colab use)
# ---------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# ---------------------------------------------
# 2. Freshwater Sound Attenuation Calculation
# ---------------------------------------------
def attenuation_freshwater(depth, temperature, frequency=1000):
    """
    Calculate attenuation in freshwater using temperature, depth, and frequency.

    Returns:
        attenuation (float): in dB/m
    """
    if temperature <= 20:
        A3 = 4.937e-4 - 2.590e-5 * temperature + 9.11e-7 * temperature**2 - 1.5e-8 * temperature**3
    else:
        A3 = 3.964e-4 - 1.146e-5 * temperature + 1.45e-7 * temperature**2 - 6.5e-10 * temperature**3

    P3 = 1 - 3.83e-5 * depth + 4.9e-10 * depth**2
    attenuation = A3 * P3 * frequency**2 / 1000  # convert dB/km to dB/m
    return attenuation

# ---------------------------------------------
# 3. Load Echo Data and Calculate Received Power
# ---------------------------------------------
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

folder = '/content/drive/MyDrive/Echosounder/Echosounder/Calibration_2'
file_name = 't406'
file_path = f'{folder}/{file_name}.mat'

data = scipy.io.loadmat(file_path)
config = data['Config']
n_cells = int(config['EchoSounder_NCells'][0, 0])
cell_size = float(config['EchoSounder_CellSize'][0, 0])
blanking_distance = float(config['EchoSounder_BlankingDistance'][0, 0])
distance_to_sensor = np.arange(1, n_cells + 1) * cell_size + blanking_distance

echo = data['Data']['Echo1Bin1_1000kHz_Echo'][0, 0].T
mean_echo = np.mean(echo, axis=1)

valid = (distance_to_sensor >= 0.1) & (distance_to_sensor <= 1.2)
distance_valid = distance_to_sensor[valid]
mean_echo_valid = mean_echo[valid]

plt.figure(figsize=(10, 5))
plt.plot(distance_valid, mean_echo_valid, 'b-')
plt.title(f'Intensity vs. Distance for {file_name}')
plt.xlabel('Distance (m)')
plt.ylabel('Mean Intensity (dB)')
plt.grid(True)
plt.show()

max_intensity = np.max(mean_echo_valid)
max_intensity_distance = distance_valid[np.argmax(mean_echo_valid)]
print(f"Max Intensity: {max_intensity:.2f} dB at {max_intensity_distance:.2f} m")

# ---------------------------------------------
# 4. Estimate Noise Threshold (NT)
# ---------------------------------------------
exclude_range = 0.1
noise_mask = (distance_valid < max_intensity_distance - exclude_range) | \
             (distance_valid > max_intensity_distance + exclude_range)
NT = np.mean(mean_echo_valid[noise_mask])
print(f"Noise Threshold (NT): {NT:.2f} dB")

# ---------------------------------------------
# 5. Calibration Gain Calculation (with NT)
# ---------------------------------------------
def calculate_gain(ts, pr, nt, R, alpha, PL):
    """
    Compute calibration gain G using SONAR equation.

    Parameters:
        ts (float): Theoretical TS (dB)
        pr (float): Received intensity (dB)
        nt (float): Noise threshold (dB)
        R (float): Range to target (m)
        alpha (float): Attenuation (dB/m)
        PL (float): Power Level (dB)
    """
    Pr_lin = 10**(pr / 10)
    NT_lin = 10**(nt / 10)

    if Pr_lin <= NT_lin:
        raise ValueError("Received power must be greater than noise threshold.")

    log_term = 10 * np.log10(Pr_lin - NT_lin)
    gain = ts - (log_term + 40 * np.log10(R) + 2 * alpha * R + PL)
    return gain

ts_theoretical = -47.25
received_power = 89.814 * 0.01
nt = 41.62 * 0.01
R = 0.85
alpha = 0.220097
PL = 0

G = calculate_gain(ts_theoretical, received_power, nt, R, alpha, PL)
print(f"Calibration Gain (G): {G:.4f} dB")

# ---------------------------------------------
# 6. Bubble Target Strength (TS) Calculation
# ---------------------------------------------
def calculate_bubble_TS(pr_bubble, R, alpha, PL, G):
    """
    Compute bubble target strength (TS) using SONAR equation.
    """
    spread = 40 * np.log10(R)
    attenuation = 2 * alpha * R
    ts_bubble = pr_bubble + spread + attenuation + PL + G
    return ts_bubble

pr_bubble = 96.2 * 0.01
bubble_ts = calculate_bubble_TS(pr_bubble, R, alpha, PL, G)
print(f"Bubble Target Strength (TS): {bubble_ts:.4f} dB")
