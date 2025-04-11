# ---------------------------------------------
# Script: Bubble Diameter Processing from Google Sheets
# Author: Mustahsin Reasad
# Description: Reads Google Sheets data from Google Drive, applies a timestamp shift,
#              computes average and middle-centered average bubble diameters,
#              and saves the result as an Excel file.
# ---------------------------------------------

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install required packages (for Google Sheets and Excel I/O)
!pip install openpyxl
!pip install gspread

# 3. Authenticate with Google
from google.colab import auth
auth.authenticate_user()

import gspread  # Access Google Sheets
from google.auth import default
import pandas as pd
import numpy as np
import os

# 4. Authenticate Google Sheets API
creds, _ = default()
gc = gspread.authorize(creds)

# === File Configuration ===
input_path = "/content/drive/MyDrive/Calculation/t421_422.gsheet"  # Input Google Sheet path
input_filename = os.path.splitext(os.path.basename(input_path))[0]  # Strip extension
output_filename = f"Calculated_{input_filename}.xlsx"  # Output Excel filename
output_path = f"/content/drive/MyDrive/Calculation/Excel Output/{output_filename}"

# 5. Read the first sheet of the Google Sheet
sheet = gc.open(input_filename).sheet1
data = sheet.get_all_values()  # Get all values as a 2D list

# Convert to pandas DataFrame (first row = header)
df = pd.DataFrame(data[1:], columns=data[0])

# 6. Clean and convert data types
df.columns = ['Timestamp', 'Diameter']
df['Diameter'] = pd.to_numeric(df['Diameter'], errors='coerce')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# 7. Apply timestamp shift
offset_seconds = -1  # Shift timestamps backward by 1 second
df['Timestamp'] += pd.Timedelta(seconds=offset_seconds)

# 8. Group by timestamp
grouped = df.groupby('Timestamp', sort=False)  # Maintain original order

# 9. Compute average bubble diameter per timestamp
avg_per_timestamp = grouped['Diameter'].mean().reset_index(name='Avg_Bubble_Diameter')

# 10. Define function to compute middle-centered average
def middle_centered_mean(subdf):
    n = len(subdf)
    if n == 0:
        return np.nan
    midpoint_index = (n - 1) // 2
    start_idx = max(0, midpoint_index - 5)
    end_idx = min(n, midpoint_index + 6)
    return subdf.iloc[start_idx:end_idx]['Diameter'].mean()

# 11. Apply middle-centered function
middle_centered_df = grouped.apply(middle_centered_mean).reset_index(name='MiddleCentered_Avg')

# 12. Merge results
result = pd.merge(avg_per_timestamp, middle_centered_df, on='Timestamp', how='outer')

# 13. Save result to Excel
result.to_excel(output_path, index=False)

print(f"✅ Calculation complete. Timestamp shifted by {offset_seconds} seconds.")
print(f"📁 Results saved to: {output_path}")
