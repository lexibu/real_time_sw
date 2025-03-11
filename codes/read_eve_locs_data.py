from datetime import datetime, timedelta

import pandas as pd
import requests

# URL of the file
eve_url = "https://lasp.colorado.edu/eve/data_access/eve_data/quicklook/L0CS/SpWx/2025/20250303_EVE_L0CS_DIODES_1m_counts.txt"

# Fetch the file content to find where the header ends
response = requests.get(eve_url)
lines = response.text.splitlines()

# Find the line number where the header ends
header_end_line = next(i for i, line in enumerate(lines) if ";END_OF_HEADER" in line)

# Extract the date information from the line after the header
date_line = lines[header_end_line + 1].strip()
year, doy, month, day = map(int, date_line.split())

# Convert DOY (Day of Year) to a proper date
base_date = datetime(year, 1, 1)  # Start from the first day of the year
date = base_date + timedelta(days=doy - 1)  # Add the DOY to get the correct date

# Read the data into a DataFrame, skipping the header rows and the date line
df = pd.read_csv(eve_url, skiprows=header_end_line + 2, sep="\\s+", header=None)

# Define column names (from the header description)
column_names = [
    "HHMM",
    "XRS-B_proxy",
    "XRS-A_proxy",
    "SEM_proxy",
    "0.1-7ESPquad",
    "17.1ESP",
    "25.7ESP",
    "30.4ESP",
    "36.6ESP",
    "darkESP",
    "121.6MEGS-P",
    "darkMEGS-P",
    "q0ESP",
    "q1ESP",
    "q2ESP",
    "q3ESP",
    "CMLat",
    "CMLon",
    "x_cool_proxy",
    "oldXRSB_proxy",
]

# Assign column names to the DataFrame
df.columns = column_names

# Convert the 'HHMM' column to datetime
df["DateTime"] = df["HHMM"].apply(lambda x: datetime.strptime(f"{x:04d}", "%H%M").time())
df["DateTime"] = df["DateTime"].apply(lambda t: datetime.combine(date.date(), t))

# Set the DateTime as the index
df.set_index("DateTime", inplace=True)
df.drop(columns=["HHMM"], inplace=True)

print(df)
