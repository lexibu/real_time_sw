import numpy as np
from datetime import datetime, timedelta
import geopack.geopack as gp


def day_of_year(date):
    """Calculate the day of the year."""
    start_of_year = datetime(date.year, 1, 1)
    return (date - start_of_year).days + 1


def solar_declination(day_of_year):
    """Calculate the solar declination angle in radians."""
    return np.radians(23.44) * np.sin(np.radians((360 / 365.25) * (day_of_year - 81)))


def dipole_tilt_angle(date):
    """Calculate the dipole tilt angle of the Earth at any given time."""
    day = day_of_year(date)
    delta = solar_declination(day)

    # Calculate the time of day in fractional hours
    time_of_day = date.hour + date.minute / 60 + date.second / 3600
    omega = np.radians(15 * (time_of_day - 12))  # Hour angle in radians

    beta = np.arcsin(np.sin(delta) * np.sin(omega))
    dipole_tilt = np.degrees(np.arcsin(np.sin(delta) * np.cos(beta)))

    return dipole_tilt


# Example usage:
date = datetime.utcnow()
tilt_angle = dipole_tilt_angle(date)
print(f"Dipole tilt angle at {date} is {tilt_angle:.2f} degrees")

# Get the date in unix timestamp format
date = datetime.utcnow()
# Add the timezone offset to the date
date = date + timedelta(hours=4)
unix_timestamp = date.timestamp()

tilt_angle_gp = gp.recalc(unix_timestamp)
# Convert the tilt angle to degrees
tilt_angle_gp = np.degrees(tilt_angle_gp)

print(f"Dipole tilt angle at {date} is {tilt_angle_gp:.2f} degrees")
