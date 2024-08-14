import numpy as np
import time
from skyfield.api import load
from astropy.coordinates import (
    CartesianRepresentation,
    ITRS,
    GCRS,
    CartesianDifferential,
    EarthLocation,
)
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import datetime


def get_gsm_coordinates():
    # Load ephemeris data
    ts = load.timescale()
    planets = load("de421.bsp")
    earth = planets["earth"]
    moon = planets["moon"]

    # Get current time
    t = ts.now()

    # Get the Moon's GCRS position
    moon_gcrs = moon.at(t).position.au
    moon_pos = CartesianRepresentation(moon_gcrs * u.au).to(u.km)

    # Get Earth's position in ITRS
    earth_location = EarthLocation.from_geocentric(0, 0, 0, unit="km")
    earth_itrs = ITRS(
        CartesianRepresentation([0, 0, 0] * u.km), obstime=Time(t.utc_iso())
    )

    # Get the GCRS position of the Earth
    earth_gcrs = earth.at(t).position.au
    earth_pos = CartesianRepresentation(earth_gcrs * u.au).to(u.km)

    # For simplicity, assume that GSM = GCRS in this example (a simplification)
    # In reality, you'd need to apply the proper rotation matrix to get the GSM coordinates.
    moon_gsm = moon_pos - earth_pos
    earth_gsm = earth_pos

    return earth_gsm, moon_gsm


def display_gsm_coordinates():
    while True:
        earth_gsm, moon_gsm = get_gsm_coordinates()
        current_time = datetime.datetime.utcnow()

        print(f"Time (UTC): {current_time}")
        print(f"Earth GSM Coordinates: {earth_gsm.xyz}")
        print(f"Moon GSM Coordinates: {moon_gsm.xyz}\n")

        # Sleep for an hour before updating
        time.sleep(3600)


if __name__ == "__main__":
    display_gsm_coordinates()
