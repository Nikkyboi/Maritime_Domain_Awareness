import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (in degrees).
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c




if __name__ == "__main__":
    # Example usage of haversine function
    lat1, lon1 = 52.2296756, 21.0122287  # Warsaw
    lat2, lon2 = 41.8919300, 12.5113300  # Rome

    distance = haversine(lat1, lon1, lat2, lon2)
    print(f"Distance between Warsaw and Rome: {distance:.2f} meters")