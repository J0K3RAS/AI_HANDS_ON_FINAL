import os
import pandas as pd
import re

def get_cords(data):
    pattern = r'POINT \(([-\d.]+) ([-\d.]+)\)'
    match = re.search(pattern, data)
    if match:
        longitude, latitude = match.groups()
        return float(longitude), float(latitude)
    return None

def get_nyc_box(data):
    latitude = []
    longitude = []
    for point in data['the_geom']:
        longitude_, latitude_ = get_cords(point)
        longitude.append(longitude_)
        latitude.append(latitude_)

    return min(latitude), max(latitude), min(longitude), max(longitude)

if __name__ == "__main__":
    # https://data.cityofnewyork.us/City-Government/Digital-City-Map-Shapefile/m2vu-mgzw
    base_path = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(base_path, '../data', 'DCM_StreetNameChanges_Points_20250622.csv')
    data = pd.read_csv(fpath)
    print(get_nyc_box(data))
