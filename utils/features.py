import numpy as np
import pandas as pd
import requests


def get_rain_probability(lat, lon, datetime_str):
    # Parse full datetime and extract just the date
    date = pd.to_datetime(datetime_str).strftime("%Y-%m-%d")

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "precipitation_probability",
        "timezone": "auto"
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if "hourly" in data and "precipitation_probability" in data["hourly"]:
            rain_probs = data["hourly"]["precipitation_probability"]
            timestamps = data["hourly"]["time"]

            input_hour = pd.to_datetime(datetime_str).hour
            filtered_probs = [
                prob for time, prob in zip(timestamps, rain_probs)
                if pd.to_datetime(time).hour == input_hour
            ]

            if filtered_probs:
                return round(filtered_probs[0], 2)
            elif rain_probs:
                return round(sum(rain_probs) / len(rain_probs), 2)
            else:
                return 0
        else:
            return 0
    except Exception as e:
        print(f"Error fetching data: {e}")
        return 0

def rain_probability(df, lat, long, date):
    return df.apply(
        lambda row: get_rain_probability(row[lat], row[long], row[date]),
        axis=1
    )


def haversine_distance(df, lat1_col, lon1_col, lat2_col, lon2_col):
    """
    Calculate the great-circle distance between two points on the earth (specified in decimal degrees).

    Parameters:
    - df: The pandas DataFrame.
    - lat1_col, lon1_col: Column names for the first point's latitude and longitude.
    - lat2_col, lon2_col: Column names for the second point's latitude and longitude.

    Returns:
    - A pandas Series containing the distance in kilometers.
    """
    # Earth's radius in kilometers.
    R = 6371

    # Convert decimal degrees to radians
    lat1 = np.radians(df[lat1_col])
    lon1 = np.radians(df[lon1_col])
    lat2 = np.radians(df[lat2_col])
    lon2 = np.radians(df[lon2_col])

    # Difference in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

def unix_time(df, col):
    return pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.timestamp())

def is_weekend(df, col):
    date = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    return date.dt.dayofweek >= 5

def is_workhours(df, col):
    date = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    return date.dt.hour.between(9, 17)

def hour(df, col):
    date = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    return np.array([np.sin(2 * np.pi * date.dt.hour / 24), np.cos(2 * np.pi * date.dt.hour / 24)]).T
