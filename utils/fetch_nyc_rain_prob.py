import os
import pandas as pd
import requests
import pickle
from tqdm import tqdm

# --- Config ---
NYC_LAT = 40.71278
NYC_LON = 74.00594
BASE = os.path.dirname(os.path.abspath(__file__))
PICKLE_FILE = os.path.join(BASE, "nyc_rain_probabilities.pkl")

# --- Weather Query ---
def get_rain_probability_for_date(date, lat=NYC_LAT, lon=NYC_LON):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "precipitation_probability",
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "hourly" in data and "precipitation_probability" in data["hourly"]:
            probs = data["hourly"]["precipitation_probability"]
            clean_probs = [p for p in probs if p is not None]
            return round(sum(clean_probs) / len(clean_probs), 2) if clean_probs else None
        else:
            return None
    except Exception as e:
        print(f"Error fetching for {date}: {e}")
        return None


# --- Main Function ---
def fetch_rain_for_unique_dates(df):
    # Extract unique dates (as strings)
    dates = pd.to_datetime(df["pickup_datetime"]).dt.strftime("%Y-%m-%d")
    unique_dates = sorted(dates.unique())

    rain_dict = {}

    for date in tqdm(unique_dates):
        rain_dict[date] = get_rain_probability_for_date(date)

    # Save as pickle
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(rain_dict, f)

    print(f"Saved {len(rain_dict)} dates to {PICKLE_FILE}")
    return rain_dict

if __name__ == "__main__":
    fpath = os.path.join(BASE, "../data", "nyc-taxi-trip-duration", "train_processed.csv")
    df = pd.read_csv(fpath)
    fetch_rain_for_unique_dates(df)