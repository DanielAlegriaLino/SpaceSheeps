"""
Satellite Distance Calculator

Takes an observer position (latitude, longitude, altitude) and returns
distances to visible satellites using the N2YO public API.

Usage:
    python distance_to_satelites.py --lat 41.39 --lon 2.17
    python distance_to_satelites.py --lat 41.39 --lon 2.17 --alt 0 --radius 70 --category 18

Get a free API key at: https://www.n2yo.com/api/

Set your API key via environment variable:
    export N2YO_API_KEY="your-api-key-here"
"""

import argparse
import math
import os
import urllib.request
import json

EARTH_RADIUS_KM = 6371.0

N2YO_BASE_URL = "https://api.n2yo.com/rest/v1/satellite"

# N2YO satellite categories:
# 0=all, 1=brightest, 2=ISS, 3=weather, 5=GPS, 6=science,
# 12=Starlink, 15=education, 18=amateur radio, 22=GPS operational
DEFAULT_CATEGORY = 18


def get_api_key():
    return os.environ.get("N2YO_API_KEY", "4GULL3-JSKLBH-W8J5TM-5NCI")


def fetch_satellites_above(lat, lon, alt, radius, category, api_key):
    """Fetch satellites above a given observer position using N2YO API."""
    url = (
        f"{N2YO_BASE_URL}/above/{lat}/{lon}/{alt}/{radius}/{category}"
        f"&apiKey={api_key}"
    )
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def compute_distance_km(obs_lat, obs_lon, obs_alt_km, sat_lat, sat_lon, sat_alt_km):
    """
    Compute straight-line distance between observer and satellite
    by converting both to ECEF (Earth-Centered, Earth-Fixed) coordinates.
    """
    obs_lat_r = math.radians(obs_lat)
    obs_lon_r = math.radians(obs_lon)
    sat_lat_r = math.radians(sat_lat)
    sat_lon_r = math.radians(sat_lon)

    r_obs = EARTH_RADIUS_KM + obs_alt_km
    r_sat = EARTH_RADIUS_KM + sat_alt_km

    # Observer ECEF
    ox = r_obs * math.cos(obs_lat_r) * math.cos(obs_lon_r)
    oy = r_obs * math.cos(obs_lat_r) * math.sin(obs_lon_r)
    oz = r_obs * math.sin(obs_lat_r)

    # Satellite ECEF
    sx = r_sat * math.cos(sat_lat_r) * math.cos(sat_lon_r)
    sy = r_sat * math.cos(sat_lat_r) * math.sin(sat_lon_r)
    sz = r_sat * math.sin(sat_lat_r)

    return math.sqrt((ox - sx) ** 2 + (oy - sy) ** 2 + (oz - sz) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Calculate distances to satellites above your position.")
    parser.add_argument("--lat", type=float, required=True, help="Observer latitude (degrees)")
    parser.add_argument("--lon", type=float, required=True, help="Observer longitude (degrees)")
    parser.add_argument("--alt", type=float, default=0, help="Observer altitude in meters above sea level (default: 0)")
    parser.add_argument("--radius", type=int, default=70, help="Search radius in degrees (0-90, default: 70)")
    parser.add_argument("--category", type=int, default=DEFAULT_CATEGORY, help=f"N2YO satellite category (default: {DEFAULT_CATEGORY})")
    args = parser.parse_args()

    api_key = get_api_key()
    obs_alt_km = args.alt / 1000.0

    print(f"Observer position: lat={args.lat}, lon={args.lon}, alt={args.alt}m")
    print(f"Search radius: {args.radius}°  |  Category: {args.category}")
    print("Fetching satellites...\n")

    data = fetch_satellites_above(args.lat, args.lon, args.alt, args.radius, args.category, api_key)

    sat_count = data.get("info", {}).get("satcount", 0)
    satellites = data.get("above", [])

    if sat_count == 0 or not satellites:
        print("No satellites found above your position.")
        return

    results = []
    for sat in satellites:
        sat_lat = sat.get("satlat", 0)
        sat_lon = sat.get("satlng", 0)
        sat_alt_km = sat.get("satalt", 0)
        name = sat.get("satname", "Unknown")
        norad_id = sat.get("satid", "?")

        distance = compute_distance_km(args.lat, args.lon, obs_alt_km, sat_lat, sat_lon, sat_alt_km)
        results.append((distance, name, norad_id, sat_lat, sat_lon, sat_alt_km))

    results.sort(key=lambda x: x[0])

    print(f"{'Satellite':<30} {'NORAD ID':>10} {'Alt (km)':>10} {'Distance (km)':>15}")
    print("-" * 70)
    for distance, name, norad_id, sat_lat, sat_lon, sat_alt_km in results:
        print(f"{name:<30} {norad_id:>10} {sat_alt_km:>10.1f} {distance:>15.1f}")

    print(f"\nTotal satellites found: {len(results)}")


if __name__ == "__main__":
    main()
