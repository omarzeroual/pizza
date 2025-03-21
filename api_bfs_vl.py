# https://data.bfs.admin.ch

import requests
import json
import pandas as pd

# API Endpoint
api_url = "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0304010000_202/px-x-0304010000_202.px"

# Define payload to request data for 2022, all regions, and median wage
payload = {
    "query": [
        {"code": "Jahr", "selection": {"filter": "item", "values": ["2022"]}},  # Select year 2022
        {"code": "Grossregion", "selection": {"filter": "item", "values": ["-1", "1", "2", "3", "4", "5", "6", "7"]}},
        # All regions
        {"code": "Zentralwert und andere Perzentile", "selection": {"filter": "item", "values": ["1"]}}  # Median wage
    ],
    "response": {"format": "json"}  # Request JSON format
}

# Send request
response = requests.post(api_url, json=payload)

# Check if request was successful
if response.status_code == 200:
    data = response.json()

    # Region code mapping
    region_map = {
        "-1": "Schweiz",
        "1": "Région lémanique",
        "2": "Espace Mittelland",
        "3": "Nordwestschweiz",
        "4": "Zürich",
        "5": "Ostschweiz",
        "6": "Zentralschweiz",
        "7": "Ticino"
    }

    # Extract and format the data
    formatted_data = []
    for entry in data["data"]:
        year, region_code, percentile = entry["key"]
        wage = entry["values"][0]

        formatted_data.append({
            "Year": year,
            "Region": region_map.get(region_code, "Unknown"),
            "Median Monthly Wage (CHF)": int(wage)
        })

    # Convert to JSON and print
    formatted_json = json.dumps(formatted_data, indent=4, ensure_ascii=False)
    print(formatted_json)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(formatted_data)
    print(df)

    # Optionally, save to CSV
    df.to_csv("median_wages_2022.csv", index=False)

else:
    print(f"Error: {response.status_code}, {response.text}")
