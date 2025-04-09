"""
Module Name: get_api.py
Description: This module gets the wage data from the BFS API
Author: Valérie
Date: 2025-04-01
Version: 1.0
"""

# https://data.bfs.admin.ch

import requests
import json
import pandas as pd

# API Endpoint
API_URL = "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0304010000_202/px-x-0304010000_202.px"

# Region code mapping
REGION_MAP = {
    "-1": "Schweiz",
    "1": "Région lémanique",
    "2": "Espace Mittelland",
    "3": "Nordwestschweiz",
    "4": "Zürich",
    "5": "Ostschweiz",
    "6": "Zentralschweiz",
    "7": "Ticino"
}

def get_wage_data(year="2022"):
    """Get wage data from the BFS API """
    payload = {
        "query": [
            {"code": "Jahr", "selection": {"filter": "item", "values": [year]}},
            {"code": "Grossregion", "selection": {"filter": "item", "values": list(REGION_MAP.keys())}},
            {"code": "Zentralwert und andere Perzentile", "selection": {"filter": "item", "values": ["1"]}}
        ],
        "response": {"format": "json"}
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def process_data(data):
    """Convert API response to dictionaries """
    if not data or "data" not in data:
        print("No valid data found in API response.")
        return []

    return [
        {
            "Year": entry["key"][0],
            "Region": REGION_MAP.get(entry["key"][1], "Unknown"),
            "Median Monthly Wage (CHF)": int(entry["values"][0])
        }
        for entry in data["data"]
    ]

def save_data(df):
    """Save Data Frame to CSV and JSON """
    df.to_csv("data/median_wages_2022.csv", index=False, sep = ";")
    df.to_json("data/median_wages_2022.json", orient="records", indent=4)
    print("Data saved successfully.")

def main():
    """Main function to get, process, and save wage data """
    data = get_wage_data()
    if data:
        formatted_data = process_data(data)
        if formatted_data:
            df = pd.DataFrame(formatted_data)
            print(df.head())  # Display first few rows
            save_data(df)

if __name__ == "__main__":
    main()



