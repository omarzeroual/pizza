"""
Module Name: visualizer.py
Description: This modules is for analysing and creation visualizations based on the pizza price data.
Author: Omar
Date: 2025-04-05
Version: 1.0
"""
import pandas as pd
import numpy as np
import kaleido
import plotly.graph_objects as go

# Read cleaned and enriched data set with pizza prices
df = pd.read_csv("data/pizza_final.csv", delimiter=";", header=0)

def ratio_Map(output="save"):
    """
    Calculates the pizza price regional wage ratio and plots a map based on this data.

    Parameters:
    output (str): Determines the output method.
                  Use 'save' to save the map as an image.
                  Use 'show' to display the map interactively.

    Raises:
    ValueError: If the output parameter is not 'save' or 'show'.
    """

    # Group by city and calculate the mean pizza price
    mean_pizza_price = df.groupby('city')['price_chf'].mean()

    # Merge with the original DataFrame to get the median monthly wage and the coordinates of cities
    merged_df = pd.merge(mean_pizza_price, df[['city', 'Median Monthly Wage (CHF)']].drop_duplicates(), on='city')
    merged_df = pd.merge(merged_df, df[["city_N", "city_E", "city"]].drop_duplicates(), on='city')

    # Remove duplicate cities
    merged_df = merged_df.drop_duplicates(subset='city')

    # Calculate the ratio of pizza price divided by the median monthly wage in percent
    merged_df['ratio'] = merged_df['price_chf'] / merged_df['Median Monthly Wage (CHF)']*100
    # Save ratio also as string
    merged_df['ratio_str'] = round(merged_df['ratio'], 2).astype(str)

    # Transform the ratio for display on map
    merged_df['ratio_transformed'] = np.sqrt(merged_df['ratio'])
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    merged_df['ratio_scaled'] = scaler.fit_transform(merged_df[['ratio']])

    # Center coordinates for Switzerland
    SwitzerlandCenter = {"lat": 46.80111, "lon": 8.22667}

    # Set up plotly.graph_objects
    fig = go.Figure()

    # Create scatter map with transformed ratio for each city
    fig.add_trace(go.Scattermap(
        lat=merged_df["city_N"],
        lon=merged_df["city_E"],
        mode='markers',
        marker=go.scattermap.Marker(
            size=merged_df["ratio_transformed"]*50,
            color=merged_df["ratio"],
            colorscale='Viridis',  # Choose a color scale
            colorbar=dict(title="Ratio in %"),  # Add color bar
            cmin = 0.25,  # Set minimum value for color scale
            cmax = 0.51  # Set maximum value for color scale
    ),
        hoverinfo='none' # no hover since map is static
    ))

    # Add scatter trace for city names
    fig.add_trace(go.Scattermap(
        lat=merged_df["city_N"],
        lon=merged_df["city_E"] + 0.1,  # Adjust horizontal position for a margin between text and ratio marker
        mode='markers+text', # need to set markers to show text
        marker=go.scattermap.Marker(
            size=0, # set to 0 to not show marker
            color=None # set to None to not show marker
    ),
        text=merged_df["city"],
        textposition='middle right',
        hoverinfo='none'
    ))

    # Layout settings for scatter map
    fig.update_layout(
        title="Pizza Price to Wage Ratio in Swiss Cities",
        hovermode='closest',
        showlegend=False,
        map=dict(
            bearing=0,
            center=go.layout.map.Center(
                lat=SwitzerlandCenter['lat'],
                lon=SwitzerlandCenter["lon"]
            ),
            pitch=0,
            zoom=6.55,
            style="carto-positron-nolabels"
        ),
        width=800,  # Set width of the map
        height=600,  # Set height of the map
        margin = dict(l=1, r=1, t=30, b=0)  # Set margins to zero
    )

    if output == "save":
        # Set the dimensions and save the scatter map as an image
        fig.write_image("images/map.png", width=800, height=600)
    elif  output == "show":
        # show scatter map
        fig.show()
    else:
        # Raise an error for invalid output value
        raise ValueError("Invalid output value. Please use 'save' or 'show'.")


def main():
    ratio_Map("save")

if __name__ == "__main__":
    main()