#!/usr/bin/env python
'''
map.py
PyGMT Mapping script
Sandy H. S. Herho <sandy.herho@email.ucr.edu>
2025/06/20
'''
import pygmt
import os

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = '../figs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the geographical region
    region = [90, 160, -20, 20]  # [West, East, South, North]
    
    # Load Earth relief data at 6 arc-minute resolution for the specified region
    grid = pygmt.datasets.load_earth_relief(resolution='06m', region=region)
    
    # Create a new figure
    fig = pygmt.Figure()
    
    # Plot the relief data with Mercator projection for rectangular grid
    # Using 'M' for Mercator projection with 15cm width
    fig.grdimage(
        grid=grid, 
        projection="M15c",
        region=region,
        cmap="geo",
        frame=["a10f5", "WSne"]  # Grid every 10°, annotations every 5°
    )
    
    # Add colorbar with elevation labels
    fig.colorbar(frame=["a2500", "x+lElevation", "y+lm"])
    
    # Display the figure
    fig.show()
    
    # Save the figure as PNG with high resolution
    fig.savefig(f'{output_dir}/map.png', dpi=400)
