#!/usr/bin/env python
'''
process_climate_data.py
Script to combine and process climate data from multiple CSV files
Keeps only dates that exist in ITF data
Converts dates to decimal years in 'time' column while keeping original 'Date' column
Removes NaN values and -9999.0 placeholders
Sandy H. S. Herho <sandy.herho@email.ucr.edu>
2025/06/20
'''
import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_climate_data():
    # Define input and output directories
    input_dir = '../raw_data'
    output_dir = '../processed_data'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Read the CSV files
    print("Reading data files...")
    
    # Read DMI data
    dmi_path = os.path.join(input_dir, 'dmi.csv')
    dmi_df = pd.read_csv(dmi_path)
    dmi_df.columns = dmi_df.columns.str.strip()  # Remove leading/trailing whitespace
    print(f"DMI data loaded: {dmi_df.shape}")
    
    # Read MEI v2 data
    meiv2_path = os.path.join(input_dir, 'meiv2.csv')
    meiv2_df = pd.read_csv(meiv2_path)
    meiv2_df.columns = meiv2_df.columns.str.strip()  # Remove leading/trailing whitespace
    print(f"MEI v2 data loaded: {meiv2_df.shape}")
    
    # Read ITF time series data
    itf_path = os.path.join(input_dir, 'itf_ts.csv')
    itf_df = pd.read_csv(itf_path)
    itf_df.columns = itf_df.columns.str.strip()  # Remove leading/trailing whitespace
    print(f"ITF data loaded: {itf_df.shape}")
    
    # Rename the 'time' column in ITF data to 'Date' for consistency
    itf_df = itf_df.rename(columns={'time': 'Date'})
    
    # Convert Date columns to datetime for proper sorting and merging
    print("\nConverting date columns to datetime format...")
    dmi_df['Date'] = pd.to_datetime(dmi_df['Date'])
    meiv2_df['Date'] = pd.to_datetime(meiv2_df['Date'])
    itf_df['Date'] = pd.to_datetime(itf_df['Date'])
    
    # Merge all datasets using ITF dates as reference (inner join on ITF data)
    print("\nMerging datasets using ITF dates as reference...")
    
    # Start with ITF data as the base
    merged_df = itf_df.copy()
    
    # Merge with DMI data (left join to keep only ITF dates)
    merged_df = pd.merge(merged_df, dmi_df, on='Date', how='left')
    print(f"After merging with DMI: {merged_df.shape}")
    
    # Merge with MEI v2 data (left join to keep only ITF dates)
    merged_df = pd.merge(merged_df, meiv2_df, on='Date', how='left')
    print(f"After merging with MEI v2: {merged_df.shape}")
    
    # Sort by date
    merged_df = merged_df.sort_values('Date')
    
    # Replace -9999.0 with NaN
    print("\nReplacing -9999.0 values with NaN...")
    merged_df = merged_df.replace(-9999.0, np.nan)
    
    # Count missing values
    print("\nMissing values per column:")
    print(merged_df.isnull().sum())
    
    # Remove rows where ALL data columns are NaN (keeping Date and time columns)
    data_columns = [col for col in merged_df.columns if col not in ['Date', 'time']]
    rows_before = len(merged_df)
    merged_df = merged_df.dropna(subset=data_columns, how='all')
    rows_after = len(merged_df)
    print(f"\nRemoved {rows_before - rows_after} rows where all data values were NaN")
    
    # Display final statistics
    print(f"\nFinal dataset shape: {merged_df.shape}")
    print(f"Number of dates from ITF data retained: {len(merged_df)}")
    print(f"Columns: {', '.join(merged_df.columns)}")
    
    # Convert Date to decimal year
    print("\nConverting dates to decimal years...")
    def date_to_decimal_year(date):
        year = date.year
        # Get the day of year
        day_of_year = date.timetuple().tm_yday
        # Get total days in the year (accounting for leap years)
        days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        # Calculate decimal year
        return year + (day_of_year - 1) / days_in_year
    
    merged_df['time'] = merged_df['Date'].apply(date_to_decimal_year)
    
    # Reorder columns to have time first, then Date, then ITF columns, then others
    column_order = ['time', 'Date', 'itf_g', 'itf_t', 'itf_s', 'DMI_HadISST1.1', 'meiv2']
    # Only include columns that exist
    column_order = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[column_order]
    
    # Display time range (showing decimal years)
    print(f"\nTime range: {merged_df['time'].min():.4f} to {merged_df['time'].max():.4f}")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    # Save the processed data
    output_path = os.path.join(output_dir, 'combined_climate_data.csv')
    # Format the Date column as string in ISO format for CSV output
    merged_df_output = merged_df.copy()
    merged_df_output['Date'] = merged_df_output['Date'].dt.strftime('%Y-%m-%d')
    merged_df_output.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    # Display first and last few rows of the combined data
    print("\nFirst 5 rows of combined data:")
    print(merged_df.head())
    print("\nLast 5 rows of combined data:")
    print(merged_df.tail())
    
    return merged_df

if __name__ == "__main__":
    # Run the processing
    combined_data = process_climate_data()
    
    # Additional analysis (optional)
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(combined_data.describe())
