# Run this to combine the .npy files that make up the flow dataset

import numpy as np
import pandas as pd
import os

# Define the directory containing the .npy files
data_dir = '/media/raid/jkruse/Temporal-Rich-Club/Human_Mobility_Flows/weekly_flows/ct2ct/merged_flows/split_npy_files/'
# Define column names (add the original column names here)
columns = ["geoid_o", "geoid_d", "flows", "t", "i", "j"]


# Initialize an empty list to store the loaded data
all_data = []

# Iterate over all .npy files in the directory
for file in sorted(os.listdir(data_dir)):
    if file.endswith(".npy"):
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file_path}...")
        
        # Load the numpy file
        chunk = np.load(file_path)
        
        # Convert to DataFrame and append to the list
        df_chunk = pd.DataFrame(chunk, columns=columns)
        all_data.append(df_chunk)

# Combine all chunks into one DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save the combined DataFrame as a CSV file
output_csv = '/media/raid/jkruse/Temporal-Rich-Club/Human_Mobility_Flows/weekly_flows/ct2ct/merged_flows/WICTs_allyears_test.csv'

combined_data.to_csv(output_csv, index=False)

print(f"Combined data saved to {output_csv}")
