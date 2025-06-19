import os
import glob

# Base directory where the files are located
base_dir = "ATPC_0nubb"

# Define the pressure and diffusion categories
pressures = ["1bar", "5bar", "10bar", "15bar", "25bar"]
diffusions = {
    "1bar": ["0.0percent", "0.05percent", "0.1percent", "0.25percent","5percent", "nexus", "nodiff"],
    "5bar": ["0.05percent", "5percent", "nexus", "nodiff"],
    "10bar": ["0.05percent", "5percent", "nexus", "nodiff"],
    "15bar": ["0.05percent", "5percent", "nexus", "nodiff"],
    "25bar": ["0.05percent", "5percent", "nexus", "nodiff"]
}

# Loop through each pressure and corresponding diffusion folders
for pressure in pressures:
    for diffusion in diffusions[pressure]:
        folder_path = os.path.join(base_dir, pressure, diffusion)

        # Get all .h5 files in the current directory
        files = glob.glob(os.path.join(folder_path, "*.h5"))

        # Rename each file
        for file in files:
            base, ext = os.path.splitext(file)  # Split filename and extension
            new_name = f"{base}_4{ext}"  # Add _4 before .h5
            os.rename(file, new_name)  # Rename the file
            print(f"Renamed: {file} -> {new_name}")

