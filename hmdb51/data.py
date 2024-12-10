import os
import csv

# Path to your dataset root directory
dataset_path = "./"

# Output annotation file
output_file = "annotations.csv"

# Collect data
data = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for video_file in os.listdir(category_path):
            if video_file.endswith(".avi"):  # Adjust for your video formats
                video_path = os.path.join(category_path, video_file)
                data.append([video_path, category])

# Write to CSV
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File Path", "Category"])  # Header
    writer.writerows(data)

print(f"Annotation file saved to {output_file}")
