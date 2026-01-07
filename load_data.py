import os
import pandas as pd
from pymongo import MongoClient

# Step 1: Connect to MongoDB (local)
client = MongoClient("mongodb://localhost:27017/")
db = client["snrt_db"]  # Name of your MongoDB database

# Step 2: Folder where CSVs are stored
DATA_FOLDER = "data"

# Step 3: List of files and the name of the MongoDB collections
file_map = {
    "MgtDB.Material_station.csv": "material_station",
    "MgtDB.Material_station_Designation.csv": "material_designation",
    "MgtDB.Material_station_Eqpt.csv": "material_eqpt",
    "MgtDB.Material_station_Site.csv": "material_site",
    "MgtDB.Material_station_family.csv": "material_family",
    "MgtDB.Material_station_Mark.csv": "material_mark",
    "MgtDB.Material_station_subfamily.csv": "material_subfamily",
    "MgtDB.Material_station_Service.csv": "material_service"
}

# Step 4: Loop through and import data
for filename, collection_name in file_map.items():
    file_path = os.path.join(DATA_FOLDER, filename)
    
    print(f"Importing {filename} into collection '{collection_name}'...")
    
    # Read CSV into DataFrame
    df = pd.read_csv(file_path)
    
    # Convert to dictionary
    data_dict = df.to_dict(orient="records")
    
    # Insert into MongoDB
    collection = db[collection_name]
    collection.delete_many({})  # Clear existing data
    if data_dict:
        collection.insert_many(data_dict)

print("All files successfully imported into MongoDB.")
