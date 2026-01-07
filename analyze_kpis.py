import pymongo
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["snrt_db"]

# Load collections as DataFrames
station_df = pd.DataFrame(list(db.material_station.find()))
family_df  = pd.DataFrame(list(db.material_family.find()))
site_df    = pd.DataFrame(list(db.material_site.find()))
eqpt_df    = pd.DataFrame(list(db.material_eqpt.find()))

print("\nColumns in material_station:", station_df.columns.tolist())
print("Columns in material_family:", family_df.columns.tolist())
print("Columns in material_site:", site_df.columns.tolist())
print("Columns in material_eqpt:", eqpt_df.columns.tolist())

# Ensure station_id is consistent
def convert_station_id(df, col="station_id"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=[col], inplace=True)
        df[col] = df[col].astype(int)
    return df

station_df = convert_station_id(station_df, "_id")
station_df.rename(columns={"_id": "station_id"}, inplace=True)
family_df = convert_station_id(family_df)
site_df   = convert_station_id(site_df)
eqpt_df   = convert_station_id(eqpt_df)

# Merge data
station_family_df = pd.merge(station_df, family_df[["station_id", "family_id"]], on="station_id", how="left")
station_site_df   = pd.merge(station_df, site_df[["station_id", "site_id"]], on="station_id", how="left")
station_user_df   = station_df.copy()
eqpt_counts       = eqpt_df["station_id"].value_counts().sort_index()

# KPIs
stations_per_family = station_family_df['family_id'].value_counts().sort_index()
stations_per_site = station_site_df['site_id'].value_counts().sort_index()
stations_per_user = station_user_df['ConfigUser'].value_counts().sort_index()
equipment_count_per_station = eqpt_counts.sort_index()

# Save to Excel with proper headers
with pd.ExcelWriter("kpi_results.xlsx") as writer:
    stations_per_family.rename_axis("Family ID").to_frame(name="Station Count").to_excel(writer, sheet_name="Stations per Family")
    stations_per_site.rename_axis("Site ID").to_frame(name="Station Count").to_excel(writer, sheet_name="Stations per Site")
    stations_per_user.rename_axis("User").to_frame(name="Station Count").to_excel(writer, sheet_name="Stations per User")
    equipment_count_per_station.rename_axis("Station ID").to_frame(name="Equipment Count").to_excel(writer, sheet_name="Equipment Count")

# Plotting function with orientation control
def plot_kpi(data, xlabel, ylabel, title, filename, color, horizontal=False):
    plt.figure(figsize=(14, 10) if horizontal else (16, 6))
    if horizontal:
        data.sort_values().plot(kind='barh', color=color)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
    else:
        data.plot(kind='bar', color=color)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=90, fontsize=7)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot KPIs using SNRT colors
plot_kpi(stations_per_family, "Number of Stations", "Family ID", "Stations per Family", "stations_per_family.png", "forestgreen", horizontal=True)
plot_kpi(stations_per_site, "Number of Stations", "Site ID", "Stations per Site", "stations_per_site.png", "indianred", horizontal=True)
plot_kpi(stations_per_user, "User (ConfigUser)", "Number of Stations", "Stations per User", "stations_per_user.png", "darkorange", horizontal=False)
plot_kpi(equipment_count_per_station, "Number of Equipments", "Station ID", "Equipment Count per Station", "equipment_count_per_station.png", "royalblue", horizontal=True)

print("\nKPI analysis completed. Results saved.")
