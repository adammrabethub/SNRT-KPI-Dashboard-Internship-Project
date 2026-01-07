import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["snrt_db"]

station_df = pd.DataFrame(list(db.material_station.find()))
family_df = pd.DataFrame(list(db.material_family.find()))
site_df = pd.DataFrame(list(db.material_site.find()))
eqpt_df = pd.DataFrame(list(db.material_eqpt.find()))

print("\n material_station sample:\n", station_df.head(3))
print("\n material_family sample:\n", family_df.head(3))
print("\n material_site sample:\n", site_df.head(3))
print("\n material_eqpt sample:\n", eqpt_df.head(3))
