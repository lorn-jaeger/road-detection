import ee
import requests
import os

ee.Authenticate()
ee.Initialize()

DATE = "2020-01-15"

region = ee.Geometry.Rectangle([83.2, 17.6, 83.35, 17.8])

OUT = "sentinel2_test.tif"

RGB = ["B4", "B3", "B2"]


collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(DATE, DATE)  
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)


image = collection.median().select(RGB)

url = image.getDownloadURL({
    "scale": 10,
    "crs": "EPSG:4326",
    "region": region,
    "format": "GEO_TIFF"
})


r = requests.get(url, stream=True)
r.raise_for_status()

with open(OUT, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

