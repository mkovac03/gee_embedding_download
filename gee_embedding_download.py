import ee
import os
import sys
import requests
import rasterio
import multiprocessing
from tqdm import tqdm
import time
import logging
from pyproj import CRS
import json

# ========== Logging Setup ==========
class RedFormatter(logging.Formatter):
    RED = '\033[31m'
    RESET = '\033[0m'
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)

class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

handler = logging.StreamHandler()
handler.setFormatter(RedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
handler.addFilter(WarningFilter())
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# ========== Load Configuration ==========
try:
    with open('config.json', 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)
except (json.JSONDecodeError, FileNotFoundError) as e:
    logging.error(f"Failed to load config.json: {e}")
    sys.exit(1)

COUNTRY_NAME = config.get('COUNTRY_NAME') or sys.exit("COUNTRY_NAME missing in config.json")
UTM_GRID_ASSET = config.get('UTM_GRID_ASSET') or sys.exit("UTM_GRID_ASSET missing in config.json")
SOUTH = config.get('SOUTH', False)
MAX_RETRIES = config.get('MAX_RETRIES', 5)
BASE_WAIT = config.get('BASE_WAIT', 2.0)

START_DATE = config['START_DATE']
END_DATE = config['END_DATE']
YEAR = START_DATE[:4]
RES = config['RES']
GRID_SIZE = config['GRID_SIZE']
ASSET_FOLDER = config['ASSET_FOLDER']
NO_DATA_VALUE = config['NO_DATA_VALUE']
OUTPUT_DIR = config['OUTPUT_DIR']

# ========== Define Band Chunks ==========
ALL_BANDS = [f"A{i:02d}" for i in range(1, 64)]
CHUNKS = {
    "bands_01_22": ALL_BANDS[0:22],   # A01–A22
    "bands_23_44": ALL_BANDS[22:44],  # A23–A44
    "bands_45_64": ALL_BANDS[44:63]   # A45–A63
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Earth Engine Init ==========
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

# ========== Load AOI and Grid ==========
country_fc = ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
utm_grid = ee.FeatureCollection(UTM_GRID_ASSET)

def asset_exists(aid):
    try:
        ee.data.getAsset(aid)
        return True
    except ee.EEException:
        return False

def export_zone_grids():
    logging.info("Starting export of per-zone grids…")
    utm_zone = utm_grid.filterBounds(country_fc)
    zones = list(utm_zone.aggregate_histogram('ZONE').getInfo().keys())
    zone_info = []
    for z in zones:
        zone = int(z)
        zone_poly = utm_grid.filter(ee.Filter.eq('ZONE', zone)).geometry()
        clipped = country_fc.geometry().intersection(zone_poly, 1)
        try:
            area_sq_m = clipped.area().getInfo()
        except Exception as e:
            logging.error(f"Error computing area for zone {zone}: {e}")
            continue
        if area_sq_m == 0:
            continue
        epsg = int(CRS.from_dict({'proj':'utm','zone':zone,'south':SOUTH}).to_authority()[1])
        crs = f"EPSG:{epsg}"
        grid = clipped.coveringGrid(crs, GRID_SIZE).map(lambda feat: feat.buffer(distance=-1)).map(lambda f: f.set('ZONE', zone))
        asset_id = f"{ASSET_FOLDER}{COUNTRY_NAME}_utm_grid_{GRID_SIZE}m_zone{zone}"
        if not asset_exists(asset_id):
            logging.info(f"  Exporting grid asset {asset_id}")
            task = ee.batch.Export.table.toAsset(
                collection=grid,
                description=f"{COUNTRY_NAME}_grid_{GRID_SIZE}_zone{zone}",
                assetId=asset_id
            )
            task.start()
            while task.active():
                logging.info(f"    Waiting for export of zone {zone}…")
                time.sleep(10)
            logging.info(f"  Export of zone {zone} completed.")
        else:
            logging.info(f"  Asset {asset_id} already exists, skipping export.")
        zone_info.append((zone, epsg, crs, asset_id))
    if not zone_info:
        logging.error("No UTM zones found; exiting.")
        sys.exit(1)
    logging.info("Finished exporting all zone grids.")
    return zone_info

# ========== Get Embedding Image with Label ==========
def get_embedding_image(feature, epsg, band_list):
    geometry = feature.geometry()

    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filter(ee.Filter.calendarRange(int(YEAR), int(YEAR), 'year'))

    embedding = collection.select(band_list).mean() \
        .clip(geometry) \
        .multiply(10000).toInt16()

    label = ee.Image("projects/ee-gmkovacs/assets/ext_wetland_2018_v2021_nw") \
        .clip(geometry) \
        .select([0], ['wetland_label']) \
        .toInt16()

    return label.addBands(embedding), epsg


# ========== Download Function ==========
def download_images(params):
    feat, idx, epsg, band_list, chunk_name = params
    feature = feat.transform(f'EPSG:{epsg}', 0.001)
    try:
        img, epsg = get_embedding_image(feature, epsg, band_list)
    except Exception as e:
        logging.error(f"Failed to get embedding image: {e}")
        return None

    out_dir = os.path.join(OUTPUT_DIR, COUNTRY_NAME, YEAR, chunk_name)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"google_embed_{COUNTRY_NAME}_{YEAR}_{RES}m_{epsg}_{chunk_name}_{idx}.tif"
    outp = os.path.join(out_dir, filename)

    if os.path.exists(outp):
        return outp

    for i in range(MAX_RETRIES):
        try:
            img_utm = img.reproject(f'EPSG:{epsg}', None, RES)
            url = img_utm.getDownloadURL({
                'region': feature.geometry(),
                'scale': RES,
                'format': 'GEO_TIFF'
            })
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with open(outp, 'wb') as f:
                f.write(r.content)
            with rasterio.open(outp, 'r+') as dst:
                dst.set_band_description(1, "wetland_label")
                for bi in range(2, dst.count + 1):
                    dst.set_band_description(bi, f"embedding_{bi-2}")
            if idx == 0:
                with open(os.path.join(out_dir, "bands_used.txt"), "w") as f:
                    f.write("\n".join(band_list))
            return outp
        except Exception as e:
            logging.warning(f"    Retry {i+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(BASE_WAIT * (2**i))
    logging.error(f"Failed to download after retries: {filename}")
    return None


# ========== Main ==========
def main():
    zone_info = export_zone_grids()
    cores = max(1, multiprocessing.cpu_count() - 1)

    for chunk_name, band_list in CHUNKS.items():
        logging.info(f"Starting download for {chunk_name} ({len(band_list)} bands): {band_list}")

        for zone, epsg, crs, aid in zone_info:
            fc = ee.FeatureCollection(aid)
            lst = fc.toList(fc.size()).getInfo()
            params = [(ee.Feature(lst[i]), i, epsg, band_list, chunk_name) for i in range(len(lst))]

            logging.info(f"  Total tasks for zone {zone}, {chunk_name}: {len(params)}")
            with multiprocessing.Pool(cores) as pool:
                results = list(tqdm(pool.imap(download_images, params), total=len(params)))
            succeeded = sum(r is not None for r in results)
            logging.info(f"Zone {zone} ({chunk_name}): {succeeded}/{len(params)} succeeded.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
