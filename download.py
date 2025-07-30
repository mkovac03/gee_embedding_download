import ee
import os
import sys
import requests
import rasterio
import multiprocessing
from tqdm import tqdm
import time
import logging
from functools import partial
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
SOUTH = config.get('SOUTH', False)
MAX_RETRIES = config.get('MAX_RETRIES', 5)
BASE_WAIT = config.get('BASE_WAIT', 2.0)

START_DATE = config['START_DATE']
YEAR = START_DATE[:4]
RES = config['RES']
OUTPUT_DIR = config['OUTPUT_DIR']
GRID_ASSET = config.get('GRID_ASSET') or sys.exit("GRID_ASSET missing in config.json")

# ========== Parse CHUNKS ==========
raw_chunks = config.get("CHUNKS")
if raw_chunks is None:
    raise ValueError("CHUNKS not defined in config.json")

CHUNKS = {}
for chunk_name, band_indices in raw_chunks.items():
    CHUNKS[chunk_name] = [f"A{i:02d}" for i in band_indices]

# ========== Earth Engine Init ==========
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

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
def check_and_download(params):
    i, feature_obj, band_list, chunk_name = params
    feat = ee.Feature(feature_obj)

    centroid = feat.geometry().centroid(1)
    lon = centroid.coordinates().get(0)
    lat = centroid.coordinates().get(1)
    utm_zone = ee.Number(lon).add(180).divide(6).floor().add(1)
    is_southern = ee.Number(lat).lt(0)
    epsg_code = ee.Algorithms.If(is_southern, ee.Number(32700).add(utm_zone), ee.Number(32600).add(utm_zone))
    epsg = int(ee.Number(epsg_code).getInfo())

    out_dir = os.path.join(OUTPUT_DIR, COUNTRY_NAME, YEAR, chunk_name)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"google_embed_{COUNTRY_NAME}_{YEAR}_{RES}m_{epsg}_{chunk_name}_{i}.tif"
    outp = os.path.join(out_dir, filename)

    if os.path.exists(outp):
        return None

    try:
        feature = feat.transform(f"EPSG:{epsg}", 0.001)
        img, epsg = get_embedding_image(feature, epsg, band_list)
    except Exception as e:
        logging.error(f"[{i}] Failed to get embedding image: {e}")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            img_utm = img.reproject(f"EPSG:{epsg}", None, RES)
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

            if i == 0:
                with open(os.path.join(out_dir, "bands_used.txt"), "w") as f:
                    f.write("\n".join(band_list))

            return outp
        except Exception as e:
            logging.warning(f"[{i}] Retry {attempt+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(BASE_WAIT * (2**attempt))

    logging.error(f"[{i}] Failed to download after {MAX_RETRIES} retries: {filename}")
    return None

# ========== Main ==========
def main():
    logging.info(f"Loading grid from: {GRID_ASSET}")
    grid_fc = ee.FeatureCollection(GRID_ASSET)
    size = grid_fc.size().getInfo()
    features = grid_fc.toList(size)

    cores = max(1, multiprocessing.cpu_count() - 1)

    for chunk_name, band_list in CHUNKS.items():
        logging.info(f"Starting download for {chunk_name} ({len(band_list)} bands): {band_list}")
        out_dir = os.path.join(OUTPUT_DIR, COUNTRY_NAME, YEAR, chunk_name)
        os.makedirs(out_dir, exist_ok=True)

        existing_files = sorted([
            int(f.split("_")[-1].replace(".tif", ""))
            for f in os.listdir(out_dir)
            if f.endswith(".tif") and f.startswith(f"google_embed_{COUNTRY_NAME}_{YEAR}_{RES}m_")
        ])
        start_index = max(existing_files) + 1 if existing_files else 0

        all_params = [
            (i, features.get(i), band_list, chunk_name)
            for i in range(start_index, size)
        ]

        with multiprocessing.Pool(cores) as pool:
            results = []
            with tqdm(total=len(all_params), desc=f"Downloading {chunk_name}") as pbar:
                def update_bar(_):
                    pbar.update()

                for args in all_params:
                    res = pool.apply_async(check_and_download, args=(args,), callback=update_bar)
                    results.append(res)

                results = [r.get() for r in results]

        succeeded = sum(r is not None for r in results)
        logging.info(f"Chunk {chunk_name}: {succeeded}/{len(all_params)} tiles downloaded successfully.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
