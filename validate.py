# validate_and_delete.py
import os
import json
import rasterio
import logging
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ========== Environment Suppression ==========
os.environ["CPL_LOG"] = "OFF"
os.environ["CPL_DEBUG"] = "OFF"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["CPL_SUPPRESS_WARNINGS"] = "YES"

warnings.filterwarnings("ignore")
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("fiona").setLevel(logging.ERROR)

# ========== Settings ==========
CONFIG_PATH = "config.json"
EXPECTED_LABEL_BANDS = 1

# ========== Load Config ==========
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OUTPUT_DIR = config["OUTPUT_DIR"]
COUNTRY = config["COUNTRY_NAME"]
YEAR = config["START_DATE"][:4]

# ========= Load CHUNKS from config ==========
raw_chunks = config.get("CHUNKS")
if raw_chunks is None:
    raise ValueError("CHUNKS not defined in config.json")

CHUNKS = {}
for chunk_name, band_indices in raw_chunks.items():
    CHUNKS[chunk_name] = [f"A{i:02d}" for i in band_indices]

# ========== Logging Setup ==========
class RedFormatter(logging.Formatter):
    RED = '\033[31m'
    RESET = '\033[0m'
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(RedFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(handlers=[handler], level=logging.INFO)

# ========== Validation Function ==========
def validate_file(args):
    full_path, expected_bands = args
    try:
        if not os.path.exists(full_path):
            return full_path
        with rasterio.open(full_path) as src:
            if src.count != expected_bands:
                return full_path
    except Exception:
        return full_path
    return None

# ========== Main ==========
def main():
    cores = max(1, cpu_count() - 1)

    for chunk_name, bands in CHUNKS.items():
        expected_bands = EXPECTED_LABEL_BANDS + len(bands)
        chunk_dir = os.path.join(OUTPUT_DIR, COUNTRY, YEAR, chunk_name)

        if not os.path.isdir(chunk_dir):
            logging.warning(f"{chunk_dir} does not exist.")
            continue

        tif_files = sorted([
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
            if f.endswith(".tif")
        ])

        with Pool(cores) as pool:
            results = list(tqdm(
                pool.imap(validate_file, [(f, expected_bands) for f in tif_files]),
                total=len(tif_files),
                desc=f"Deleting invalid in {chunk_name}",
                unit="file"
            ))

        deleted = 0
        for path in results:
            if path is not None and os.path.exists(path):
                try:
                    os.remove(path)
                    deleted += 1
                    logging.warning(f"Deleted invalid file: {path}")
                except Exception as e:
                    logging.error(f"Failed to delete {path}: {e}")

        logging.info(f"{chunk_name}: {deleted} invalid or corrupted files deleted.")

if __name__ == "__main__":
    main()