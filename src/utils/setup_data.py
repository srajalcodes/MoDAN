import shutil
import zipfile
from pathlib import Path
import requests

ZENODO_URL = "https://zenodo.org/api/records/21221081/files-archive"

ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_ZIP = ROOT / "MoDAN_Zenodo_Archive.zip"
TEMP_DIR = ROOT / "_temp_zenodo"

DESTINATIONS = {
    # Model
    "best_biomodal_model.pt": ROOT / "models",

    # Embeddings
    "chemberta_embeddings.pkl": ROOT / "data" / "embeddings",
    "esm2_embeddings.pkl": ROOT / "data" / "embeddings",
    "biobert_drug_embeddings.pkl": ROOT / "data" / "embeddings",

    # Processed DrugBank
    "train_cold.csv": ROOT / "data" / "processed",
    "test_cold_S1.csv": ROOT / "data" / "processed",
    "test_cold_S2.csv": ROOT / "data" / "processed",

    # BIOSNAP
    "BIOSNAP_train_cold.csv": ROOT / "data" / "benchmark_splits",
    "BIOSNAP_test_cold_S1.csv": ROOT / "data" / "benchmark_splits",
    "BIOSNAP_test_cold_S2.csv": ROOT / "data" / "benchmark_splits",

    # ZhangDDI
    "ZhangDDI_train_cold.csv": ROOT / "data" / "benchmark_splits",
    "ZhangDDI_test_cold_S1.csv": ROOT / "data" / "benchmark_splits",
    "ZhangDDI_test_cold_S2.csv": ROOT / "data" / "benchmark_splits",

    # Metadata
    "final_drug_nodes.csv": ROOT / "data" / "metadata",
}

def download_zip():

    print("=" * 60)
    print("Downloading reproducibility package from Zenodo...")
    print("=" * 60)

    response = requests.get(ZENODO_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    downloaded = 0

    with open(DOWNLOAD_ZIP, "wb") as f:

        for chunk in response.iter_content(chunk_size=8192):

            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = downloaded * 100 / total
                    print(f"\rDownloading... {percent:.1f}%", end="")
    print("\nDownload complete.\n")

def extract_zip():

    print("Extracting archive...")
    with zipfile.ZipFile(DOWNLOAD_ZIP, "r") as z:
        z.extractall(TEMP_DIR)
    print("Extraction complete.\n")

def install_files():
   print("Installing files...\n")
    for file_name, destination in DESTINATIONS.items():
        destination.mkdir(parents=True, exist_ok=True)
        matches = list(TEMP_DIR.rglob(file_name))

        if len(matches) == 0:
            print(f"[WARNING] Missing: {file_name}")
            continue

        src = matches[0]
        dst = destination / file_name
        shutil.move(str(src), str(dst))

        print(f"[OK] {file_name}")
    print()

def cleanup():

    print("Cleaning temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    if DOWNLOAD_ZIP.exists():
        DOWNLOAD_ZIP.unlink()
    print("Done.\n")


def main():

    download_zip()
    extract_zip()
    install_files()
    cleanup()

    print("=" * 60)
    print("MoDAN data installation completed successfully.")
    print("=" * 60)
    print("Repository is ready.")
    print("You can now run:")
    print("python src/training/train_modan.py")

if __name__ == "__main__":
    main()