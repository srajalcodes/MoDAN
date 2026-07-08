import shutil
import zipfile
import requests
import sys
from pathlib import Path

# --- CONFIGURATION ---
# DIRECT link to the exact file to prevent the "Zip inside a Zip" issue
ZENODO_URL = "https://zenodo.org/records/21221081/files/MoDAN_Zenodo_Archive.zip?download=1"

# Repository root directory
ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_ZIP = ROOT / "MoDAN_Zenodo_Archive.zip"
TEMP_DIR = ROOT / "_temp_zenodo"

# Expected files and their destination folders
DESTINATIONS = {
    "modan_final_model.pt": ROOT / "models",
    "chemberta_embeddings.pkl": ROOT / "data" / "embeddings",
    "esm2_embeddings.pkl": ROOT / "data" / "embeddings",
    "biobert_drug_embeddings.pkl": ROOT / "data" / "embeddings",
    "train_cold.csv": ROOT / "data" / "processed",
    "test_cold_S1.csv": ROOT / "data" / "processed",
    "test_cold_S2.csv": ROOT / "data" / "processed",
    "BIOSNAP_train_cold.csv": ROOT / "data" / "benchmark_splits",
    "BIOSNAP_test_cold_S1.csv": ROOT / "data" / "benchmark_splits",
    "BIOSNAP_test_cold_S2.csv": ROOT / "data" / "benchmark_splits",
    "ZhangDDI_train_cold.csv": ROOT / "data" / "benchmark_splits",
    "ZhangDDI_test_cold_S1.csv": ROOT / "data" / "benchmark_splits",
    "ZhangDDI_test_cold_S2.csv": ROOT / "data" / "benchmark_splits",
    "final_drug_nodes.csv": ROOT / "data" / "metadata",
}

def download_zip():
    print("=" * 60)
    print("Downloading reproducibility package from Zenodo...")
    print("=" * 60)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(ZENODO_URL, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to connect to Zenodo: {e}")
        sys.exit(1)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(DOWNLOAD_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    print(f"\rDownloading... {percent:.1f}%", end="")
                    
    print("\n✅ Download complete.\n")

def extract_zip():
    print("Extracting archive...")
    if not DOWNLOAD_ZIP.exists():
        print("[ERROR] Zip file not found!")
        sys.exit(1)
        
    with zipfile.ZipFile(DOWNLOAD_ZIP, "r") as z:
        z.extractall(TEMP_DIR)
    print("✅ Extraction complete.\n")

def install_files():
    print("Installing files into correct directories...\n")
    missing_files = []
    
    for file_name, destination in DESTINATIONS.items():
        destination.mkdir(parents=True, exist_ok=True)
        
        matches = list(TEMP_DIR.rglob(file_name))

        # FALLBACK: If looking for modan_final_model.pt, check if it's still named best_biomodal_model.pt
        if len(matches) == 0 and file_name == "modan_final_model.pt":
            matches = list(TEMP_DIR.rglob("best_biomodal_model.pt"))
            if len(matches) > 0:
                print("[INFO] Found older model name 'best_biomodal_model.pt'. Renaming automatically...")

        if len(matches) == 0:
            print(f"[❌ ERROR] Missing from archive: {file_name}")
            missing_files.append(file_name)
            continue

        src = matches[0]
        dst = destination / file_name
        shutil.move(str(src), str(dst))
        print(f"[✅ OK] Installed: {file_name} -> {destination.name}/")

    return missing_files

def cleanup():
    print("\nCleaning temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    if DOWNLOAD_ZIP.exists():
        DOWNLOAD_ZIP.unlink()
    print("✅ Cleanup done.\n")

def main():
    download_zip()
    extract_zip()
    missing = install_files()
    cleanup()

    print("=" * 60)
    if missing:
        print(f"⚠️ INSTALLATION INCOMPLETE. Missing {len(missing)} files.")
        sys.exit(1)
    else:
        print("MoDAN data installation completed successfully!")
        print("Repository is ready for evaluation.")
    print("=" * 60)

if __name__ == "__main__":
    main()