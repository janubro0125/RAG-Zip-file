import os
import requests

PDF_DIR = os.path.join("data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

PDFS = {
    "FAO_food_agriculture.pdf":
        "https://www.fao.org/3/i3434e/i3434e.pdf",

    "FAO_state_food_agriculture_2023.pdf":
        "https://www.fao.org/3/cc7724en/cc7724en.pdf",

    "FAO_state_food_agriculture_2023_brief.pdf":
        "https://www.fao.org/3/cc7937en/cc7937en.pdf",
}


for filename, url in PDFS.items():

    path = os.path.join(PDF_DIR, filename)

    print(f"\nDownloading: {filename}")

    try:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        if not response.content.startswith(b"%PDF"):
            print("Skipped: URL did not return a valid PDF")
            continue

        with open(path, "wb") as f:
            f.write(response.content)

        print(f"Saved: {path}")
        print(f"Size: {len(response.content) / 1024:.1f} KB")

    except Exception as e:
        print(f"ERROR: {e}")


print("\nPDF download process completed.")