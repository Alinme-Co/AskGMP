import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_DIR = "KnowledgeBase/mhra"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

DOCUMENTS = [
    {
        "folder": "data_integrity",
        "filename": "MHRA_GxP_Data_Integrity_Guidance.pdf",
        "page_url": (
            "https://www.gov.uk/government/publications/"
            "guidance-on-gxp-data-integrity"
        )
    },
    {
        "folder": "oos",
        "filename": "MHRA_Out_of_Specification_Investigations.pptx",
        "page_url": (
            "https://www.gov.uk/government/publications/"
            "out-of-specification-investigations"
        )
    },
    {
        "folder": "defective_products",
        "filename": "MHRA_Guide_to_Defective_Medicinal_Products.pdf",
        "page_url": (
            "https://www.gov.uk/government/publications/"
            "a-guide-to-defective-medicinal-products"
        )
    }
]


def find_download_url(page_url):
    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    candidates = []

    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        text = a.get_text(" ", strip=True).lower()

        # PDF
        if ".pdf" in href.lower():
            return href

        # PowerPoint / PPTX
        if ".ppt" in href.lower() or ".pptx" in href.lower():
            return href

        # GOV.UK asset links can lack obvious extensions
        if "assets.publishing.service.gov.uk" in href:
            candidates.append(href)

    if candidates:
        return candidates[0]

    return None


def download_document(document):

    output_dir = os.path.join(
        BASE_DIR,
        document["folder"]
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    filename = document["filename"]
    page_url = document["page_url"]

    print("\n" + "=" * 80)
    print(filename)
    print(page_url)

    download_url = find_download_url(
        page_url
    )

    if not download_url:
        print("✗ Could not find downloadable document")
        return False

    print(f"Download URL: {download_url}")

    response = requests.get(
        download_url,
        headers=HEADERS,
        timeout=120,
        allow_redirects=True
    )

    response.raise_for_status()

    # --------------------------------------------------
    # Validate file type
    # --------------------------------------------------

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension == ".pdf":

        if response.content[:4] != b"%PDF":
            print("✗ Expected PDF but content is not PDF")
            return False

    elif extension == ".pptx":

        # PPTX files are ZIP containers
        if response.content[:2] != b"PK":
            print("✗ Expected PPTX but content is not PPTX")
            return False

    # --------------------------------------------------
    # Build filepath BEFORE using it
    # --------------------------------------------------

    filepath = os.path.join(
        output_dir,
        filename
    )

    # --------------------------------------------------
    # Save file
    # --------------------------------------------------

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            response.content
        )

    print(
        f"✓ Saved: {filepath} "
        f"({len(response.content) / 1024:.1f} KB)"
    )

    return True


def main():
    print("=" * 80)
    print("ASKGMP — MHRA Downloader")
    print("=" * 80)

    downloaded = 0
    failed = 0

    for document in DOCUMENTS:
        try:
            if download_document(document):
                downloaded += 1
            else:
                failed += 1

        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Downloaded: {downloaded}")
    print(f"Failed:     {failed}")
    print(f"Saved in:   {os.path.abspath(BASE_DIR)}")


if __name__ == "__main__":
    main()