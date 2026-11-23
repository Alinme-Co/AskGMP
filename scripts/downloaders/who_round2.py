import os
import requests

OUTPUT_DIR = "KnowledgeBase/who/production"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

DOCUMENTS = [
    {
        "filename": "TRS1060_Annex2_Nitrosamines_Prevention_and_Control.pdf",
        "url": "https://www.who.int/publications/m/item/trs-1060---annex-2--who-good-practice-considerations-for-the-prevention-and-control-of-nitrosamines-in-pharmaceutical-products"
    },
    {
        "filename": "TRS1033_Annex2_HBELs_in_Cleaning_Validation.pdf",
        "url": "https://www.who.int/publications/m/item/annex-2-trs-1033"
    },
    {
        "filename": "TRS1025_Annex2_GMP_Radiopharmaceutical_Products.pdf",
        "url": "https://www.who.int/publications/m/item/trs1025-annex2"
    },
    {
        "filename": "TRS1025_Annex6_Antimicrobial_Resistance_Manufacturing.pdf",
        "url": "https://www.who.int/publications/m/item/trs-1025-annex-6"
    },
    {
        "filename": "TRS1044_Annex7_GMP_Investigational_Products.pdf",
        "url": "https://www.who.int/publications/m/item/trs1044-annex7"
    }
]


def get_pdf_url(page_url):
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    # Direct PDF
    if response.content[:4] == b"%PDF":
        return page_url

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for a in soup.find_all("a", href=True):
        href = a["href"]

        absolute_url = urljoin(
            page_url,
            href
        )

        if ".pdf" in absolute_url.lower():
            return absolute_url

    return None


def download_document(document):
    page_url = document["url"]
    filename = document["filename"]

    print("\n" + "=" * 80)
    print(filename)
    print(page_url)

    pdf_url = get_pdf_url(page_url)

    if not pdf_url:
        print("✗ No PDF found")
        return False

    print(f"PDF: {pdf_url}")

    response = requests.get(
        pdf_url,
        headers=HEADERS,
        timeout=120,
        allow_redirects=True
    )
    response.raise_for_status()

    if response.content[:4] != b"%PDF":
        print("✗ Downloaded content is not a PDF")
        return False

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(
        f"✓ Saved: {filename} "
        f"({len(response.content) / 1024:.1f} KB)"
    )

    return True


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("ASKGMP — WHO Production Round 2")
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
    print(f"Saved in:   {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()