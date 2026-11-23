import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

OUTPUT_DIR = "KnowledgeBase/fda/guidance"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}


DOCUMENTS = [
    {
        "filename": "FDA_CGMP_for_Phase_1_Investigational_Drugs.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "current-good-manufacturing-practice-phase-1-investigational-drugs"
        )
    },
    {
        "filename": "FDA_PAT_Framework_for_Pharmaceutical_Manufacturing.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "pat-framework-innovative-pharmaceutical-development-"
            "manufacturing-and-quality-assurance"
        )
    },
    {
        "filename": "FDA_ICH_Q13_Continuous_Manufacturing.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "q13-continuous-manufacturing-drug-substances-and-drug-products"
        )
    },
    {
        "filename": "FDA_Comparability_Protocols_Postapproval_CMC_Changes.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "comparability-protocols-postapproval-changes-chemistry-"
            "manufacturing-and-controls-information-nda"
        )
    }
]


def find_pdf_url(page_url):

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    candidates = []

    for a in soup.find_all("a", href=True):

        text = a.get_text(
            " ",
            strip=True
        ).lower()

        href = a["href"]

        absolute_url = urljoin(
            page_url,
            href
        )

        # Best match
        if "download the final guidance" in text:
            return absolute_url

        # Fallback FDA media/download links
        if (
            "/media/" in absolute_url.lower()
            and "download" in absolute_url.lower()
        ):
            candidates.append(
                absolute_url
            )

        elif ".pdf" in absolute_url.lower():
            candidates.append(
                absolute_url
            )

    if candidates:
        return candidates[0]

    return None


def download_document(document):

    filename = document["filename"]
    page_url = document["page_url"]

    print("\n" + "=" * 80)
    print(filename)
    print(page_url)

    pdf_url = find_pdf_url(
        page_url
    )

    if not pdf_url:

        print(
            "✗ Could not locate final guidance PDF"
        )

        return False

    print(
        f"PDF: {pdf_url}"
    )

    response = requests.get(
        pdf_url,
        headers=HEADERS,
        timeout=120,
        allow_redirects=True
    )

    response.raise_for_status()

    if response.content[:4] != b"%PDF":

        print(
            "✗ Downloaded content is not a PDF"
        )

        print(
            "Content-Type:",
            response.headers.get(
                "Content-Type"
            )
        )

        return False

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(
        filepath,
        "wb"
    ) as f:

        f.write(
            response.content
        )

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
    print("ASKGMP — FDA Guidance Round 2")
    print("=" * 80)

    print(
        f"\nDocuments configured: {len(DOCUMENTS)}"
    )

    downloaded = 0
    failed = 0

    for document in DOCUMENTS:

        try:

            if download_document(document):
                downloaded += 1

            else:
                failed += 1

        except Exception as e:

            print(
                f"✗ FAILED: {e}"
            )

            failed += 1

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(
        f"Downloaded: {downloaded}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Saved in:\n"
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()