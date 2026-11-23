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
        "filename": "FDA_Process_Validation_General_Principles_and_Practices.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "process-validation-general-principles-and-practices"
        )
    },
    {
        "filename": "FDA_Data_Integrity_and_Compliance_with_Drug_CGMP_QA.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "data-integrity-and-compliance-drug-cgmp-questions-and-answers"
        )
    },
    {
        "filename": "FDA_Investigating_OOS_Test_Results.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "investigating-out-specification-oos-test-results-"
            "pharmaceutical-production-level-2-revision"
        )
    },
    {
        "filename": "FDA_Quality_Systems_Approach_to_Pharmaceutical_CGMP.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "quality-systems-approach-pharmaceutical-current-"
            "good-manufacturing-practice-regulations"
        )
    },
    {
        "filename": "FDA_Contract_Manufacturing_Quality_Agreements.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "contract-manufacturing-arrangements-drugs-"
            "quality-agreements-guidance-industry"
        )
    },
    {
        "filename": "FDA_Aseptic_Processing_CGMP.pdf",
        "page_url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "sterile-drug-products-produced-aseptic-processing-"
            "current-good-manufacturing-practice"
        )
    }
]


def find_pdf_url(page_url):
    """
    Open an FDA guidance landing page and locate
    its final guidance PDF.
    """

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

        # FDA final guidance links often contain
        # /media/.../download or direct PDF paths
        looks_like_download = (
            "/media/" in absolute_url.lower()
            and "download" in absolute_url.lower()
        )

        looks_like_pdf = (
            ".pdf" in absolute_url.lower()
        )

        is_final_guidance = (
            "download the final guidance" in text
            or "final guidance document" in text
        )

        if is_final_guidance:
            return absolute_url

        if looks_like_download or looks_like_pdf:
            candidates.append(
                absolute_url
            )

    # fallback
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
        print("✗ Could not locate final guidance PDF")
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
    print("ASKGMP — FDA Guidance Downloader")
    print("=" * 80)

    print(
        f"\nDocuments configured: {len(DOCUMENTS)}"
    )

    downloaded = 0
    failed = 0

    for document in DOCUMENTS:

        try:

            if download_document(
                document
            ):
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