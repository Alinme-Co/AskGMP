import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

PAGE_URL = (
    "https://www.who.int/teams/health-product-and-policy-standards/"
    "standards-and-specifications/norms-and-standards-for-pharmaceuticals/"
    "guidelines/production"
)

OUTPUT_DIR = "KnowledgeBase/who/production"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

# ============================================================
# ONLY DOWNLOAD DOCUMENTS MATCHING THESE TOPICS
# ============================================================

KEYWORDS = [
    "main principles",
    "sterile pharmaceutical products",
    "data integrity",
    "guidelines on validation",
    "technology transfer",
    "water for pharmaceutical use",
    "quality risk management",
    "active pharmaceutical ingredients",
    "biological products",
    "medicinal gases",
    "excipients",
    "hold-time",
    "heating, ventilation and air-conditioning",
    "continuous manufacturing",
]


def clean_filename(text, max_length=120):
    """
    Create a Windows-safe filename and keep it reasonably short.
    """

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # WHO page often repeats the title twice.
    # If the second half is identical to the first, keep only one half.
    words = text.split()

    if len(words) >= 8:
        half = len(words) // 2

        first = " ".join(words[:half])
        second = " ".join(words[half:])

        if first.lower() == second.lower():
            text = first

    # Remove characters invalid in Windows filenames
    text = re.sub(r'[<>:"/\\|?*]', "_", text)

    # Replace spaces with underscores
    text = re.sub(r"\s+", "_", text)

    # Collapse repeated underscores
    text = re.sub(r"_+", "_", text)

    text = text.strip("_. ")

    # Limit filename length
    if len(text) > max_length:
        text = text[:max_length].rstrip("_")

    return text

def is_relevant(title):
    title_lower = title.lower()

    return any(
        keyword.lower() in title_lower
        for keyword in KEYWORDS
    )

def create_filename(title, index):
    """
    Produce a short, meaningful WHO filename.
    Example:
    01_TRS1067_Annex2_Continuous_Manufacturing.pdf
    """

    # Find TRS number
    trs_match = re.search(
        r"TRS\s*(\d+)",
        title,
        re.IGNORECASE
    )

    trs = (
        f"TRS{trs_match.group(1)}"
        if trs_match
        else "WHO"
    )

    # Find Annex number
    annex_match = re.search(
        r"Annex\s*(\d+)",
        title,
        re.IGNORECASE
    )

    annex = (
        f"Annex{annex_match.group(1)}"
        if annex_match
        else ""
    )

    # Define short topic names
    title_lower = title.lower()

    if "continuous manufacturing" in title_lower:
        topic = "Continuous_Manufacturing"

    elif "excipients" in title_lower:
        topic = "GMP_Excipients"

    elif "medicinal gases" in title_lower:
        topic = "GMP_Medicinal_Gases"

    elif "sterile pharmaceutical" in title_lower:
        topic = "GMP_Sterile_Products"

    elif "technology transfer" in title_lower:
        topic = "Technology_Transfer"

    elif "water for pharmaceutical" in title_lower:
        topic = "Water_for_Pharmaceutical_Use"

    elif "data integrity" in title_lower:
        topic = "Data_Integrity"

    elif "heating" in title_lower and "ventilation" in title_lower:
        topic = "HVAC"

    elif "validation" in title_lower:
        topic = "Validation"

    elif "biological products" in title_lower:
        topic = "GMP_Biological_Products"

    elif "hold-time" in title_lower:
        topic = "Hold_Time_Studies"

    elif "main principles" in title_lower:
        topic = "GMP_Main_Principles"

    elif "quality risk management" in title_lower:
        topic = "Quality_Risk_Management"

    elif "active pharmaceutical ingredients" in title_lower:
        topic = "GMP_APIs"

    else:
        topic = clean_filename(title, max_length=60)

    parts = [
        f"{index:02d}",
        trs
    ]

    if annex:
        parts.append(annex)

    parts.append(topic)

    return "_".join(parts) + ".pdf"


def find_pdf_on_detail_page(url):
    """
    WHO catalogue links often point to a document landing page,
    not directly to a PDF.

    Open that page and locate the first PDF link.
    """

    r = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )
    r.raise_for_status()

    # Direct PDF
    if r.content[:4] == b"%PDF":
        return url

    soup = BeautifulSoup(
        r.text,
        "html.parser"
    )

    pdf_links = []

    for a in soup.find_all("a", href=True):

        href = a["href"]

        absolute_url = urljoin(
            url,
            href
        )

        if ".pdf" in absolute_url.lower():
            pdf_links.append(absolute_url)

    if pdf_links:
        return pdf_links[0]

    return None


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("ASKGMP — WHO GMP Downloader")
    print("=" * 80)

    print(f"\nReading:\n{PAGE_URL}\n")

    response = requests.get(
        PAGE_URL,
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    candidates = []

    # --------------------------------------------------------
    # Collect relevant WHO document links
    # --------------------------------------------------------

    for a in soup.find_all("a", href=True):

        title = a.get_text(
            " ",
            strip=True
        )

        href = a["href"]

        if not title:
            continue

        if not is_relevant(title):
            continue

        url = urljoin(
            PAGE_URL,
            href
        )

        candidates.append({
            "title": title,
            "url": url
        })

    # Deduplicate
    unique_candidates = []
    seen = set()

    for item in candidates:

        if item["url"] not in seen:

            seen.add(item["url"])
            unique_candidates.append(item)

    print(
        f"Found {len(unique_candidates)} relevant WHO documents.\n"
    )

    for i, item in enumerate(
        unique_candidates,
        start=1
    ):
        print(
            f"{i:02d}. {item['title']}"
        )

    # --------------------------------------------------------
    # Download PDFs
    # --------------------------------------------------------

    downloaded = 0
    failed = 0

    for index, item in enumerate(
        unique_candidates,
        start=1
    ):

        print("\n" + "=" * 80)
        print(item["title"])
        print(item["url"])

        try:

            pdf_url = find_pdf_on_detail_page(
                item["url"]
            )

            if not pdf_url:

                print("✗ Could not find PDF")
                failed += 1
                continue

            print(f"PDF: {pdf_url}")

            r = requests.get(
                pdf_url,
                headers=HEADERS,
                timeout=120,
                allow_redirects=True
            )

            r.raise_for_status()

            if r.content[:4] != b"%PDF":

                print(
                    "✗ Downloaded content is not a PDF"
                )

                failed += 1
                continue

            filename = create_filename(
                item["title"],
                index
            )

            filepath = os.path.join(
                OUTPUT_DIR,
                filename
            )

            with open(
                filepath,
                "wb"
            ) as f:

                f.write(
                    r.content
                )

            print(
                f"✓ Saved: {filename} "
                f"({len(r.content) / 1024:.1f} KB)"
            )

            downloaded += 1

        except Exception as e:

            print(
                f"✗ FAILED: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

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