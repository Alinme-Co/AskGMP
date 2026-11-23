import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

PAGE_URL = "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-4_en"

# ============================================================
# CHANGE ONLY THIS VALUE FOR EACH STEP
# ============================================================

SECTION = "glossary"

# Allowed values:
# "part_1"
# "part_2"
# "part_3"
# "annexes"
# "glossary"
# "part_4"

# ============================================================

BASE_OUTPUT_DIR = "KnowledgeBase/eudralex_volume_4"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}


SECTION_CONFIG = {
    "part_1": {
        "heading_contains": [
            "Part I",
            "Basic Requirements for Medicinal Products"
        ],
        "folder": "part_1",
    },

    "part_2": {
        "heading_contains": [
            "Part II",
            "Basic Requirements for Active Substances"
        ],
        "folder": "part_2",
    },

    "part_3": {
        "heading_contains": [
            "Part III",
            "GMP related documents"
        ],
        "folder": "part_3",
    },

    "annexes": {
        "heading_contains": [
            "Annexes"
        ],
        "folder": "annexes",
    },

    "glossary": {
        "heading_contains": [
            "Glossary"
        ],
        "folder": "glossary",
    },

    "part_4": {
        "heading_contains": [
            "Part IV",
            "Advanced Therapy Medicinal Products"
        ],
        "folder": "part_4",
    }
}


def clean_filename(text):
    """
    Create a safe Windows/Linux filename.
    """
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_. ")


def find_section_heading(soup, config):
    """
    Find the heading corresponding to the requested section.
    """
    for heading in soup.find_all(["h2", "h3", "h4"]):

        text = heading.get_text(" ", strip=True)

        matches = all(
            keyword.lower() in text.lower()
            for keyword in config["heading_contains"]
        )

        if matches:
            return heading

    return None


def collect_section_links(section_heading):
    """
    Collect links until the next heading of the same/higher level.
    """

    links = []

    heading_level = int(section_heading.name[1])

    current = section_heading.find_next()

    while current:

        # ----------------------------------------------------
        # Stop at next section heading
        # ----------------------------------------------------

        if current.name in ["h2", "h3", "h4"]:

            current_level = int(current.name[1])

            if current_level <= heading_level:
                break

        # ----------------------------------------------------
        # Collect links
        # ----------------------------------------------------

        if current.name == "a":

            href = current.get("href")
            title = current.get_text(" ", strip=True)

            if href and title:

                absolute_url = urljoin(PAGE_URL, href)

                # Ignore anchors/javascript
                if absolute_url.startswith("http"):
                    links.append({
                        "title": title,
                        "url": absolute_url
                    })

        current = current.find_next()

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    unique_links = []
    seen = set()

    for item in links:

        if item["url"] not in seen:

            seen.add(item["url"])
            unique_links.append(item)

    return unique_links


def determine_extension(response, url):
    """
    Determine whether downloaded content is PDF or HTML.
    """

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if response.content[:4] == b"%PDF":
        return ".pdf"

    if "application/pdf" in content_type:
        return ".pdf"

    url_path = urlparse(url).path.lower()

    if url_path.endswith(".pdf"):
        return ".pdf"

    if "html" in content_type:
        return ".html"

    return ".bin"


def download_document(item, output_dir, index):

    title = item["title"]
    url = item["url"]

    print("\n" + "=" * 80)
    print(f"[{index}] {title}")
    print(url)

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
            allow_redirects=True
        )

        response.raise_for_status()

        extension = determine_extension(
            response,
            response.url
        )

        filename = (
            f"{index:02d}_"
            f"{clean_filename(title)}"
            f"{extension}"
        )

        filepath = os.path.join(
            output_dir,
            filename
        )

        with open(filepath, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024

        print(
            f"✓ Saved: {filename} "
            f"({size_kb:.1f} KB)"
        )

        return True

    except Exception as e:

        print(f"✗ FAILED: {e}")

        return False


def main():

    # --------------------------------------------------------
    # Validate section
    # --------------------------------------------------------

    if SECTION not in SECTION_CONFIG:

        raise ValueError(
            f"Unknown SECTION '{SECTION}'. "
            f"Allowed values: "
            f"{list(SECTION_CONFIG.keys())}"
        )

    config = SECTION_CONFIG[SECTION]

    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        config["folder"]
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("=" * 80)
    print("ASKGMP — EudraLex Downloader")
    print("=" * 80)

    print(f"\nSection: {SECTION}")
    print(f"Output:  {output_dir}")

    # --------------------------------------------------------
    # Download page
    # --------------------------------------------------------

    print(
        f"\nReading EudraLex page:\n"
        f"{PAGE_URL}\n"
    )

    response = requests.get(
        PAGE_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # --------------------------------------------------------
    # Find section
    # --------------------------------------------------------

    section_heading = find_section_heading(
        soup,
        config
    )

    if not section_heading:

        raise RuntimeError(
            f"Could not find section: {SECTION}"
        )

    print(
        "Found section:"
    )

    print(
        section_heading.get_text(
            " ",
            strip=True
        )
    )

    # --------------------------------------------------------
    # Collect links
    # --------------------------------------------------------

    links = collect_section_links(
        section_heading
    )

    print(
        f"\nFound {len(links)} links:"
    )

    for i, item in enumerate(
        links,
        start=1
    ):

        print(
            f"{i:02d}. {item['title']}"
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    downloaded = 0
    failed = 0

    for index, item in enumerate(
        links,
        start=1
    ):

        success = download_document(
            item,
            output_dir,
            index
        )

        if success:
            downloaded += 1
        else:
            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Section:     {SECTION}")
    print(f"Downloaded:  {downloaded}")
    print(f"Failed:      {failed}")

    print(
        f"Saved in:\n"
        f"{os.path.abspath(output_dir)}"
    )


if __name__ == "__main__":
    main()