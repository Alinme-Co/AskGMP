import os
import requests

OUTPUT_DIR = "KnowledgeBase/fda/regulations"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

# eCFR provides PDF endpoints for individual CFR parts.
DOCUMENTS = [
    {
        "filename": "21_CFR_Part_210_cGMP_General.pdf",
        "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-210"
    },
    {
        "filename": "21_CFR_Part_211_cGMP_Finished_Pharmaceuticals.pdf",
        "url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211"
    }
]


def download_html(document):

    print("\n" + "=" * 80)
    print(document["filename"])
    print(document["url"])

    response = requests.get(
        document["url"],
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    # eCFR pages are HTML.
    # Save the authoritative page locally first.
    filename = document["filename"].replace(
        ".pdf",
        ".html"
    )

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            response.text
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
    print("ASKGMP — FDA / eCFR Regulations Downloader")
    print("=" * 80)

    downloaded = 0
    failed = 0

    for document in DOCUMENTS:

        try:

            if download_html(document):
                downloaded += 1

        except Exception as e:

            print(
                f"✗ FAILED: {e}"
            )

            failed += 1

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Downloaded: {downloaded}")
    print(f"Failed:     {failed}")

    print(
        f"Saved in:\n"
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()