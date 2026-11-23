import os
import requests

BASE_DIR = "KnowledgeBase/pics"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

DOCUMENTS = [
    {
        "folder": "gmp_guide",
        "filename": "PICS_PE009_17_Introduction.pdf",
        "url": "https://picscheme.org/docview/6605",
        "status": "effective"
    },
    {
        "folder": "gmp_guide",
        "filename": "PICS_PE009_17_Annexes.pdf",
        "url": "https://picscheme.org/docview/8881",
        "status": "effective"
    },
    {
        "folder": "data_integrity",
        "filename": "PICS_PI041_1_Data_Integrity.pdf",
        "url": "https://picscheme.org/docview/4234",
        "status": "effective"
    },
    {
        "folder": "qrm",
        "filename": "PICS_PI038_2_QRM_Aide_Memoire.pdf",
        "url": "https://picscheme.org/docview/3823",
        "status": "effective"
    },
    {
        "folder": "change_management",
        "filename": "PICS_PI054_1_Risk_Based_Change_Management.pdf",
        "url": "https://picscheme.org/docview/4294",
        "status": "effective"
    }
]


def download_document(document):

    output_dir = os.path.join(
        BASE_DIR,
        document["folder"]
    )

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(
        output_dir,
        document["filename"]
    )

    print("\n" + "=" * 80)
    print(document["filename"])
    print(f"Status: {document['status']}")
    print(document["url"])

    try:

        response = requests.get(
            document["url"],
            headers=HEADERS,
            timeout=120,
            allow_redirects=True
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # --------------------------------------------------
        # Validate PDF
        # --------------------------------------------------

        if response.content[:4] != b"%PDF":

            print("✗ Content is not a PDF")
            print(f"Content-Type: {content_type}")
            return False

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(
            f"✓ Saved: {filepath} "
            f"({len(response.content) / 1024:.1f} KB)"
        )

        return True

    except Exception as e:

        print(f"✗ FAILED: {e}")
        return False


def main():

    print("=" * 80)
    print("ASKGMP — PIC/S Downloader")
    print("=" * 80)

    print(f"\nDocuments configured: {len(DOCUMENTS)}")

    downloaded = 0
    failed = 0

    for document in DOCUMENTS:

        if download_document(document):
            downloaded += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Downloaded: {downloaded}")
    print(f"Failed:     {failed}")

    print(
        f"Saved in:\n"
        f"{os.path.abspath(BASE_DIR)}"
    )


if __name__ == "__main__":
    main()