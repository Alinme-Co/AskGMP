import os
import requests

GUIDELINE = "IMPLEMENTATION"  # Change this to the desired ICH guideline

# Later change to:
# "Q7"
# "Q8"
# "Q9"
# "Q10"
# "Q11"
# "Q12"
# "IMPLEMENTATION"


BASE_OUTPUT_DIR = "KnowledgeBase/ich"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

ICH_DOCUMENTS = {
    "Q7": [
        {
            "filename": "ICH_Q7_Good_Manufacturing_Practice_for_APIs.pdf",
            "url": "https://database.ich.org/sites/default/files/Q7_Guideline.pdf"
        },
        {
            "filename": "ICH_Q7_Questions_and_Answers.pdf",
            "url": "https://database.ich.org/sites/default/files/Q7%20Q%26As%20Questions%20%26%20Answers.pdf"
        }
    ],

    "Q8": [
        {
            "filename": "ICH_Q8_R2_Pharmaceutical_Development.pdf",
            "url": "https://database.ich.org/sites/default/files/Q8_R2_Guideline.pdf"
        }
    ],
    "Q9": [
        {
            "filename": "ICH_Q9_R1_Quality_Risk_Management.pdf",
            "url": "https://database.ich.org/sites/default/files/ICH_Q9%28R1%29_Guideline_Step4_2022_1219.pdf"
        }
    ],
    "Q10": [
        {
            "filename": "ICH_Q10_Pharmaceutical_Quality_System.pdf",
            "url": "https://database.ich.org/sites/default/files/Q10_Guideline.pdf"
        }
    ],
    "Q11": [
        {
            "filename": "ICH_Q11_Development_and_Manufacture_of_Drug_Substances.pdf",
            "url": "https://database.ich.org/sites/default/files/Q11_Guideline.pdf"
        },
        {
            "filename": "ICH_Q11_Questions_and_Answers.pdf",
            "url": "https://database.ich.org/sites/default/files/Q11_Q%26As_Q%26As.pdf"
        }
    ],
    "Q12": [
        {
            "filename": "ICH_Q12_Pharmaceutical_Product_Lifecycle_Management.pdf",
            "url": "https://database.ich.org/sites/default/files/Q12_Guideline_Step4_2019_1119.pdf"
        }
    ],
    "IMPLEMENTATION": [
        {
            "filename": "ICH_Q8_Q9_Q10_Questions_and_Answers_R5.pdf",
            "url": "https://admin.ich.org/sites/default/files/inline-files/ICH_Q9%28R1%29_Annex_1_Q8Q9Q10_QAs%28R5%29_1030.pdf"
        }
    ]
}


def download_file(url, filepath):
    print(f"\nDownloading:")
    print(url)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
        allow_redirects=True
    )

    response.raise_for_status()

    if response.content[:4] != b"%PDF":
        print("WARNING: Content does not look like a PDF.")
        print("Content-Type:", response.headers.get("Content-Type"))

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(
        f"✓ Saved: {filepath} "
        f"({len(response.content) / 1024:.1f} KB)"
    )


def main():

    output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        GUIDELINE.lower()
    )

    os.makedirs(output_dir, exist_ok=True)

    documents = ICH_DOCUMENTS.get(GUIDELINE, [])

    if not documents:
        raise ValueError(
            f"No document URLs configured for {GUIDELINE}"
        )

    print("=" * 80)
    print("ASKGMP — ICH Downloader")
    print("=" * 80)

    print(f"Guideline: {GUIDELINE}")
    print(f"Documents: {len(documents)}")
    print(f"Output:    {output_dir}")

    downloaded = 0
    failed = 0

    for document in documents:

        filepath = os.path.join(
            output_dir,
            document["filename"]
        )

        try:
            download_file(
                document["url"],
                filepath
            )

            downloaded += 1

        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Downloaded: {downloaded}")
    print(f"Failed:     {failed}")


if __name__ == "__main__":
    main()