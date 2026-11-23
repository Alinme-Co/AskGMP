import os
import re
import json
from bs4 import BeautifulSoup

INPUT_DIR = "KnowledgeBase/fda/regulations"
OUTPUT_DIR = "KnowledgeBase/fda/regulations/structured"

os.makedirs(OUTPUT_DIR, exist_ok=True)


FILES = {
    "210": {
        "html": "21_CFR_Part_210_cGMP_General.html",
        "title": "Current Good Manufacturing Practice in Manufacturing, Processing, Packing, or Holding of Drugs; General"
    },
    "211": {
        "html": "21_CFR_Part_211_cGMP_Finished_Pharmaceuticals.html",
        "title": "Current Good Manufacturing Practice for Finished Pharmaceuticals"
    }
}


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_subpart(text):
    match = re.search(r"Subpart\s+([A-Z])", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_section_number(text):
    """
    Extract section numbers such as:
    § 211.22
    211.192
    """
    match = re.search(
        r"(?:§\s*)?(\d{3}\.\d+)",
        text
    )

    return match.group(1) if match else None


def parse_ecfr_file(filepath, part_number, part_title):

    print(f"\nParsing: {filepath}")

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")

    if not main:
        main = soup

    records = []

    current_subpart = None

    # -----------------------------------------------------
    # Find useful structural elements
    # -----------------------------------------------------

    elements = main.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "div",
            "p"
        ],
        recursive=True
    )

    current_section = None
    current_heading = None
    current_text_parts = []

    def save_current():

        nonlocal current_section
        nonlocal current_heading
        nonlocal current_text_parts

        if not current_section:
            return

        text = clean_text(
            " ".join(current_text_parts)
        )

        if not text:
            return

        records.append({
            "authority": "eCFR",
            "jurisdiction": "United States",
            "regulation": "21 CFR",
            "title_number": "21",
            "part": part_number,
            "part_title": part_title,
            "subpart": current_subpart,
            "section": current_section,
            "heading": current_heading,
            "text": text,
            "source_url": (
                f"https://www.ecfr.gov/current/title-21/"
                f"chapter-I/subchapter-C/part-{part_number}"
                f"/section-{current_section}"
            )
        })

        current_section = None
        current_heading = None
        current_text_parts = []

    # -----------------------------------------------------
    # Parse
    # -----------------------------------------------------

    for element in elements:

        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        # ----------------------------------------------
        # Detect Subpart headings
        # ----------------------------------------------

        if text.lower().startswith("subpart "):

            detected = extract_subpart(text)

            if detected:
                current_subpart = detected

        # ----------------------------------------------
        # Detect section headings
        # ----------------------------------------------

        section_number = extract_section_number(text)

        if section_number:

            # Avoid treating ordinary paragraphs containing
            # references to other CFR sections as headings.
            if element.name in ["h2", "h3", "h4", "h5"]:

                save_current()

                current_section = section_number

                # Remove section number from heading
                heading = re.sub(
                    rf"^.*?{re.escape(section_number)}\s*",
                    "",
                    text
                )

                heading = heading.lstrip("—-–: ")

                current_heading = (
                    heading
                    if heading
                    else f"Section {section_number}"
                )

                continue

        # ----------------------------------------------
        # Collect text belonging to current section
        # ----------------------------------------------

        if current_section:

            # Avoid duplicate text from nested containers
            if element.name == "p":

                current_text_parts.append(text)

    save_current()

    return records


def save_json(records, part_number):

    filepath = os.path.join(
        OUTPUT_DIR,
        f"21_CFR_Part_{part_number}.json"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"✓ JSON: {filepath}")


def save_markdown(records, part_number, part_title):

    filepath = os.path.join(
        OUTPUT_DIR,
        f"21_CFR_Part_{part_number}.md"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"# 21 CFR Part {part_number}\n\n"
        )

        f.write(
            f"{part_title}\n\n"
        )

        previous_subpart = None

        for record in records:

            if record["subpart"] != previous_subpart:

                if record["subpart"]:

                    f.write(
                        f"\n## Subpart "
                        f"{record['subpart']}\n\n"
                    )

                previous_subpart = record["subpart"]

            f.write(
                f"### § {record['section']} "
                f"{record['heading']}\n\n"
            )

            f.write(
                f"{record['text']}\n\n"
            )

            f.write(
                f"Source: {record['source_url']}\n\n"
            )

    print(f"✓ Markdown: {filepath}")


def main():

    print("=" * 80)
    print("ASKGMP — eCFR Structured Parser")
    print("=" * 80)

    total_records = 0

    for part_number, config in FILES.items():

        filepath = os.path.join(
            INPUT_DIR,
            config["html"]
        )

        if not os.path.exists(filepath):

            print(
                f"\n✗ Missing file: {filepath}"
            )
            continue

        records = parse_ecfr_file(
            filepath,
            part_number,
            config["title"]
        )

        print(
            f"Found {len(records)} regulatory sections "
            f"in Part {part_number}"
        )

        save_json(
            records,
            part_number
        )

        save_markdown(
            records,
            part_number,
            config["title"]
        )

        total_records += len(records)

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(
        f"Total structured sections: {total_records}"
    )

    print(
        f"Saved in:\n"
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()