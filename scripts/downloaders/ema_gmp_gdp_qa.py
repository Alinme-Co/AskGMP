import os
import re
import json
import requests
from bs4 import BeautifulSoup

PAGE_URL = (
    "https://www.ema.europa.eu/en/human-regulatory-overview/"
    "research-development/compliance-research-development/"
    "good-manufacturing-practice/"
    "guidance-good-manufacturing-practice-good-distribution-practice-questions-answers"
)

OUTPUT_DIR = "KnowledgeBase/ema/gmp_gdp_qa"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("ASKGMP — EMA GMP/GDP Q&A Downloader")
    print("=" * 80)

    print(f"\nReading:\n{PAGE_URL}\n")

    response = requests.get(
        PAGE_URL,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # ---------------------------------------------------------
    # Find the main page content
    # ---------------------------------------------------------

    main = soup.find("main")

    if not main:
        raise RuntimeError("Could not find EMA main content.")

    records = []

    current_section = None
    current_question = None
    current_answer_parts = []

    # ---------------------------------------------------------
    # Parse headings and paragraphs in document order
    # ---------------------------------------------------------

    elements = main.find_all(
        ["h2", "h3", "p", "ul", "ol"],
        recursive=True
    )

    def save_current_question():
        nonlocal current_question, current_answer_parts

        if current_question:

            answer = clean_text(
                " ".join(current_answer_parts)
            )

            records.append({
                "section": current_section,
                "question": current_question,
                "answer": answer,
                "source": PAGE_URL
            })

        current_question = None
        current_answer_parts = []

    for element in elements:

        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        # ---------------------------------------------
        # Section heading
        # ---------------------------------------------

        if element.name == "h2":

            save_current_question()

            current_section = text

        # ---------------------------------------------
        # Individual question
        # ---------------------------------------------

        elif element.name == "h3":

            save_current_question()

            current_question = text

        # ---------------------------------------------
        # Answer content
        # ---------------------------------------------

        elif element.name in ["p", "ul", "ol"]:

            if current_question:
                current_answer_parts.append(text)

    save_current_question()

    # ---------------------------------------------------------
    # Filter obvious navigation/non-Q&A records
    # ---------------------------------------------------------

    records = [
        r for r in records
        if r["question"]
        and r["answer"]
        and r["section"]
    ]

    print(f"Found {len(records)} Q&A records.")

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "EMA_GMP_GDP_QA.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"✓ Saved JSON: {json_path}")

    # ---------------------------------------------------------
    # Save Markdown
    # ---------------------------------------------------------

    md_path = os.path.join(
        OUTPUT_DIR,
        "EMA_GMP_GDP_QA.md"
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("# EMA GMP/GDP Questions and Answers\n\n")
        f.write(f"Source: {PAGE_URL}\n\n")

        previous_section = None

        for record in records:

            if record["section"] != previous_section:

                f.write(
                    f"\n## {record['section']}\n\n"
                )

                previous_section = record["section"]

            f.write(
                f"### {record['question']}\n\n"
            )

            f.write(
                f"{record['answer']}\n\n"
            )

    print(f"✓ Saved Markdown: {md_path}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(f"Q&A records: {len(records)}")
    print(f"Saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()