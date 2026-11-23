import os
import re
import json
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = "KnowledgeBase/fda/cgmp_qa"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

TOPIC_PAGES = [
    {
        "topic": "General Provisions",
        "url": (
            "https://www.fda.gov/regulatory-information/"
            "search-fda-guidance-documents/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "questions-and-answers-current-good"
        )
    },
    {
        "topic": "Buildings and Facilities",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-buildings-and-facilities"
        )
    },
    {
        "topic": "Equipment",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-equipment"
        )
    },
    {
        "topic": "Control of Components and Drug Product Containers and Closures",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-control-components-and-drug"
        )
    },
    {
        "topic": "Production and Process Controls",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "regulations-production-and-process"
        )
    },
    {
        "topic": "Holding and Distribution",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-holding-and-distribution"
        )
    },
    {
        "topic": "Laboratory Controls",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-laboratory-controls"
        )
    },
    {
        "topic": "Records and Reports",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-records-and-reports"
        )
    },
    {
        "topic": "Returned and Salvaged Drug Products",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-returned-and-salvaged-drug"
        )
    },
    {
    "topic": "Records and Reports",
        "url": (
            "https://www.fda.gov/drugs/guidances-drugs/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-records-and-reports"
        )
    },
    {
        "topic": "Returned and Salvaged Drug Products",
        "url": (
            "https://www.fda.gov/drugs/pharmaceutical-quality-resources/"
            "questions-and-answers-current-good-manufacturing-practice-"
            "requirements-returned-and-salvaged-drug"
        )
    }
]


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def looks_like_question(text):
    """
    FDA questions usually begin with a number,
    but their HTML/text does not always end cleanly with '?'.
    """
    return bool(
        re.match(
            r"^\s*\d+\.\s+",
            text.strip()
        )
    )


def strip_number(text):
    return re.sub(
        r"^\d+\.\s*",
        "",
        text
    ).strip()


def parse_topic(topic, url):

    print("\n" + "=" * 80)
    print(topic)
    print(url)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    main = soup.find("main") or soup

    records = []

    # Get document content in order.
    elements = main.find_all(
        ["p", "li", "h2", "h3", "h4", "strong"],
        recursive=True
    )

    current_question = None
    current_answer = []

    def save_current():
        nonlocal current_question, current_answer

        if not current_question:
            return

        answer = clean_text(
            " ".join(current_answer)
        )

        if answer:
            records.append({
                "authority": "FDA",
                "jurisdiction": "United States",
                "source_type": "CGMP Q&A",
                "topic": topic,
                "question": current_question,
                "answer": answer,
                "source_url": url
            })

        current_question = None
        current_answer = []

    for element in elements:

        # Avoid nested duplicates.
        if element.find(["p", "li", "h2", "h3", "h4"]):
            continue

        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        if looks_like_question(text):

            save_current()

            current_question = strip_number(
                text
            )

            continue

        if current_question:

            lower = text.lower()

            # Stop/ignore page boilerplate.
            if any(
                x in lower
                for x in [
                    "contains nonbinding recommendations",
                    "contact for further information",
                    "submit comments",
                    "content current as of",
                    "related information",
                    "back to top"
                ]
            ):
                continue

            current_answer.append(text)

    save_current()

    print(
        f"Found {len(records)} Q&A records"
    )

    return records


def save_json(records):

    filepath = os.path.join(
        OUTPUT_DIR,
        "FDA_cGMP_QA.json"
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


def save_markdown(records):

    filepath = os.path.join(
        OUTPUT_DIR,
        "FDA_cGMP_QA.md"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# FDA Questions and Answers on CGMP Requirements\n\n"
        )

        previous_topic = None

        for record in records:

            if record["topic"] != previous_topic:

                f.write(
                    f"\n## {record['topic']}\n\n"
                )

                previous_topic = record["topic"]

            f.write(
                f"### {record['question']}\n\n"
            )

            f.write(
                f"{record['answer']}\n\n"
            )

            f.write(
                f"Source: {record['source_url']}\n\n"
            )

    print(f"✓ Markdown: {filepath}")


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("ASKGMP — FDA CGMP Q&A Extractor")
    print("=" * 80)

    all_records = []

    for page in TOPIC_PAGES:

        try:

            records = parse_topic(
                page["topic"],
                page["url"]
            )

            all_records.extend(
                records
            )

        except Exception as e:

            print(
                f"✗ FAILED: {page['topic']} — {e}"
            )

    # Remove duplicates
    unique_records = []
    seen = set()

    for record in all_records:

        key = (
            record["topic"].lower(),
            record["question"].lower()
        )

        if key not in seen:

            seen.add(key)
            unique_records.append(
                record
            )

    print("\n" + "=" * 80)
    print(
        f"Unique Q&A records: {len(unique_records)}"
    )
    print("=" * 80)

    save_json(
        unique_records
    )

    save_markdown(
        unique_records
    )

    print("\nFINISHED")

    print(
        f"Saved in:\n"
        f"{os.path.abspath(OUTPUT_DIR)}"
    )


if __name__ == "__main__":
    main()