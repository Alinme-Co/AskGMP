"""
scripts — operational entry points, not application code.

    scripts/downloaders/       fetch source documents into KnowledgeBase/
    build_document_registry.py scan KnowledgeBase/ and refresh the registry CSV
    convert_fda_to_json.py     one-off format conversion
    index_knowledge_base.py    run the indexer in app/knowledge_base/

Run them from the project root with `python -m scripts.<name>`. Anything with
real logic belongs in app/; these stay thin enough that fixing a bug never
means fixing it twice.
"""
