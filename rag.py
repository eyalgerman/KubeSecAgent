
import json
import os
# from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


def start_rag(tags_tool="checkov", misconfigs_map_path="misconfigs_map.json"):
    """Build a vector store for retrieval-augmented generation.

    Parameters
    ----------
    tags_tool : str, optional
        Key in ``misconfigs_map.json`` that selects which tool's
        misconfiguration descriptions to embed. Defaults to ``"checkov"``.

    The function reads ``misconfigs_map.json`` and writes a Chroma database
    to ``rag_db_<tags_tool>`` containing deduplicated description vectors.
    """
    print("Building RAG database...")

    with open(misconfigs_map_path) as f:
        raw_data = json.load(f)
        tool_data = raw_data.get(tags_tool, {})

    print(f"Loaded {len(tool_data)} entries from JSON")

    seen_descriptions = set()
    docs = []
    for tag, desc in tool_data.items():
        desc = desc.strip()
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            docs.append(Document(page_content=desc, metadata={"tag": tag}))

    print(f"Unique descriptions: {len(docs)}")


    persist_dir = f"rag_db_{tags_tool}"
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_dir,
        collection_name=f"{tags_tool}_misconfigs"
    )
    vectorstore.persist()


    print(f"RAG DB saved to '{persist_dir}' with {len(docs)} vectors")

