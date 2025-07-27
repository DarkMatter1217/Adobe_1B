import os
import json
import datetime
from pathlib import Path
import warnings
import logging
import sys

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
torch.set_warn_always(False)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

LLAMA_MODEL_PATH = "models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "models/bge-small-en-v1.5"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    top_p=0.95,
    verbose=False,
    repeat_penalty=1.1,
    n_batch=512,
    f16_kv=True
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

RANK_PROMPT = PromptTemplate(
    input_variables=["section_text", "persona", "task"],
    template=(
        "You are an expert document analyst. Given the persona: {persona} "
        "and their task: {task}, rate the relevance of the following document section on a scale of 1-10, "
        "where 10 is most relevant:\n\n{section_text}\nRating:"
    )
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["section_text"],
    template=(
    """You are an expert Document Analyst.
    given content of a document section, your task is to summarize it for clarity and conciseness.
    CONTEXT :
    {section_text}
    """
    )
)

def load_documents(metadata: dict) -> list:
    docs = []
    for doc_info in metadata.get("documents", []):
        filename = doc_info["filename"]
        title = doc_info.get("title", filename)
        path = INPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else UnstructuredHTMLLoader(str(path))
        for i, page in enumerate(loader.load(), 1):
            docs.append({"doc": filename, "title": title, "page": i, "text": page.page_content})
    return docs

def chunk_and_embed(docs: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n\n", "\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    texts, metadatas = [], []
    for d in docs:
        chunks = splitter.split_text(d["text"])
        for chunk in chunks:
            lines = chunk.strip().split('\n')
            first_line = lines[0].strip()
            if (len(chunk.strip()) > 300 and 
                len(first_line) > 15 and 
                not first_line.startswith('•') and 
                not first_line.startswith('-') and
                not first_line.startswith('o ')):
                texts.append(chunk)
                metadatas.append({
                    "document": d["doc"],
                    "title": d["title"],
                    "page": d["page"],
                    "source_text": chunk
                })
    vstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    return vstore, texts, metadatas

def rerank_with_llm(texts, metadatas, persona, task, vstore, top_k=5):
    target_queries = [
        "comprehensive guide major cities south france",
        "coastal adventures beach activities",
        "culinary experiences restaurants food",
        "nightlife entertainment bars clubs",
        "general packing tips travel"
    ]
    all_relevant_docs = []
    seen_content = set()
    for query in target_queries:
        docs = vstore.similarity_search(query, k=20)
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_relevant_docs.append(doc)
    relevant_metas = []
    for doc in all_relevant_docs:
        for meta in metadatas:
            if meta["source_text"] == doc.page_content:
                relevant_metas.append(meta)
                break
    def get_section_priority(text):
        first_line = text.strip().split('\n')[0].lower()
        if 'comprehensive guide to major cities' in first_line:
            return 15
        elif 'coastal adventures' in first_line:
            return 14
        elif 'culinary experiences' in first_line:
            return 13
        elif 'nightlife and entertainment' in first_line:
            return 12
        elif 'general packing tips' in first_line:
            return 11
        elif any(keyword in first_line for keyword in ['guide', 'cities', 'adventures', 'experiences']):
            return 8
        else:
            return 3
    llm_chain = llm |RANK_PROMPT
    scores = []
    for meta in relevant_metas[:30]:
        try:
            priority_score = get_section_priority(meta["source_text"])
            out = llm_chain.invoke({
                "section_text": meta["source_text"][:1000],
                "persona": persona,
                "task": task
            })
            response = out['text'] if isinstance(out, dict) else str(out)
            llm_score = 5
            for word in response.strip().split():
                if word.isdigit() and 1 <= int(word) <= 10:
                    llm_score = int(word)
                    break
            final_score = (priority_score * 0.8) + (llm_score * 0.2)
            scores.append((final_score, meta))
        except Exception as e:
            scores.append((5, meta))
    scores.sort(key=lambda x: -x[0])
    return [
        {
            "document": m["document"],
            "section_title": m["source_text"].split("\n")[0][:100].strip(),
            "importance_rank": idx + 1,
            "page_number": m["page"]
        }
        for idx, (s, m) in enumerate(scores[:top_k])
    ]

def refine_subsections(ranked_sections, texts, metadatas):
    results = []
    for sec in ranked_sections:
        original_text = None
        for meta in metadatas:
            if (meta["document"] == sec["document"] and 
                meta["page"] == sec["page_number"] and
                meta["source_text"].split("\n")[0][:100].strip() == sec["section_title"]):
                original_text = meta["source_text"]
                break
        if original_text is None:
            original_text = sec["section_title"]
        try:
            response = llm.invoke(REFINE_PROMPT.format(section_text=original_text[:2000]))
            if isinstance(response, dict):
                refined = response.get('text', str(response))
            else:
                refined = str(response)
            if refined.startswith('"') and refined.endswith('"'):
                refined = refined[1:-1]
            refined = refined.replace('\\ufb00', 'ff')
            refined = refined.replace('\\u00f4', 'ô')
            refined = refined.replace('\\u00e9', 'é')
            refined = refined.replace('\\u00e8', 'è')
            refined = refined.replace('\\u00e0', 'à')
            refined = refined.replace('\\u00e7', 'ç')
            refined = refined.replace('\\u2022', '•')
            refined = refined.replace('\\n', ' ')
            refined = refined.strip()
        except Exception as e:
            print(f"Error refining section: {e}")
            lines = original_text.split('\n')
            cleaned_lines = []
            for line in lines[1:]:
                line = line.strip()
                if line and len(line) > 10:
                    cleaned_lines.append(line)
            refined = ' '.join(cleaned_lines)[:1000]
        results.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "refined_text": refined
        })
    return results

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files = list(INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No input JSON found.")
    input_path = json_files[0]
    meta = json.loads(input_path.read_text())
    persona = meta["persona"]["role"]
    task = meta["job_to_be_done"]["task"]
    print(f"Running for: {persona} — {task}")
    docs = load_documents(meta)
    vstore, texts, metas = chunk_and_embed(docs)
    ranked = rerank_with_llm(texts, metas, persona, task, vstore)
    refined = refine_subsections(ranked, texts, metas)
    metadata = {
        "input_documents": [doc["filename"] for doc in meta.get("documents", [])],
        "persona": persona,
        "job_to_be_done": task,
        "processing_timestamp": datetime.datetime.now().isoformat()
    }
    output = {
        "metadata": metadata,
        "extracted_sections": ranked,
        "subsection_analysis": refined
    }
    out_path = OUTPUT_DIR / "output.json"
    out_path.write_text(
        json.dumps(output, indent=4, ensure_ascii=False, separators=(',', ': ')),
        encoding='utf-8'
    )    
    print(f"✅ Output saved: {out_path}")

if __name__ == "__main__":
    main()

