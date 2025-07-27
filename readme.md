# 🐳 Run Adobe Hackathon 1B Solution using Docker

## 📆 About the Project

This Dockerized project solves the **Adobe Hackathon Round 1B** problem:

> Persona-driven document analysis — extract & summarize the most relevant sections based on a user persona and task.

Technologies used:

* 🦩 TinyLLaMA (via `llama-cpp-python`)
* 🔗 LangChain
* 🤗 HuggingFace Embeddings
* 🧠 FAISS Vector Store
  Runs fully **offline** using Docker.

---

## 📁 Folder Structure

```
.
├── dockerfile                 # Docker instructions
├── .dockerignore              # Ignore output/models etc. during docker builds
├── solution.py                # Main script
├── requirements.txt           # All Python dependencies
├── input/
│   ├── input.json             # Contains persona, task, and document info
│   └── <document>.pdf/html    # Supporting documents
├── models/
│   ├── tinyllama/             # Quantized TinyLLaMA model
│   └── bge-small-en-v1.5/     # Embedding model folder
├── output/                    # Auto-created for results
```

---

## 🛠️ Setup and Run with Docker

### ✅ 1. Prepare Required Files

* `solution.py`, `dockerfile`, `requirements.txt`
* `input/input.json`: Your input metadata file (see format below)
* All referenced PDFs/HTMLs inside `input/`
* Required models inside `models/`

Example folder:

```
models/
├── tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
└── bge-small-en-v1.5/*

input/
├── input.json
└── FranceTravelGuide.pdf
```

---

### 🐳 2. Build Docker Image

```bash
docker build -t adobe-1b-solution .
```

Make sure your `dockerfile` includes:

```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "solution.py"]
```

---

### ▶️ 3. Run Container

```bash
# Windows (PowerShell)
docker run --rm -v ${PWD}:/app adobe-1b-solution

# macOS/Linux
docker run --rm -v $PWD:/app adobe-1b-solution
```

---

### 📄 4. Output Location

Final result will be saved at:

```
output/output.json
```

It contains:

* Ranked document sections relevant to the persona
* Refined summaries for each section

---

## 🧠 Sample `input/input.json`

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner"
  },
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  },
  "documents": [
    {
      "filename": "FranceTravelGuide.pdf",
      "title": "France Travel"
    }
  ]
}
```

---

## ✅ Recap Commands

```bash
# Build
docker build -t adobe-1b-solution .

# Run (mount current directory to container)
docker run --rm -v ${PWD}:/app adobe-1b-solution
```

---

## 📦 Python Dependencies

Make sure `requirements.txt` includes:

```txt
langchain-community
langchain-huggingface
llama-cpp-python
transformers
sentence-transformers
torch
faiss-cpu
PyPDF2
pypdf
unstructured
huggingface-hub
numpy
pathlib
```

---

## 📝 Notes

* Models must be downloaded manually and stored in `models/`
* Uses `TinyLLaMA` (GGUF format) for LLM inference
* Designed for **offline CPU** execution
* All results saved in structured `JSON` format
* ⚠️ **Model files are pushed using Git LFS**

  * Clone the repo like this:

    ```bash
    git lfs install
    git clone https://github.com/DarkMatter1217/Adobe_1B.git
    ```
