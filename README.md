# Closed-Domain Technical Documentation QA Bot

A lightweight extractive question-answering system for answering student questions **only from a university technical manual**.

## Architecture

```text
Student Question
      |
      v
TF-IDF Retriever
      |
      v
Top-3 Manual Sections
      |
      v
RoBERTa Extractive QA
      |
      v
Confidence / Abstention
      |
      +----> Answer + Source
      |
      +----> Abstain if unsupported
```

The default reader is `deepset/roberta-base-squad2`. The application loads it lazily from Hugging Face or from `models/qa_model/` if a locally trained model exists. If it cannot be downloaded, the demo uses a small deterministic extractive fallback so the interface and retrieval pipeline can still be tested.

## Setup — Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run the web app

```powershell
streamlit run app.py
```

The first model-backed run may download the RoBERTa model from Hugging Face. No model weights or datasets are included in the ZIP.

## Run the terminal demo without downloading a model

```powershell
python main.py --demo --fallback
```

## Train on SQuAD 2.0

Use a GPU machine such as Google Colab:

```bash
pip install -r requirements.txt
python train.py
```

The trained model is saved under `models/qa_model/`. Keep model weights outside GitHub/source ZIPs because they are large.

## Evaluate

```powershell
python evaluate.py
```

Results are written to `results/evaluation.json`.

## Dataset

The project uses SQuAD 2.0 through Hugging Face Datasets for optional fine-tuning. A compact custom `data/university_manual.json` provides the closed-domain demonstration corpus. SQuAD and model weights are downloaded separately and are intentionally excluded from the project ZIP.

## Closed-domain behavior

The retriever first checks whether a manual section is sufficiently relevant. The reader then extracts an answer span from that manual text. If the evidence is insufficient, the bot abstains instead of inventing an answer.

## Example questions

- What is the minimum attendance required?
- How many books can a student borrow?
- When is the computer lab open?
- How many devices can connect to campus Wi-Fi?
- How long can a student borrow an Arduino kit?

## Limitations

TF-IDF is intentionally simple and works best when questions share terminology with the manual. For a larger manual, semantic retrieval can be added later. The demo fallback is not a substitute for the RoBERTa reader; it exists to make local testing possible without a model download.
