# RAG Resume Analyzer

A beginner-friendly Streamlit project that compares a resume with a job description using Retrieval-Augmented Generation (RAG).

## What the project does

1. Uploads a resume in PDF, DOCX, or TXT format.
2. Extracts the resume text.
3. Splits the text into smaller chunks.
4. Converts chunks and the job description into embeddings.
5. Retrieves the resume chunks that are most similar to the job description.
6. Sends only those retrieved chunks to Gemini for a grounded analysis.

## Project structure

```text
rag_resume_analyzer/
├── app.py
├── requirements.txt
├── .env.example
├── sample_job_description.txt
├── README.md
└── sample_resumes/
    └── sample_resume.txt
```

## Setup in VS Code

### 1. Open the project folder

```powershell
cd path\to\rag_resume_analyzer
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate it

```powershell
venv\Scripts\activate
```

### 4. Install packages

```powershell
pip install -r requirements.txt
```

### 5. Add the Gemini API key

Copy `.env.example` and rename the copy to `.env`.

```env
GEMINI_API_KEY=your_real_api_key
```

You can also paste the key directly into the app sidebar.

### 6. Run the application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, normally `http://localhost:8501`.

## RAG flow used in this project

```text
Resume file
   ↓
Text extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Similarity search
   ↓
Top relevant chunks
   ↓
Gemini analysis
```

## Important note

The semantic score is only a learning feature. It should not be used as the sole basis for hiring decisions. The prompt instructs the model not to consider protected personal information.
