# DocuRAG

DocuRAG is a document question-answering app. Upload a text-based PDF, ask a question, and get an answer based on relevant passages from the document. It can also create a summary.

## Features

- Extracts text and page numbers from PDFs
- Finds relevant passages using semantic search
- Answers questions using the selected passages
- Shows the source passages and page numbers
- Summarizes the PDF

## Requirements

- Python 3.13
- A Gemini API key from Google AI Studio

The Gemini API free tier has usage limits. Availability and limits can change.

## Setup on Windows

Open PowerShell in the project folder and activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1