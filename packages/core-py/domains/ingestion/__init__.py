"""
ingestion — document ingestion & training pipeline.

Supports:
  - Internet Archive text download (_djvu.txt)
  - PDF text extraction (PyMuPDF)
  - Direct text training via train_transformer_on_text
"""
from .pdf_ingest import (
    download_ia_text,
    extract_pdf_text,
    ingest_book,
    BookIngestResult,
)
