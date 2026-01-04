# 📄 RAG Chatbot for Saudi NDMO Regulations

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **National Data Governance Interim Regulations** from Saudi Arabia, based on the official PDF.

## Features
- Loads and processes a PDF file using embeddings (sentence-transformers)
- Retrieves top relevant sections
- Generates answers using Groq API (LLaMA3 model)
- Simple Streamlit interface

## How to Use

Use the Link: 
https://mhnd-chatbot.streamlit.app/

## 🌐 Deploy to Streamlit Cloud
Make sure to upload:
- `app.py`
- `requirements.txt`
- `ndmo_en.pdf`
- `.streamlit/secrets.toml` (or add secret key via Streamlit Cloud UI)
