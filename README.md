# Chat with PDF using RAG Pipeline

This project provides a Streamlit application that allows users to interact with PDF documents by asking questions. It uses the Retrieval-Augmented Generation (RAG) pipeline to extract context from the PDFs and generate detailed answers using Google's Generative AI models.

## Features
- Upload and process multiple PDF files.
- Extract text from PDFs and split it into manageable chunks.
- Create and manage a FAISS index for efficient document retrieval.
- Handle both natural language and tabular data queries.
- Display tabular data in a structured format using Pandas.
- Real-time conversational AI integration.

## Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Web interface for interaction.
- **PyPDF2**: PDF text extraction.
- **LangChain**: Text splitting and RAG pipeline integration.
- **FAISS**: Vector store for document similarity search.
- **Google Generative AI**: Embeddings and conversational AI models.
- **Pandas**: Tabular data handling.

---

## Getting Started

### Prerequisites
Ensure the following are installed:
- Python (>= 3.8)
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Generative AI SDK](https://developers.generative.ai/)
- [dotenv](https://pypi.org/project/python-dotenv/)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload PDFs:** Use the sidebar to upload one or more PDF files.
2. **Process PDFs:** Click the "Submit & Process" button to extract and index text from the PDFs.
3. **Ask Questions:** Use the text input box on the main page to ask questions about the uploaded PDFs.
4. **View Results:** Receive answers in text or tabular format, depending on the content.

---

## Folder Structure
```
chat-with-pdf/
|— app.py                 # Main Streamlit app file
|— requirements.txt       # Python dependencies
|— .env                   # Environment variables (not included in the repo)
|— faiss_index/           # Folder for FAISS index (created during runtime)
```

---

## Important Notes
- Ensure your `.env` file contains a valid `GOOGLE_API_KEY`.
- FAISS index is stored locally in the `faiss_index` folder and will be overwritten when new PDFs are processed.
- Clear the `faiss_index` folder if encountering errors or starting fresh.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

---

## Acknowledgements
- [LangChain](https://langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://developers.generative.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)

