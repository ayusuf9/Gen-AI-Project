import os
import glob
import time
from pathlib import Path
import streamlit as st
from typing import Optional, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from quanthub.util import llm2

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor."""
        self.embeddings = self._get_embedding_model()
        self.vector_stores: Dict[str, FAISS] = {}
        
    def _get_embedding_model(self) -> OpenAIEmbeddings:
        """Create and return an OpenAI embeddings model instance."""
        openai_api_client = llm2.get_llm_client()
        return OpenAIEmbeddings(
            deployment="text-embedding-3-large",
            model="text-embedding-3-large",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=100
        )
    
    def _get_index_directory(self, pdf_path: str) -> Path:
        """Generate the index directory path for a PDF."""
        filename = Path(pdf_path).stem
        index_dir = Path(f"{filename}_index")
        return index_dir
    
    def process_pdf(self, pdf_path: str, force_rebuild: bool = False) -> Optional[FAISS]:
        """Process a single PDF file and create or load its vector store."""
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found: {pdf_path}")
            return None
            
        index_dir = self._get_index_directory(pdf_path)
        filename = Path(pdf_path).stem
        
        try:
            # Load existing index if available and not forcing rebuild
            if not force_rebuild and index_dir.exists():
                vector_store = FAISS.load_local(str(index_dir), embeddings=self.embeddings)
                st.success(f"Loaded existing index for '{filename}'")
                self.vector_stores[pdf_path] = vector_store
                return vector_store
            
            # Process the PDF
            with st.spinner(f"Processing '{filename}'..."):
                # Load PDF pages
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()  # This loads the PDF page by page
                
                # Create and save the vector store using full pages
                vector_store = FAISS.from_documents(pages, self.embeddings)
                index_dir.mkdir(parents=True, exist_ok=True)
                vector_store.save_local(str(index_dir))
                
                self.vector_stores[pdf_path] = vector_store
                st.success(f"Successfully processed '{filename}' and saved index to {index_dir}")
                return vector_store
                
        except Exception as e:
            st.error(f"Error processing '{filename}': {str(e)}")
            return None
    
    def process_directory(self, directory_path: str, force_rebuild: bool = False):
        """Process all PDFs in a directory."""
        pdf_paths = glob.glob(os.path.join(directory_path, '*.pdf'))
        if not pdf_paths:
            st.warning("No PDF files found in the specified directory.")
            return
        
        total_pdfs = len(pdf_paths)
        st.info(f"Found {total_pdfs} PDF(s) in the directory.")
        progress_bar = st.progress(0)
        
        for idx, pdf_path in enumerate(pdf_paths, 1):
            self.process_pdf(pdf_path, force_rebuild)
            progress = idx / total_pdfs
            progress_bar.progress(progress)
            time.sleep(0.1)  # Small delay for UI responsiveness
            
        st.success("Finished processing all PDFs.")
    
    def get_vector_store(self, pdf_path: str) -> Optional[FAISS]:
        """Retrieve the vector store for a specific PDF."""
        return self.vector_stores.get(pdf_path)

# Usage example
def main():
    st.title("PDF Processing System")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # UI for directory input
    directory_path = st.text_input("Enter PDF directory path:")
    force_rebuild = st.checkbox("Force rebuild indexes")
    
    if st.button("Process PDFs") and directory_path:
        processor.process_directory(directory_path, force_rebuild)

if __name__ == "__main__":
    main()