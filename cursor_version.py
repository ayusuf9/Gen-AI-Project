import os
from typing import Optional, List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class PDFTableExtractor:
    def __init__(self, openai_api_key: str):
        """Initialize the PDF Table Extractor with OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
    def load_pdf(self, pdf_path: str) -> List[str]:
        """Load PDF and split into pages."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return pages
    
    def find_table_page(self, pages: List[str], table_description: str) -> Optional[tuple]:
        """Find the page containing the target table using semantic search."""
        # Create text chunks for vector search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Create vector store from splits
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Search for the page containing the table
        relevant_chunks = vectorstore.similarity_search(
            f"Find a table that contains {table_description}",
            k=3
        )
        
        if not relevant_chunks:
            return None
            
        # Find the original page number
        for page_num, page in enumerate(pages):
            if relevant_chunks[0].page_content in page.page_content:
                return (page.page_content, page_num + 1)
        
        return None
    
    def extract_table(self, page_content: str) -> pd.DataFrame:
        """Extract table from page content using GPT-4."""
        prompt = ChatPromptTemplate.from_template("""
        Extract the table from the following text and format it as a CSV string. 
        Only include the actual table data, with headers in the first row.
        Separate columns with commas and rows with newlines.
        Text: {text}
        CSV Table:
        """)
        
        # Get CSV string from GPT-4
        messages = prompt.format_messages(text=page_content)
        response = self.llm.invoke(messages)
        csv_string = response.content.strip()
        
        # Convert CSV string to DataFrame
        try:
            return pd.read_csv(pd.StringIO(csv_string))
        except Exception as e:
            raise ValueError(f"Failed to parse table: {str(e)}")
    
    def process_pdf(self, pdf_path: str, table_description: str = None) -> tuple[pd.DataFrame, int]:
        """Process PDF and extract table data into DataFrame.
        Returns tuple of (DataFrame, page_number)"""
        # Load PDF
        pages = self.load_pdf(pdf_path)
        
        if table_description:
            # Find specific table
            result = self.find_table_page(pages, table_description)
            if not result:
                raise ValueError("Could not find the specified table in the PDF")
            page_content, page_num = result
        else:
            # Use first page if no description provided
            page_content = pages[0].page_content
            page_num = 1
        
        # Extract table
        df = self.extract_table(page_content)
        return df, page_num

def main():
    # Example usage
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize extractor
    extractor = PDFTableExtractor(api_key)
    
    # Process PDF
    try:
        # Extract specific table
        df, page_num = extractor.process_pdf(
            pdf_path="path/to/your/pdf",
            table_description="description of the table you're looking for"
        )
        print(f"Extracted table from page {page_num}:")
        print(df)
        
        # Or extract first table found
        df, page_num = extractor.process_pdf(
            pdf_path="path/to/your/pdf"
        )
        print(f"Extracted first table found (page {page_num}):")
        print(df)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 