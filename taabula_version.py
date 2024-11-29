import os
from typing import Optional, List, Tuple
import pandas as pd
import tabula
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

class PDFTableExtractor:
    def __init__(self, api_key: str):
        """
        Initialize the PDFTableExtractor with OpenAI API key.
        
        Args:
            api_key (str): OpenAI API key for embeddings and chat completion
        """
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        
    def create_vector_db(self, pdf_path: str) -> FAISS:
        """
        Create a vector database from the PDF content for semantic search.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            FAISS: Vector store containing document embeddings
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store
    
    def find_table_page(self, vector_store: FAISS, table_description: str) -> int:
        """
        Find the most likely page containing the desired table.
        
        Args:
            vector_store (FAISS): Vector store containing document embeddings
            table_description (str): Description of the table to find
            
        Returns:
            int: Most likely page number containing the table
        """
        docs = vector_store.similarity_search(
            f"Find a table that contains {table_description}",
            k=3
        )
        
        page_numbers = [doc.metadata['page'] + 1 for doc in docs]
        return max(set(page_numbers), key=page_numbers.count)
    
    def extract_table(self, pdf_path: str, page_number: int) -> List[pd.DataFrame]:
        """
        Extract tables from the specified page using tabula.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_number (int): Page number to extract tables from
            
        Returns:
            List[pd.DataFrame]: List of extracted tables as pandas DataFrames
        """
        # Extract all tables from the specified page
        tables = tabula.read_pdf(
            pdf_path,
            pages=page_number,
            multiple_tables=True,
            guess=True,
            lattice=True,
            stream=True
        )
        
        return tables
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format the extracted DataFrame.
        
        Args:
            df (pd.DataFrame): Raw extracted DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean whitespace in string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Handle potential multi-level headers
        if df.columns.nlevels > 1:
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        return df
    
    def extract_and_process_table(
        self,
        pdf_path: str,
        table_description: str
    ) -> Tuple[pd.DataFrame, int]:
        """
        Main method to extract and process a table from a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            table_description (str): Description of the table to find
            
        Returns:
            Tuple[pd.DataFrame, int]: Cleaned DataFrame and page number where table was found
        """
        # Create vector store
        vector_store = self.create_vector_db(pdf_path)
        
        # Find table page
        page_number = self.find_table_page(vector_store, table_description)
        
        # Extract tables
        tables = self.extract_table(pdf_path, page_number)
        
        if not tables:
            raise ValueError(f"No tables found on page {page_number}")
        
        # Select largest table if multiple found
        largest_table = max(tables, key=lambda df: df.size)
        
        # Clean and return table
        cleaned_table = self.clean_dataframe(largest_table)
        return cleaned_table, page_number

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    api_key = "your-openai-api-key"
    extractor = PDFTableExtractor(api_key)
    
    # Extract table
    pdf_path = "path/to/your/pdf"
    table_description = "financial statements showing revenue and expenses"
    
    try:
        df, page = extractor.extract_and_process_table(pdf_path, table_description)
        print(f"Table found on page {page}")
        print("\nExtracted Table:")
        print(df)
        
        # Optionally save to CSV
        df.to_csv("extracted_table.csv", index=False)
    except Exception as e:
        print(f"Error: {str(e)}")