import openai
import os
import streamlit as st
import time
import re
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from streamlit_navigation_bar import st_navbar
# from streamlit_extras.switch_page_button import switch_page
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain_community.document_loaders.parsers.pdf import PDFPlumberParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.document_loaders.blob_loaders import Blob
from pydantic import BaseModel, Field
from quanthub.util import llm
import pages as pg

def initialize_session_state():
  openai_api_client = llm.get_azure_openai_client()
  if 'uploaded_file' not in st.session_state:
      st.session_state.uploaded_file = None
  if 'vector_store' not in st.session_state:
      st.session_state.vector_store = None
  if 'chat_history' not in st.session_state:
      st.session_state.chat_history = []
  
  if 'embedding_model' not in st.session_state:
      st.session_state.embedding_model = get_embedding_model()
  if 'llm_instance' not in st.session_state:
      st.session_state.llm_instance = load_llm_model()
  
  if 'processing_status' not in st.session_state:
      st.session_state.processing_status = False
  if 'last_processed_file' not in st.session_state:
      st.session_state.last_processed_file = None


def load_llm_model():
    openai_api_client = llm.get_azure_openai_client()
    try:
        llm_instance = AzureChatOpenAI(model='gpt-4o',
                                    deployment_name=llm.GPT_4_OMNI_MODEL,
                                    openai_api_version="2024-02-15-preview",
                                    openai_api_key=openai_api_client.api_key,
                                    openai_api_base=openai_api_client.api_base,
                                    openai_api_type=openai_api_client.api_type,
                                    temperature=0)
        return llm_instance
    except Exception as e:
        st.error("Failed to load LLM, please refresh.")
        st.session_state.stderr_logger.error("Failed to load LLM: %s", e)



@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(Exception),
)
def embed_texts(texts, embedding_model):
    return embedding_model.embed_documents(texts)

def get_embedding_model():
    openai_api_client = llm.get_azure_openai_client()
    return OpenAIEmbeddings(
        deployment='text-embedding-ada-002',
        model='text-embedding-ada-002', 
        openai_api_key=openai_api_client.api_key,
        openai_api_base=openai_api_client.api_base,
        openai_api_type=openai_api_client.api_type,
        chunk_size=100
    )

def get_avatar(role):
  if role == "user":
      return ":material/account_box:"
  elif role == "assistant":
      return ":material/psychology_alt:"
  else:
      return None 

def process_pdf(pdf_file) -> Optional[FAISS]:
  with st.spinner(text="Processing PDF..."):

      blob = Blob(
          data=pdf_file.getvalue(),
          source=pdf_file.name,
          blob_type="application/pdf",
      )

      parser = PDFPlumberParser()

      documents = parser.parse(blob)
      documents = []
      for page_num, doc in enumerate(parser.parse(blob), start=1):
          doc.metadata['page_number'] = page_num
          documents.append(doc)

      batch_size = 10
      all_texts = []
      all_embeddings = []

      st.markdown("""
          <style>
              .stProgress > div > div > div > div {
                  background-color: rgb(70, 130, 180);
              }
          </style>
      """, unsafe_allow_html=True)

      progress_bar = st.progress(0)
      for i in range(0, len(documents), batch_size):
          batch_docs = documents[i:i + batch_size]
          texts = [doc.page_content for doc in batch_docs]
          embeddings = embed_texts(texts, st.session_state.embedding_model)
          all_texts.extend(batch_docs)
          all_embeddings.extend(embeddings)
          progress = min((i + batch_size) / len(documents), 1.0)
          progress_bar.progress(progress)
          time.sleep(0.1)
      
      pure_texts = [page.page_content for page in all_texts]
      textual_embeddings = list(zip(pure_texts, all_embeddings))
      vector_store = FAISS.from_embeddings(textual_embeddings, st.session_state.embedding_model)

  return vector_store


def chat():
    openai_api_client = llm.get_azure_openai_client()
    initialize_session_state()

    def clean_latex(text):
        cleaned = re.sub(r'$ *\\text\{([^}]*)\}', r'\1', text)
        cleaned = cleaned.replace('\\approx', '≈')
        cleaned = re.sub(r'\$([^$]*)\$', r'\1', cleaned)
        cleaned = cleaned.replace('\\', '')
        cleaned = cleaned.replace('[', '').replace(']', '')
        return cleaned

    uploaded_file = st.file_uploader(
        "Upload a Muni PDF document",
        type="pdf",
        help="Upload a PDF file to start chatting"
    )


    if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.processing_status = True
        vector_store = process_pdf(uploaded_file)
        
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.chat_history = []
            st.session_state.last_processed_file = uploaded_file.name
            st.session_state.processing_status = False
            st.success(f"Successfully processed {uploaded_file.name}")

    if st.session_state.vector_store is not None:
        st.divider()

        base_retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 20})
        class LineListOutputParser(BaseOutputParser[List[str]]):
            def parse(self, text: str) -> List[str]:
                lines = text.strip().split("\n")
                return list(filter(None, lines)) 


        output_parser = LineListOutputParser()

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are a Financial Analyst AI language model assistant. Your task is to generate three 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. Be sure to Focus on municipal bond and financial analysis perspectives. 
            By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )


        llm_chain = QUERY_PROMPT | st.session_state.llm_instance | output_parser

        multiquery_retriever = MultiQueryRetriever(retriever=base_retriever, llm_chain=llm_chain, parser_key="lines")

        chat_prompt = PromptTemplate(
            template="""
                Question: {question}
                
                Context (This is the data/information you should use in answering the question):
                  {context}
        
                  You are a seasoned financial analyst specializing in analyzing municipal bond prospectuses. 
                  You provide accurate and insightful answers to questions, just like a financial professional would.               
                  Your response should be structured as follows:
                  
                  When presented with a prospectus and a question, focus on the following:
        
                    1. **Accuracy is key**: Provide precise and correct information based on the prospectus content.
                    2. **Calculations matter**: When asked to calculate, ensure your calculations are accurate and reflect a professional standard.
                    3. **Context is crucial**: Frame your answers within the context of the specific bond issue and the overall municipal bond market.
                    
                    For example, if asked:
                    
                    (a). "What is the total debt?" Accurately calculate the answer based on the provided financial statements. 
                          Note that total debt is the sum of all Series principal amounts (e.g., Series A, Series B, etc.) listed in the document. 
                    (b). "What is the purpose of this bond issuance?" provide a clear and concise answer directly from the prospectus.
                    (c). "What are the risks associated with this bond?" analyze and explain the risk factors outlined in the prospectus.
                    
                    Remember, when responding, ensure to draw upon your deep understanding of municipal credit fundamentals and use industry-standard financial calculations and metrics to provide precise calculations, and clear explanations.
        
                  Your response should be structured as follows and do not use latex to output calculations:
                  Answer: [Provide a clear, precise, and concise answer to the question, including only the most relevant information and numerical results.]
                  Sources: [List the page numbers where the information was found in the format "Page X, Page Y, etc."]
        
                    """,
            input_variables=["question", "context"]
          )
      
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm_instance,
            retriever=multiquery_retriever,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
            return_source_documents=True,
        )

        try:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"], avatar=get_avatar(message["role"])):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What is your question?"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=":material/account_box:"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar=":material/psychology_alt:"):
                    with st.spinner("Thinking..."):
                        response = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]})
                        st.markdown(clean_latex(response['answer']))
                        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

        except Exception as e:
            st.error("An error occurred while processing your request.")

if __name__ == "__main__":
  chat()