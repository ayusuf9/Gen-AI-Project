import os
import openai
import json
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from streamlit_navigation_bar import st_navbar
from streamlit_extras.switch_page_button import switch_page
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from quanthub.util import llm

CACHE_FILE = "/app/pdfs_qa/modified_data.json"

def initialize_session_state():
    session_vars = {
        'custom_questions': [],
        'qa_results': pd.DataFrame(),
        'processed_documents': set(),
        'vector_stores': {},
        'embeddings': None,
        'llm': None
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

    if 'results_df' not in st.session_state:
        st.session_state.results_df = load_cached_results()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm_model()
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = load_embeddings_model()

def load_cached_results():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                return pd.DataFrame.from_dict(cache_data.get('results', {}))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading cache: {str(e)}")
        return pd.DataFrame()

def load_embeddings_model():
    openai_api_client = llm.get_azure_openai_client()
    try:
        return OpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type=openai_api_client.api_type,
            chunk_size=1
        )
    except Exception as e:
        st.error(f"Failed to load embeddings: {str(e)}")
        return None

def load_llm_model():
    openai_api_client =llm.get_azure_openai_client()
    try:
        return AzureChatOpenAI(
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            openai_api_version="2024-02-01",
            openai_api_key=openai_api_client.api_key,
            openai_api_base=openai_api_client.api_base,
            openai_api_type="azure_ad",
            temperature=0.0,
            streaming=True
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

def get_multi_query_retriever(vector_store):
    """Setup retriever"""
    question_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert in municipal finance. Break down the question:
        Question: {question}
        Generate similar alternative questions."""
    )
    
    try:
        return MultiQueryRetriever.from_llm(
            llm=st.session_state.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            parser_key="questions",
            prompt=question_prompt
        )
    except Exception as e:
        st.error(f"Retriever error: {str(e)}")
        return None

def process_queries(vector_stores, queries, progress_bar=None):
    """Process queries against vector stores"""
    if not st.session_state.llm:
        st.error("LLM not initialized properly")
        return pd.DataFrame()

    chat_prompt = PromptTemplate(
        template="""
            **Question:**

            {question}

            **Instructions for the AI Assistant:**

            You are an expert investment analyst specializing in analyzing municipal bond issuance documents, particularly those from MuniOs.com. Your role is to provide accurate, insightful, and professional answers to questions about municipal bonds, just like a seasoned financial professional in a leading financial company.

            When generating your response, please follow these steps:

            1. **Understand the Question and Context**: Carefully read the question and the provided context to fully comprehend what is being asked.

            2. **Extract Relevant Information**: Identify the key pieces of information in the context that are pertinent to answering the question.

            3. **Perform Detailed Analysis**: Analyze the extracted information, performing any necessary calculations or evaluations. Ensure all calculations are accurate and clearly explained.

            4. **Synthesize Your Findings**: Combine your analysis into a coherent response that addresses all aspects of the question.

            5. **Provide a Clear and Professional Answer**: Present your conclusions in a concise and precise manner, using proper financial terminology and maintaining a professional tone.

            **Guidelines:**

            - **Use Only the Provided Context**: Base your answer solely on the information given in the context. Do not include information that is not present in the context. Avoid introducing any outside knowledge or assumptions.

            - **Accuracy and Precision**: Ensure all information you provide is correct, especially numerical data and financial terms.

            - **Calculations**: Show all relevant calculation steps and provide the final results.

            - **Clarity and Professionalism**: Write in clear, concise language appropriate for a professional audience.

            **Note**:

            - The total debt is the sum of the principal amounts without interest.

            **Context:**

            {context}

            **Answer:**

            [Provide your detailed analysis and final answer here.]
            """
                ,
        input_variables=["question", "context"]
    )
    
    simplify_prompt = PromptTemplate(
        input_variables=["answer"],
        template="""
          Given the following answer, extract and return only the key point. The key point should be a concise summary that directly conveys the main information, such as:
  
          - For property addresses, extract only the **city and state** (e.g., 'Santa Rosa, California').
          - If a **year** is mentioned, extract the year (e.g., '2003').
          - For **quantities or counts**, extract the numerical value (e.g., '277').
          - For **percentages or rates**, extract the numerical value along with the percent symbol (e.g., '3.50%').
          - A specific number along with its unit or currency symbol (e.g., '180 units', '$65,338.68').
          - A **percentage** (e.g., '94.63%').
          - A **name or proper noun** (e.g., 'Waterscape Apartments', 'Riverside County, California').
          - A **brief descriptive phrase** that includes necessary qualifiers (e.g., 'trending positively', 'decreased by $59,800.84', 'increase of $1,243.72').
  
          Do not include any additional explanation, context, or restatement of the answer. **Provide only the key point as it directly relates to the main information.**
  
          **Examples:**
  
          1. **Answer:** The subject property is located at 4656 Quigg Drive, Santa Rosa, California 95409.
  
          **Key point:** Santa Rosa, California
  
          2. **Answer:** The property located at 4656 Quigg Drive, Santa Rosa, California 95409 was built in 2003.
  
          **Key point:** 2003
  
          3. **Answer:** The total number of units at the property located at 4656 Quigg Drive, Santa Rosa, California 95409 is 277 residential apartment units.
  
          **Key point:** 277
  
          4. **Answer:** The Cap Rate for the property located at 4656 Quigg Drive, Santa Rosa, California 95409 is 3.50%. This is based on the concluded going-in capitalization rate derived from the income capitalization approach and market participant discussions.
  
          **Key point:** 3.50%
  
          5. **Answer:** The property is located at Waterscape Apartments, identified by the code "cawater2."
  
          **Key point:** Waterscape Apartments
  
          6. **Answer:** The property, Waterscape Apartments, is located in Riverside County, California.
  
          **Key point:** Riverside County, California
  
          7. **Answer:** The change in total income over the last three months is $65,338.68.
  
          **Key point:** $65,338.68
  
          8. **Answer:** The most recent occupancy rate is 94.63%.
  
          **Key point:** 94.63%
  
          9. **Answer:** The total number of units is 180.
  
          **Key point:** 180 units
  
          10. **Answer:** The occupancy rate is trending positively.
  
              **Key point:** trending positively
  
          11. **Answer:** The total concessions as a percentage of total income are approximately -1.05%.
  
              **Key point:** -1.05%
  
          12. **Answer:** The total property expenses decreased by $59,800.84 over the last three months.
  
              **Key point:** decreased by $59,800.84
  
          13. **Answer:** The change in total property expenses over the last 12 months is an increase of $1,243.72.
  
              **Key point:** increase of $1,243.72
  
          Now, given the following answer, extract and return just the **key point**.
  
          **Answer:**
          {answer}
  
          **[final answer here]**
                  """
  
    )
    
    simplify_chain = LLMChain(llm=st.session_state.llm, prompt=simplify_prompt)
    
    results = pd.DataFrame(columns=list(vector_stores.keys()), index=queries)
    total_operations = len(queries) * len(vector_stores)
    current_operation = 0
    
    for identifier, vector_store in vector_stores.items():
        multi_retriever = get_multi_query_retriever(vector_store)
        if not multi_retriever:
            continue
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=multi_retriever,
            chain_type_kwargs={"prompt": chat_prompt}
        )
        
        for query in queries:
            try:
                result = qa_chain.run(query)
                simplified_result = simplify_chain.run(answer=result)
                results.at[query, identifier] = simplified_result.strip()
            except Exception as e:
                results.at[query, identifier] = f"Error: {str(e)}"
            
            current_operation += 1
            if progress_bar is not None:
                progress_bar.progress(current_operation / total_operations)
    
    return results


def load_vector_stores(identifier, base_path):
    try:
        if identifier in st.session_state.vector_stores:
            return st.session_state.vector_stores[identifier]
        
        index_folder = os.path.join(base_path, f"{identifier}_faiss_index")
        
        if not os.path.exists(index_folder):
            st.error(f"Index folder not found for {identifier}")
            return None
        
        vector_store = FAISS.load_local(
            index_folder,
            embeddings=st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vector_stores[identifier] = vector_store
        return vector_store
    except Exception as e:
        st.error(f"Error loading index for {identifier}: {str(e)}")
        return None


def style_dataframe(df):
    """Style the results table"""
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'padding': '8px',
        'border': '1px solid lightgrey'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#e0e0e0'),
            #('font-weight', 'bold'),
            ('font-size', '15px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]}
    ])


def tables():
  initialize_session_state()

  st.success("Select Documents of your choice to display result")
  st.warning("The function to add more questions is underway...")

  
  if 'results_df' not in st.session_state:
      st.error("No data loaded. Please initialize results_df first.")
      return

  available_docs = st.session_state.results_df.columns.tolist()
  st.markdown(
      """
  <style>
  span[data-baseweb="tag"] {
    background-color: rgb(70, 130, 180) !important;
  }
  </style>
  """,
      unsafe_allow_html=True,
  )
  
  selected_docs = st.multiselect("Select documents:", available_docs, default=available_docs[:5])
  
  if selected_docs:
      display_df = st.session_state.results_df[selected_docs]
      styled_df = style_dataframe(display_df)
      st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

      csv = display_df.to_csv(index=False).encode('utf-8')
      st.download_button(
            label="Download CSV",
            data=csv,
            file_name='display_data.csv',
            mime='text/csv',
            key='download-csv'
        )

      default_questions = [
          "Where is this property located?",
          "What year was the property built",
          "What is the total number of units?",
          "What is the Cap Rate?",
          "What is the Total Senior Debt?",
          "What is the Total Mezzanine Debt?",
          "What is the Total Debt?",
          "How much is in the coverage reserve fund",
          "How much is in the senior capitalized interest fund",
          "How much is in the mezzanine capitalized interest",
          "What is the Rental Revenues in 2024",
          "Net Operating Income in 2024?"
      ]

if __name__ == "__main__":
  tables()