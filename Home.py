import streamlit as st
import pandas as pd
import numpy as np
from pages.tables import initialize_session_state
from pages.Chat import chat

st.set_page_config(
     page_title="Municipal Document Genie",
     page_icon="qh-logo.svg",
     layout="wide"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# TODO: Move this custom css into a different file
custom_css = """
<style>
/* Add this at the beginning of your CSS */
div.block-container {
    padding-top: 1rem !important;
}

.header {
    text-align: left;
    padding: 0;
    margin: 0;
    padding-bottom: 0.15em;
    font-weight: 600;
    font-size: 3em;
    line-height: 1.2;
}
.subheader {
    padding: 0;
    margin: 0;
    border-radius: 0;
    font-size: 20px;
    font-weight: 100;
    line-height: 1.4;
}
.col-header {
    text-align: left;
    padding: 0;
    margin: 0;
    font-weight: 300;
    font-size: 1.5em;
    line-height: 1.2;
}
.divider {
    padding: 0;
    margin: 10px 0;
}
.footnote {
    text-align: left;
    padding: 0;
    margin: 5px 0;
    font-size: 0.75em;
}
.main-content {
    padding-top: 0;
    margin-top: -20px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px;
    color: #666;
    font-size: 16px;
    font-weight: 400;
}
.stTabs [aria-selected="true"] {
    background-color: rgb(70, 130, 180) !important;
    color: white !important;
}
span[data-baseweb="tag"] {
    background-color: rgb(70, 130, 180) !important;
}
/* Table styling */
.table-container {
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
}
.dataframe {
    width: 100% !important;
    max-width: none !important;
}
.stDataFrame {
    width: 100%;
}

/* Chat container styling */
.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 300px);
    overflow-y: auto;
    padding: 1rem;
}

.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-input {
    position: sticky;
    bottom: 0;
    background-color: white;
    padding: 1rem 0;
    border-top: 1px solid #e0e0e0;
}

/* Message styling */
.chat-message {
    margin-bottom: 1rem;
    padding: 0.5rem;
    border-radius: 4px;
}

.user-message {
    background-color: #f0f0f0;
}

.assistant-message {
    background-color: #f8f9fa;
}
</style>
"""

def style_dataframe(df):
    """Style the results table"""
    return df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '16px',
        'padding': '8px',
        'border': '1px solid lightgrey',
        'width': '100%',
        'min-width': '100%'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#e0e0e0'),
            ('font-size', '15px'),
            ('white-space', 'nowrap'),
            ('min-width', '150px'),
            ('max-width', '200px')
        ]},
        {'selector': 'tr:nth-of-type(even)', 'props': [
            ('background-color', '#f9f9f9')
        ]},
        {'selector': 'table', 'props': [
            ('width', '100%'),
            ('table-layout', 'fixed'),
            ('margin-left', '0px'),
            ('margin-right', '0px')
        ]},
        {'selector': 'td', 'props': [
            ('max-width', '200px'),
            ('overflow', 'hidden'),
            ('text-overflow', 'ellipsis'),
            ('white-space', 'nowrap')
        ]}
    ])

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
with st.container():
    st.markdown("""
        <style>
        [data-testid="stVerticalBlock"] {
            padding-top: 0;
            margin-top: -2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    image, title,_,_ = st.columns([1,14,1,1],gap="small")
    with image:
        st.image("qh-logo.svg", width=80)
    with title:
        st.markdown('<div class="header">Muni Document Genie</div>', unsafe_allow_html=True)

# Reduced spacing for subheader and footnote
st.markdown(f"<div class='subheader'>Effortlessly read and interact with municipal bond documents using our advanced question and answering capabilities. Simplify your document analysis and get the insights you need in real-time.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='footnote'>For any support on the application please contact <a href='' target='_blank'>QuantHub_Support@capgroup.com</a></div>", unsafe_allow_html=True)

# Shorter divider with less padding
st.markdown(f"<hr style='margin: 10px 0; padding: 0;'>", unsafe_allow_html=True)

# Content with reduced top spacing
st.markdown('<div class="main-content">', unsafe_allow_html=True)
table, vertical_divider, chat_col = st.columns([0.70, 0.005, 0.30])

with table:
    st.markdown(f"<div class='col-header'>Muni Analysis</div>", unsafe_allow_html=True)
    
    # Initialize session state for active tab if not exists
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Table View"
    
    # Create tabs
    tab1, tab2 = st.tabs(["Table View", "Chat View"])
    
    with tab1:
        st.write("Compare PreLoaded Municipal Documents")
        st.success("Select Documents of your choice to display result")
        st.warning("The function to add more questions is underway...")
        
        # Initialize session state and display table
        initialize_session_state()
        
        if 'results_df' in st.session_state:
            available_docs = st.session_state.results_df.columns.tolist()
            selected_docs = st.multiselect("Select documents:", available_docs, default=available_docs[:5])
            
            if selected_docs:
                display_df = st.session_state.results_df[selected_docs]
                styled_df = style_dataframe(display_df)
                st.markdown('<div class="table-container">', unsafe_allow_html=True)
                st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add download button
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='display_data.csv',
                    mime='text/csv',
                    key='download-csv'
                )
                
                st.markdown(f"<div class='footnote'>Note: The ability to add more questions to the list is in progress.</div>", unsafe_allow_html=True)
        else:
            st.warning("No data available. Please ensure the data is properly loaded.")
    
    with tab2:
        st.markdown("""
            <style>
            .chat-content {
                display: flex;
                flex-direction: column;
                height: auto;
                min-height: 500px;
            }
            .upload-section {
                padding: 1rem 0;
            }
            .chat-messages {
                display: flex;
                flex-direction: column;
            }
            .stChatFloatingInputContainer {
                position: relative !important;
                margin-top: 1rem !important;
                background: white !important;
                padding: 1rem !important;
                border-top: 1px solid #e0e0e0 !important;
            }
            [data-testid="stChatInput"] {
                max-width: 800px;
                margin: 0 auto;
            }
            .stMarkdown {
                min-height: 0 !important;
            }
            /* Ensure chat messages stack properly */
            div[data-testid="stChatMessageContent"] {
                margin-bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)
        
        # Upload section at the top
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # Chat messages container
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        chat()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
with vertical_divider:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border: 2px solid white;
                        height: 50em;
                        margin-right: 1em;
                        margin-left: .1em;
                    }
                </style>
            '''
        )