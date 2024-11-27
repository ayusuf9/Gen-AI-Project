import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import jwt
import os
from typing import Literal
from pages.tables import initialize_session_state
from pages.Chat import chat

# Constants for AD Groups
AD_GROUPS = ["authorized-group-1", "authorized-group-2"]  # Replace with your actual authorized groups

def authorize_user() -> Literal["authorized", "token_expired", "no_token", "decode_error", "invalid_groups"]:
    """
    Authorize user based on JWT token from query parameters
    """
    try:
        token = st.query_params["token"]
        token = token.removeprefix("Bearer ")
        os.environ["TOKEN"] = token
    except KeyError:
        st.session_state.logger.debug("Token not passed as query parameter.")
        return "no_token"

    try:
        decoded_data = jwt.decode(jwt=token, verify=False, algorithms=["RS256"])
        st.session_state.logger.debug("Got JWT: {}".format(decoded_data))
        exp = datetime.fromtimestamp(decoded_data["exp"])
        user_groups = decoded_data["groups"]
    except Exception as e:
        st.session_state.logger.error(e)
        return "decode_error"

    pst_now_dt = datetime.utcnow() - timedelta(hours=8)
    pst_now_ts = int(pst_now_dt.timestamp())

    if pst_now_ts > int(decoded_data["exp"]):
        return "token_expired"

    if any(group in AD_GROUPS for group in user_groups):
        st.session_state.logger.debug("User authenticated successfully")
        return "authorized"

    st.session_state.logger.debug(
        "JWT decoded successfully but user not in necessary AD groups."
    )
    return "invalid_groups"

def initialize_auth_state():
    """Initialize authentication-related session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = None

def show_auth_error():
    """Display appropriate error message based on authentication status"""
    auth_messages = {
        "token_expired": "Your session has expired. Please log in again.",
        "no_token": "No authentication token found. Please log in.",
        "decode_error": "Authentication error. Please try logging in again.",
        "invalid_groups": "You don't have permission to access this application."
    }
    if st.session_state.auth_status in auth_messages:
        st.error(auth_messages[st.session_state.auth_status])
        return True
    return False

def main():
    st.set_page_config(
        page_title="Municipal Document Genie",
        page_icon="qh-logo.svg",
        layout="wide"
    )

    # Initialize authentication state
    initialize_auth_state()

    # Check authorization
    if not st.session_state.authenticated:
        st.session_state.auth_status = authorize_user()
        if st.session_state.auth_status == "authorized":
            st.session_state.authenticated = True
        else:
            show_auth_error()
            return

    # Your existing CSS and styling code
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Only show main content if authenticated
    if st.session_state.authenticated:
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
            image, title, _, _ = st.columns([1,14,1,1], gap="small")
            with image:
                st.image("qh-logo.svg", width=80)
            with title:
                st.markdown('<div class="header">Muni Document Genie</div>', unsafe_allow_html=True)

        # Rest of your existing application code
        st.markdown(f"<div class='subheader'>Effortlessly read and interact with municipal bond documents using our advanced question and answering capabilities. Simplify your document analysis and get the insights you need in real-time.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='footnote'>For any support on the application please contact <a href='' target='_blank'>QuantHub_Support@capgroup.com</a></div>", unsafe_allow_html=True)
        st.markdown(f"<hr style='margin: 10px 0; padding: 0;'>", unsafe_allow_html=True)

        # Main content columns
        table, vertical_divider, chat_col = st.columns([0.70, 0.005, 0.30])
        
        with table:
            st.markdown(f"<div class='col-header'>Muni Analysis</div>", unsafe_allow_html=True)
            # Your existing table code...

        with vertical_divider:
            # Your existing divider code...
            pass

if __name__ == "__main__":
    main()