def main():
    initialize_auth_state()

    if st.session_state.authenticated:
        # Custom CSS to control the layout spacing
        st.markdown("""
            <style>
            [data-testid="stVerticalBlock"] {
                padding-top: 0;
                margin-top: 0rem;
            }
            .header-container {
                display: flex;
                align-items: center;
                padding: 0;
                margin: 0;
            }
            .header-title {
                margin-left: 1rem;
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
            [data-testid="column"] {
                padding: 0 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        tabs1, tabs2 = st.columns(2)
        
        with tabs1:
            # Create a container for the header row
            header_container = st.container()
            
            # Use columns with minimal width ratio for logo
            logo_col, title_col = header_container.columns([1, 20])
            
            with logo_col:
                st.image("qh-logo.svg", width=80)
            
            with title_col:
                st.markdown('<div class="header">Muni Document Genie</div>', unsafe_allow_html=True)
            
        with tabs2:
            st.markdown(f"""<div class='footnote'>
                For any support on the application please contact 
                <a href='' target='_blank'>QuantHub_Support@capgroup.com</a>
                </div>""", unsafe_allow_html=True)