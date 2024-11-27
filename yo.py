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
                gap: 1rem;
                max-width: 1200px;
                margin: 0 auto;
            }
            .logo-image {
                flex: 0 0 80px;
            }
            .header-title {
                flex: 1;
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
            </style>
        """, unsafe_allow_html=True)

        tabs1, tabs2 = st.columns(2)
        
        with tabs1:
            # Use a single HTML structure for the header
            st.markdown("""
                <div class="header-container">
                    <div class="logo-image">
                        <img src="qh-logo.svg" width="80">
                    </div>
                    <div class="header-title">
                        <div class="header">Muni Document Genie</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with tabs2:
            st.markdown(f"""<div class='footnote'>
                For any support on the application please contact 
                <a href='' target='_blank'>QuantHub_Support@capgroup.com</a>
                </div>""", unsafe_allow_html=True)