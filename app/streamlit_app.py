"""
streamlit_app.py

Purpose:
    Streamlit frontend application for the SHL Recommender system.
    Connects to the FastAPI backend to retrieve recommendations.
"""

import os
import requests
import streamlit as st
import pandas as pd

# Configure page settings
st.set_page_config(
    page_title="SHL Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API endpoint (adjust if your backend is running elsewhere)
API_URL = os.environ.get("API_URL", "https://shl-recommender-cszf.onrender.com/recommend")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        color: #212529;
        margin-bottom: 1rem;
        border-left: 5px solid #0056b3;
    }
    .result-card a {
        color: #0056b3;
        text-decoration: none;
        font-weight: 500;
    }
    .result-card a:hover {
        text-decoration: underline;
    }
    h1 {
        color: #0056b3;
    }
    .recommendation-title {
        color: #212529;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_recommendations(query: str):
    """Fetches recommendations from the FastAPI backend."""
    try:
        response = requests.post(
            API_URL,
            json={"query": query},
            timeout=60  # Increased timeout to allow for LLM reranking
        )
        response.raise_for_status()
        return response.json().get("recommendations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend API: {e}")
        return None


def main():
    st.title("🎯 SHL Assessment Recommender")
    st.markdown("Find the best SHL assessments based on specific skills, roles, or attributes.")

    # Search section
    st.subheader("Search")
    query = st.text_input(
        "Describe what you are looking for in an assessment:",
        placeholder="e.g., Python developer with strong communication skills and SQL experience",
        key="search_query"
    )

    if st.button("Recommend Assessments", type="primary"):
        if not query.strip():
            st.warning("Please enter a query to search.")
            return

        with st.spinner("Analyzing your request and generating recommendations..."):
            recommendations = get_recommendations(query)

        if recommendations is None:
            return  # Error occurred
            
        if not recommendations:
            st.info("No relevant assessments found for your query. Try different keywords.")
            return

        st.subheader("Top Recommendations")
        st.success(f"Found {len(recommendations)} matches for your query.")
        
        # Display results in structured format 
        for idx, rec in enumerate(recommendations, start=1):
            name = rec.get('name', 'Unknown Assessment')
            url = rec.get('url', '#')
            
            st.markdown(f"""
            <div class="result-card">
                <div class="recommendation-title">{idx}. {name}</div>
                <a href="{url}" target="_blank">View Assessment Details &rarr;</a>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
