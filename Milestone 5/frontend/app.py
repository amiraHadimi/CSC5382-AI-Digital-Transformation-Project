import streamlit as st
import requests

API_URL = "http://api:8000/predict"

st.title("LLAMA3SP Story Point Estimator")
st.write("Enter a Jira issue and get predicted story points.")

title = st.text_input("Issue Title")
description = st.text_area("Issue Description")

if st.button("Predict"):
    if title and description:
        payload = {
            "title": title,
            "description": description
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=600)
            result = response.json()

            st.success(f"Story Points: {result['story_points']}")
            st.info(f"Model used: {result['model_used']}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please fill all fields")