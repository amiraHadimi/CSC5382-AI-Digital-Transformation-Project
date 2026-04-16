import streamlit as st
from model import predict_story_points

st.set_page_config(page_title="LLAMA3SP Story Point Estimator")

st.title("LLAMA3SP Story Point Estimator")
st.write("Enter a Jira issue and get predicted story points.")

title = st.text_input("Issue Title")
description = st.text_area("Issue Description")

if st.button("Predict"):
    if not title or not description:
        st.warning("Please fill both fields.")
    else:
        points, model_used = predict_story_points(title, description)
        st.success(f"Story Points: {points}")
        st.info(f"Model used: {model_used}")