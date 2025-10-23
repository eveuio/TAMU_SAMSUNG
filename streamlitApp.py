import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

#Set page config
st.set_page_config(page_title = "Transformer Health Monitoring", layout = "wide")
if "data" not in st.session_state:
    st.session_state["data"] = []
if "id" not in st.session_state:
    st.session_state["id"] = 0
if "loggedin" not in st.session_state:
    st.session_state["loggedin"] = True

#store emoji 
redIndicator = "🔴"
yellowIndicator = "🟡"
greenIndicator = "🟢"

#
xfmr_json = requests.get("http://localhost:8000/transformers/").json()
st.session_state["list"] = xfmr_json



xfmrDisplay = st.Page("xfmrDisplay.py", title = "Transformer Data Display")
createDeletePage = st.Page("createDeletePage.py",title = "Create/Delete Transformer")
pg = st.navigation([xfmrDisplay,createDeletePage])
pg.run()