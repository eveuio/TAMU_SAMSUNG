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

#store emoji 
redIndicator = "🔴"
yellowIndicator = "🟡"
greenIndicator = "🟢"

#
if "read_error" not in st.session_state:
    st.session_state["read_error"] = False
xfmr_json = requests.get("http://localhost:8000/transformers/").json()
if "detail" in xfmr_json:
    st.session_state["read_error"] = True
elif xfmr_json == []:
    st.session_state["read_error"] = True
else:
    st.session_state["list"] = xfmr_json



xfmrDisplay = st.Page("xfmrDisplay.py", title = "Transformer Data Display")
createDeletePage = st.Page("createDeletePage.py",title = "Create/Delete Transformer")
errorPage = st.Page("errorPage.py",title = "error")
if st.session_state["read_error"] == False:
    pg = st.navigation([xfmrDisplay,createDeletePage])
else:
    pg = st.navigation([errorPage])
pg.run()