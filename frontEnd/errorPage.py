import streamlit as st
import requests
def retry():
    xfmr_json = requests.get("http://localhost:8000/transformers").json()
    if "detail" in xfmr_json or xfmr_json == []:
        st.session_state["read_error"] = True
        st.session_state["error_message"] = "No transformers found."
    else:
        xfmr_data = requests.get(f"http://localhost:8000/transformers/{xfmr_json[0]['transformer_name']}")
        if xfmr_data == []:
            st.session_state["read_error"] = True
        else:
            st.session_state["read_error"] = False
st.write("A read error has occured. Check the database and restart programs.")
st.write("Error: " + st.session_state["error_message"])
st.button("Retry",on_click = retry)
