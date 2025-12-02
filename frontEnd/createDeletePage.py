import streamlit as st
import requests
import time
import datetime


col1, col2 = st.columns(2)


def createxfmr(xfmrdict,upload_file):
    createrequest = requests.post("http://localhost:8000/transformers/", json=xfmrdict)
    if upload_file is not None:
        upload_file.name = f"{xfmr_name}.xlsx"

        b = upload_file.getvalue()
        with open(f"../DataProcessing/CompleteTransformerData/{upload_file.name}", 'wb') as f:
            f.write(b)
        if createrequest:
            st.write("Transformer successfully created")
        else:
            st.markdown(f":red[{createrequest.json()['detail']}]")
        
    else:
        st.write("Please input Excel Sheet")

def updatexfmr(xfmr_name,upload_file):
    db = requests.get("http://localhost:8000/transformers/")
    xfmr_list = db.json()
    for i in range(len(xfmr_list)):
        if xfmr_name == xfmr_list[i]["transformer_name"]:
            if upload_file is not None:
                upload_file.name = f"{xfmr_name}.xlsx"
                b = upload_file.getvalue()
                with open(f"../DataProcessing/CompleteTransformerData/{upload_file.name}", 'wb') as f:
                    f.write(b)
                st.write("Excel Sheet successfully updated")
            else:
                st.write("Input Excel Sheet")
        else:
            st.write(f"{xfmr_name} does not exist in DB")


@st.dialog("Are you sure?")
def confirm(name):
    st.write(f"Delete {name}?")
    confirm = st.button("Confirm")
    close = st.button("Close")
    if confirm:
        deleterequest = requests.delete("http://localhost:8000/transformers/"+name)
        if deleterequest:
            st.write("Transformer successfully deleted")
            time.sleep(5)
            st.rerun()
    if close:
        st.rerun()

def refresh_list():
    st.session_state["list"] = requests.get("http://localhost:8000/transformers/").json()




with col1:
    st.header("Create Transformer")
    with st.form("new_xfmr_form", enter_to_submit = False):
        st.write("Input parameters")
        xfmr_name = st.text_input("Transformer Name")
        kva = st.number_input("Power (KVA)")
        rated_voltageHV = st.number_input("Rated Voltage HV")
        rated_currentHV = st.number_input("Rated Current HV")
        rated_voltageLV = st.number_input("Rated Voltage LV")
        rated_currentLV = st.number_input("Rated Current LV")
        rated_thermal_class = st.number_input("Rated Thermal Class")
        rated_avg_winding_temp_rise = st.number_input("Rated Avg Winding Temp Rise")
        winding_material = st.selectbox("Winding Material", ("Aluminum", "Copper"))
        weight_CoreAndCoil_kg = st.number_input("Weight (Core and Coil) in kg")
        weight_Total_kg = st.number_input("Total Weight in kg")
        rated_impedance = st.number_input("Rated Impedance")
        manufacture_date = st.text_input("Manufacture Year")
        new_xfmr_dict = {
            "transformer_name": xfmr_name,
            "kva": kva,
            "rated_voltage_HV": rated_voltageHV,
            "rated_current_HV": rated_currentHV,
            "rated_voltage_LV": rated_voltageLV,
            "rated_current_LV": rated_currentLV,
            "rated_thermal_class": rated_thermal_class,
            "rated_avg_winding_temp_rise": rated_avg_winding_temp_rise,
            "winding_material": winding_material,
            "weight_CoreAndCoil_kg": weight_CoreAndCoil_kg,
            "weight_Total_kg": weight_Total_kg,
            "rated_impedance": rated_impedance,
            "manufacture_date": manufacture_date,
            "status":"new"
            }
        upload_file = st.file_uploader("Choose a file", type ="xlsx")
        
        submit_create = st.form_submit_button("Submit")
        if submit_create:
            createxfmr(new_xfmr_dict,upload_file)


with col2:
    st.header("Delete Transformer")
    xfmr_list = []
    for i in range(len(st.session_state["list"])):
            xfmr_list.append(st.session_state["list"][i]["transformer_name"])
    transformer_select_box = st.selectbox("Choose a Transformer to Delete",xfmr_list)
    submit_delete = st.button("Submit")
    refresh = st.button("Refresh",on_click = refresh_list)
    if submit_delete:
        confirm(transformer_select_box)
    
    st.header("Update Transformer Data")
    with st.form("upd_xfmr_form", enter_to_submit=False):
        st.write("Input new Excel Sheet")
        xfmr_name_to_update = st.selectbox("Choose a Transformer to Update", xfmr_list, key="upd_xfmr_select")
        upload_update_file = st.file_uploader("Choose a file", type="xlsx")
        submit_update = st.form_submit_button("Submit")
        if submit_update:
            updatexfmr(xfmr_name_to_update, upload_update_file)
        

        