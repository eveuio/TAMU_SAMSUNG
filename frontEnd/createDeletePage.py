import streamlit as st
import requests
import time
from datetime import datetime
from pathlib import Path
import os
import json
import pandas

col1, col2 = st.columns(2)


def createxfmr(xfmrdict, upload_file):
    # Validate file upload first
    if upload_file is None:
        st.error("⚠️ Please input Excel Sheet")
        return False
    
    try:
        #TODO: if transformer already exists, do not upload/update excel file, throw message and exit
        existing_transformers = [xfmr["transformer_name"] for xfmr in st.session_state["list"]]
        xfmr_name =xfmrdict["transformer_name"]
        
        if xfmr_name in existing_transformers:
            st.error(f"⚠️ Transformer '{xfmr_name}' already exists in the database. Please use the 'Update Transformer Data' section to upload data for an existing transformer.")
            return False
        
        #TODO: if transfomrer doesnt exist, upload excel sheet and ensure successful upload before adding transformer to database
        file_extension = os.path.splitext(upload_file.name)[1]
        new_filename = f"{xfmr_name}{file_extension}"
        target_path = Path("../DataProcessing/CompleteTransformerData") / new_filename
        
        # Write file
        with open(target_path, 'wb') as f:
            f.write(upload_file.getvalue())
        
        st.success(f"✅ File uploaded successfully as '{new_filename}'")

        #TODO: Write most recent datetime from the excel file to a json in the same directory
        cache_file = target_path.parent / "file_timestamps.json"
        
        # Read the Excel file to get the last timestamp
        df = pandas.read_excel(target_path, sheet_name=0 if file_extension == '.xlsx' else 0)
        
        timestamp_col = df.columns[0]

        if timestamp_col is not None:
            # Get the last timestamp from the file
            last_timestamp = df[timestamp_col].max()
            last_timestamp_str = last_timestamp.strftime("%Y-%m-%d %H:%M:%S")
           
        else:
            st.warning("⚠️ No timestamp column found in Excel file, please include a timestamp column.")
            
            #TODO: need to delete the uploaded file if no datetime column found
            if target_path.exists():
                target_path.unlink()  

            return False
        
        # Load existing cache or create new one
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                timestamp_cache = json.load(f)
        else:
            timestamp_cache = {}
        
        # Update cache with last timestamp from file and upload date
        timestamp_cache[new_filename] = {
            "last_timestamp": last_timestamp_str,
            "last_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save updated cache
        with open(cache_file, 'w') as f:
            json.dump(timestamp_cache, f, indent=2)

        #TODO: Create transformer instance in master table, ensure successful creation
        createrequest = requests.post("http://localhost:8000/transformers/", json=xfmrdict)
        
        # Check if transformer creation was successful
        if createrequest.status_code != 200:
            st.error(f":red[{createrequest.json()['detail']}]")
            return False
        
        st.success("✅ Transformer successfully created in database")
       
        return True
        
    except Exception as e:
        st.error(f"❌ Error creating transformer: {e}")
        return False
    

#!Added functionality to allow for data refresh once new excel file uploaded to DataProcessing/CompleteTransformerData
def refresh_and_update(xfmr_name):
    
    try:
        response = requests.post(
            "http://localhost:8000/update-tables/",
            json={"xfmr_name": xfmr_name},  
            timeout=30
        )
        if response.status_code == 200:
            st.success("Tables updated successfully!")
        else:
            st.error(f"Failed to update tables: {response.text}")

    except Exception as e:
        st.error(f"Error contacting server: {e}")

    refresh_list()
    return

def updatexfmr(xfmr_name,upload_file):
    #TODO: rename excel file to {transformer_name}.xlsx and identify last timestamp for transformer_name in ../DataProcessing/CompleteTransformerData/file_timestamps.json
    
    new_filename = f"{xfmr_name}.xlsx"
    target_path = Path("../DataProcessing/CompleteTransformerData") / new_filename

    cache_file = target_path.parent / "file_timestamps.json"

    with open(cache_file, 'r') as f:
        timestamp_cache = json.load(f)
    
    transformer_key = f"{xfmr_name}.xlsx"
    last_timestamp_json = timestamp_cache[transformer_key]["last_timestamp"]

    #TODO: upload excel file to ../DataProcessing/CompleteTransformerData/file_timestamps.json, check last timestamp to ensure it is greater than the json timestamp. Else, delete uploaded file and throw error
    with open(target_path, 'wb') as f:
        f.write(upload_file.getvalue())
    
    df = pandas.read_excel(target_path)
    datetime_col = df.columns[0]

    if datetime_col:
        df["DATETIME"] = pandas.to_datetime(df["DATETIME"], errors="coerce")
        df = df.dropna(subset=["DATETIME"])
        
        if not df.empty:
            last_timestamp_excel = df["DATETIME"].max().strftime("%Y-%m-%d %H:%M:%S")
            # print(f"Last timestamp from Excel: {last_timestamp_excel}")
    
    if last_timestamp_excel > last_timestamp_json:
        #TODO: call update methods etc
        refresh_and_update(xfmr_name)


    else:
        #TODO: uploaded data not a true update, throw warning message, delete the uploaded file and exit
        st.warning(
                    f"Uploaded file max timestamp:({last_timestamp_excel}) is not newer than existing max timestamp:({last_timestamp_json}). "
                    "Please upload a more recent dataset."
                )
        os.remove(target_path)
        return


# def updatexfmr(xfmr_name,upload_file):
#     db = requests.get("http://localhost:8000/transformers/")
#     xfmr_list = db.json()
    
#     #
#     # for i in range(len(xfmr_list)):
#     #     if xfmr_name == xfmr_list[i]["transformer_name"]:
#     #         if upload_file is not None:
#     #             upload_file.name = f"{xfmr_name}.xlsx"
#     #             b = upload_file.getvalue()
#     #             with open(f"../DataProcessing/CompleteTransformerData/{upload_file.name}", 'wb') as f:
#     #                 f.write(b)
#     #             st.write("Excel Sheet successfully updated")
#     #         else:
#     #             st.write("Input Excel Sheet")
#     #     else:
#     #         st.write(f"{xfmr_name} does not exist in DB")
#     #TODO: Ensure that timestamp of newly uploaded file and stored timestamp are different, and that the newly uploaded file has a more recent timestamp. 
#         # If timestamp is the same or less, prompt saying "The last timestamp of this file is the same or less than the data for {transformer_name} already in the system. please upload a more recent set of data"
    
#     #TODO: 


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
        st.write("Please Input Name and Rated Parameters")
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
        upload_file = st.file_uploader("Upload Transformer Data", type ="xlsx",accept_multiple_files=False)
        
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
        upload_update_file = st.file_uploader("Choose a file", type="xlsx",accept_multiple_files=False)
        submit_update = st.form_submit_button("Submit")
        if submit_update:
            updatexfmr(xfmr_name_to_update, upload_update_file)
        

        