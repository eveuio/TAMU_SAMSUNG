import streamlit as st
import requests
import time
from datetime import datetime
from pathlib import Path
import os
import json
import pandas
from errorPage import retry

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
        
        #TODO: check if rated values are a null or zero, transformer name is an empty string and manufacture year (when typecast to an int) is a 4 digit number
        if xfmr_name == "":
            #throw error/exception saying 'transformer name cannot be empty
            st.error("Transformer name cannot be empty.")
            return False


        # Collect all rated parameters into a list
        rated_values = [
            xfmrdict["kva"],
            xfmrdict["rated_voltage_HV"],
            xfmrdict["rated_current_HV"],
            xfmrdict["rated_voltage_LV"],
            xfmrdict["rated_current_LV"],
            xfmrdict["rated_thermal_class"],
            xfmrdict["rated_avg_winding_temp_rise"],
            xfmrdict["weight_CoreAndCoil_kg"],
            xfmrdict["weight_Total_kg"],
            xfmrdict["rated_impedance"]
        ]

        # Check if any value is None or <= 0
        if any(val is None or val <= 0 for val in rated_values):
            st.error("Transformer Rated Parameters must be positive and non-zero.")
            return False

        
        if (str(xfmrdict["manufacture_date"]).isdigit() == False or len(str(xfmrdict["manufacture_date"])) != 4):
            #throw error saying 'transformer manufacture date is invalid. please enter a valid year'
            st.error("Transformer Manufacture Date must be a valid calendar year (i.e. 2006)")
            return False
        
        if xfmr_name in existing_transformers:
            st.error(f"Transformer '{xfmr_name}' already exists in the database. Please use the 'Update Transformer Data' section to upload data for an existing transformer.")
            return False
        
        #TODO: if transfomrer doesnt exist, upload excel sheet and ensure successful upload before adding transformer to database
        file_extension = os.path.splitext(upload_file.name)[1]
        new_filename = f"{xfmr_name}{file_extension}"
        target_path = Path("../DataProcessing/CompleteTransformerData") / new_filename
        
        # Write file
        with open(target_path, 'wb') as f:
            f.write(upload_file.getvalue())

        #TODO: Check to make sure there are at least 10 columns in excel sheet
        df = pandas.read_excel(target_path, sheet_name=0 if file_extension == '.xlsx' else 0)

        if df.shape[1] < 10:
            st.error(f"Upload failed: Excel sheet must have at least 10 columns. Found {df.shape[1]} columns.")
            if target_path.exists():
                target_path.unlink()  
            return False  # Stop execution
        
        #TODO: Ensure a timestamp or datetime column exists and is the correct format/type
       
        timestamp_col = df.columns[0]

        # Validate timestamp format before conversion
        # Convert to string for validation
        timestamp_values = df[timestamp_col].astype(str)

        # Check: Should contain datetime separators (-, /, :, or space)
        no_separators = ~timestamp_values.str.contains(r'[-/:\s]', regex=True, na=False)
        invalid_count = no_separators.sum()

        if invalid_count > 0:
            st.error(f"❌ Upload failed: Found {invalid_count} rows with invalid timestamp format.")
            st.error(f"Expected datetime formats like: 'YYYY-MM-DD HH:MM:SS' or 'MM/DD/YYYY HH:MM:SS'")
            # Show examples of invalid values
            invalid_examples = df.loc[no_separators, timestamp_col].head(3).tolist()
            st.error(f"Invalid examples: {invalid_examples}")
            if target_path.exists():
                target_path.unlink()
            return False

        # Convert timestamp column to datetime
        df['DATETIME'] = pandas.to_datetime(df[timestamp_col], errors='coerce')

        # Remove rows where DATETIME is NaT (null)
        valid_data = df[df['DATETIME'].notna()].copy()

        if len(valid_data) == 0:
            st.error(f"❌ Upload failed: No valid datetime entries found.")
            if target_path.exists():
                target_path.unlink()  
            return False

        #TODO: Collect last timestamp for json file write down below
        # Get the last timestamp from the file
        last_timestamp = valid_data['DATETIME'].max()
        last_timestamp_str = last_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        #TODO: Check to make sure that there is at least 3 days worth of data in the first 10 columns by finding the first non-zero row and counting 3 days after that
        # Get first 10 columns (datetime plus winding temp (x3), current (x3) and voltage (x3))
        numeric_part = valid_data.iloc[:, 0:10]

        # Replace NaN with 0 just to be safe
        numeric_part = numeric_part.fillna(0)

        # Boolean mask: True if all first 10 columns are non-zero
        mask = (numeric_part != 0).all(axis=1)

        # Get the index of the first row where all 10 columns are non-zero
        if not mask.any():
            st.error(f"❌ Upload failed: No complete non-zero data rows found in the first 10 columns.")
            if target_path.exists():
                target_path.unlink()
            return False

        start_index = mask.idxmax()  # gives first True index
        data_from_first_complete = valid_data.loc[start_index:].reset_index(drop=True)

        # Get the date range from first complete row
        first_date = data_from_first_complete['DATETIME'].iloc[0]
        last_date = data_from_first_complete['DATETIME'].iloc[-1]
        date_range = (last_date - first_date).total_seconds() / (24 * 3600)  # Convert to days

        # For 10-minute intervals: 3 days = 432 data points (6 per hour * 24 hours * 3 days)
        min_rows_for_3_days = 432
        actual_rows = len(data_from_first_complete)

        if date_range < 3:
            st.error(f"❌ Upload failed: Data must span at least 3 days. Found {date_range:.2f} days of valid data.")
            if target_path.exists():
                target_path.unlink()  
            return False

        if actual_rows < min_rows_for_3_days:
            st.error(f"❌ Upload Failed: Expected {min_rows_for_3_days} rows for 3 days of 10-minute data. Found {actual_rows} rows.")
            if target_path.exists():
                target_path.unlink()
            return False

        st.success(f"✅ Validation passed: {actual_rows} complete rows spanning {date_range:.2f} days")
        
        #TODO: Write most recent datetime from the excel file to a json in the same directory
        cache_file = target_path.parent / "file_timestamps.json"
        
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

        st.success(f"✅ File uploaded successfully as '{new_filename}'")
        #TODO: Create transformer instance in master table, ensure successful creation
        createrequest = requests.post("http://localhost:8000/transformers/", json=xfmrdict)
        
        # Check if transformer creation was successful
        if createrequest.status_code != 200:
            st.error(f":red[{createrequest.json()['detail']}]")
            return False
        
        st.success("✅ Transformer successfully created in database")
        st.session_state["list"] = requests.get("http://localhost:8000/transformers/").json()
        retry()

       
        return True
        
    except Exception as e:
        target_path.unlink()  # Delete file if DB insert fails
        st.error(f"❌ Error creating transformer: {e}")
        return False
    

#!Added functionality to allow for data refresh once new excel file uploaded to DataProcessing/CompleteTransformerData
def refresh_and_update(xfmr_name,last_timestamp_excel):
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
        
        #TODO: Update json timestamp:
        new_filename = f"{xfmr_name}.xlsx"
        target_path = Path("../DataProcessing/CompleteTransformerData") / new_filename
        cache_file = target_path.parent / "file_timestamps.json"

        with open(cache_file, 'r') as f:
            timestamp_cache = json.load(f)  

        timestamp_cache[new_filename] = {
            "last_timestamp": last_timestamp_excel,
            "last_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save updated cache
        with open(cache_file, 'w') as f:
            json.dump(timestamp_cache, f, indent=2)


    except Exception as e:
        st.error(f"Error contacting server: {e}")

    refresh_list()

    return

def updatexfmr(xfmr_name,upload_file):
    #TODO: rename excel file to {transformer_name}.xlsx and identify last timestamp for transformer_name in ../DataProcessing/CompleteTransformerData/file_timestamps.json
    if upload_file is None:
        st.error("⚠️ Please input Excel Sheet")
        return False
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
        refresh_and_update(xfmr_name,last_timestamp_excel)


    else:
        #TODO: uploaded data not a true update, throw warning message, delete the uploaded file and exit
        st.warning(
                    f"Uploaded file max timestamp:({last_timestamp_excel}) is not newer than existing max timestamp:({last_timestamp_json}). "
                    "Please upload a more recent dataset."
                )
        os.remove(target_path)
        return False
    retry()


@st.dialog("Are you sure?")
def confirm(name):
    if name != None:
        st.write(f"Delete {name}?")
        confirm = st.button("Confirm")
        close = st.button("Close")
        if confirm:
            try:
                deleterequest = requests.delete("http://localhost:8000/transformers/"+name)
                if deleterequest:
                    st.write("Transformer successfully deleted")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.write("No transformer to delete")
        close = st.button("Close")

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
    if xfmr_list == []:
        st.write("No transformer to delete")
    else:
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
        

        