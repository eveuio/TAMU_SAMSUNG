import streamlit as st
import requests
import pandas as pd
import numpy as np
#from fpdf import FPDF
import altair as alt
from datetime import date,datetime,timedelta


#Get individual xfmr data from database
def get_xfmr_data(id):
    response = requests.get("http://localhost:8000/transformers/"+str(id))
    if response.json() == []:
        st.session_state["read_error"] = True
    else:
        st.session_state["read_error"] = False
        st.session_state["data"] = response.json()
#Refresh list of transformers
def refresh_list():
    st.session_state["list"] = requests.get("http://localhost:8000/transformers/").json()
def get_xfmr_status(id):
    response = requests.get("http://localhost:8000/transformers/status/"+str(id))
    xfmr_status_data = response.json()
    return xfmr_status_data
    
_="""
@st.cache_resource(ttl="1d")
def create_pdf(xfmr_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', 'B', 12)
    pdf.cell(160,10,"Transformer Report",align = "l")
    pdf.set_font('Times', '', 12)
    pdf.cell(0,10,"Date:" + str(date.today()),align = "r",ln = 1)
    pdf.cell(28.2,10,"Name",border = 1)
    pdf.cell(28.2,10,"Overall Status",border = 1)
    pdf.cell(28.2,10,"Avg Current", border = 1)
    pdf.cell(28.2,10,"Avg Voltage",border = 1)
    pdf.cell(28.2,10,"Ambient Temp", border = 1)
    pdf.cell(28.2,10,"Winding Temp", border = 1)
    pdf.cell(28.2,10,"Power Factor", ln = 1, border = 1)
    for i in range(len(xfmr_list)):
        data = requests.get("http://localhost:8000/transformers/"+str(xfmr_list[i]["transformer_name"]))
        status_data = requests.get("http://localhost:8000/transformers/status/"+str(xfmr_list[i]["transformer_name"]]))
        dataJSON = data.json()
        avgCurrent = round(int(dataJSON["avgCurrentA"]) + int(dataJSON["avgCurrentB"]) + int(dataJSON["avgCurrentC"])/3,2)
        avgVoltage = round(int(dataJSON["avgVoltageA"]) + int(dataJSON["avgVoltageB"]) + int(dataJSON["avgVoltageC"])/3,2)
        pdf.cell(28.2, 10, dataJSON["name"],border = 1) 
        pdf.cell(28.2, 10,dataJSON["overallStatus"],border= 1)
        pdf.cell(28.2, 10,str(avgCurrent),border = 1)
        pdf.cell(28.2,10, str(avgVoltage), border = 1)
        pdf.cell(28.2,10,  str(dataJSON["avgAmbientTemp"]),border = 1)
        pdf.cell(28.2,10, str(dataJSON["avgWindingTemp"]),border =1)
        pdf.cell(40,10, str(dataJSON["avgPF"]),ln = 1, border = 1)


    pdf.output('xfmr_report.pdf', 'F')
"""

#divide page into two equally sized columns
if "read_error" not in st.session_state:
    st.session_state["read_error"] = False
st.set_page_config(layout = "wide")
col1,col2 = st.columns(2)

#store emoji
bad = "🔴"
ok = "🟡"
good = "🟢"
#get xfmr json
xfmr_list =[]

xfmr_json = st.session_state["list"]

#populate lists

for i in range(len(xfmr_json)): 
    xfmr_list.append(xfmr_json[i]["transformer_name"])

xfmr_list.sort() #sorts list (since status emoji is in string, it automatically sorts red then yellow then red)

#change selected transformer using sidebar
transformer_select = st.sidebar.selectbox("Choose a Transformer", xfmr_list)
st.session_state["xfmr_select"] = transformer_select
for i in range(len(xfmr_json)):
    if st.session_state["xfmr_select"]==xfmr_json[i]["transformer_name"]:
        st.session_state["id"] = xfmr_json[i]["transformer_name"]

get_xfmr_data(st.session_state["id"])
xfmr_status_dict = get_xfmr_status(st.session_state["id"])

#refresh data and list
refresh_list_button = st.sidebar.button("Refresh List", on_click = refresh_list)

_="""
with open("xfmr_report.pdf", "rb") as f:
    create_pdf(xfmr_json)
    PDFByte = f.read()
    st.sidebar.download_button("Download Report",data = PDFByte, file_name = "xfmr_report.pdf")
"""

#fill current dataframe for chart
secondaryCurrent = {"Phase":[],"current":[],"DateTime":[]}
for i in range(len(st.session_state["data"])):
    for Phase in ("avg_secondary_current_a_phase","avg_secondary_current_b_phase","avg_secondary_current_c_phase", "avg_secondary_current_total_phase"):
        if Phase[-7:] in ("a_phase", "b_phase", "c_phase"):
            secondaryCurrent["Phase"].append(Phase[-7:])
        else:
            secondaryCurrent["Phase"].append(Phase[-11:])
        secondaryCurrent["current"].append(st.session_state["data"][i][Phase])
        secondaryCurrent["DateTime"].append(st.session_state["data"][i]["DATETIME"])
secondaryCurrent = pd.DataFrame(secondaryCurrent)
secondaryCurrent["DateTime"] = pd.to_datetime(secondaryCurrent["DateTime"])

#fill secondary voltage dataframe for chart
secondaryVoltage = {"Phase":[],"voltage":[],"DateTime":[]}
for i in range(len(st.session_state["data"])):
    for Phase in ("avg_secondary_current_a_phase","avg_secondary_current_b_phase","avg_secondary_current_c_phase"):
        secondaryVoltage["Phase"].append(Phase[-7:])
        secondaryVoltage["voltage"].append(st.session_state["data"][i][Phase])
        secondaryVoltage["DateTime"].append(st.session_state["data"][i]["DATETIME"])
secondaryVoltage = pd.DataFrame(secondaryVoltage)
secondaryVoltage["DateTime"] = pd.to_datetime(secondaryVoltage["DateTime"])


#fill powerfactor dataframe for chart
powerFactor = {"Power Factor": [], "DateTime": []}
for i in range(len(st.session_state["data"])):
    powerFactor["Power Factor"].append(st.session_state["data"][i]["avg_power_factor"])
    if powerFactor["Power Factor"][i] == None:
        powerFactor["Power Factor"][i] = 0
    powerFactor["DateTime"].append(st.session_state["data"][i]["DATETIME"])
powerFactor = pd.DataFrame(powerFactor)
powerFactor["DateTime"] = pd.to_datetime(powerFactor["DateTime"])

#fill temperature dataframe for chart
temperature = {"Type":[],"temp":[],"DateTime":[]}
for i in range(len(st.session_state["data"])):
    for Type in ("avg_winding_temp_a_phase", "avg_winding_temp_b_phase", "avg_winding_temp_c_phase"):
        temperature["Type"].append(Type[-7:])
        temperature["temp"].append(st.session_state["data"][i][Type])
        temperature["DateTime"].append(st.session_state["data"][i]["DATETIME"])
temperature = pd.DataFrame(temperature)
temperature["DateTime"] = pd.to_datetime(temperature["DateTime"])



#fill lifetime chart data
_ = """
lifetimeChart = {
    "Lifetime": [i["lifetime"]for i in st.session_state["data"]["lifetime"]],
    "Time": [i["time"]for i in st.session_state["data"]["lifetime"]]
    }

end_date = lifetimeChart["Time"][-1]
lifetimeChart["Time"] = pd.to_datetime(lifetimeChart["Time"])
"""


#fill datatable

df = {"Parameter":["Secondary Voltage A-Phase","Secondary Voltage B-Phase","Secondary Voltage C-Phase","Secondary Current A-Phase","Secondary Current B-Phase", "Secondary Current C-Phase","Winding Temp A-Phase","Winding Temp B-Phase","Winding Temp C-Phase"],
"Average":[round(float(xfmr_status_dict[0]["average_value"]),2),round(float(xfmr_status_dict[1]["average_value"]),2),round(float(xfmr_status_dict[2]["average_value"]),2),round(float(xfmr_status_dict[3]["average_value"]),2),round(float(xfmr_status_dict[4]["average_value"]),2),round(float(xfmr_status_dict[5]["average_value"]),2),round(float(xfmr_status_dict[6]["average_value"]),2),round(float(xfmr_status_dict[7]["average_value"]),2),round(float(xfmr_status_dict[8]["average_value"]),2)],
"Rated":[round(xfmr_status_dict[0]["rated_value"],2),round(xfmr_status_dict[1]["rated_value"],2),round(xfmr_status_dict[2]["rated_value"],2),round(xfmr_status_dict[3]["rated_value"],2),round(xfmr_status_dict[4]["rated_value"],2),round(xfmr_status_dict[5]["rated_value"],2),round(xfmr_status_dict[6]["rated_value"],2),round(xfmr_status_dict[7]["rated_value"],2),round(xfmr_status_dict[8]["rated_value"],2)],
"Status":[xfmr_status_dict[0]["status"],xfmr_status_dict[1]["status"],xfmr_status_dict[2]["status"],xfmr_status_dict[3]["status"],xfmr_status_dict[4]["status"],xfmr_status_dict[5]["status"],xfmr_status_dict[6]["status"],xfmr_status_dict[7]["status"],xfmr_status_dict[8]["status"]]}

for i in range(len(df["Status"])):
    if df["Status"][i] == "Green":
        df["Status"][i] =good
    elif df["Status"][i] == "Yellow":
        df["Status"][i] = ok
    elif df["Status"][i] == "Red":
        df["Status"][i] = bad
#display data
with col1:
    #display current xfmr

    if xfmr_status_dict[0]["overall_color"] == "Green":
        overall_status = good
    elif xfmr_status_dict[0]["overall_color"] == "Yellow":
        overall_status = ok
    elif xfmr_status_dict[0]["overall_color"] == "Red":
        overall_status = bad

    st.header(overall_status +" "+st.session_state["xfmr_select"])
    #datatable
    st.dataframe(df,hide_index = 1,width = "stretch")

    #secondary current chart
    st.header("Secondary Current")

    #filter current chart data (all, 1 year, 3 months, 1 month)
    scFilterChoice = st.segmented_control("Filter data:",["All","Past Year", "Past 3 Months", "Past Month","Select Dates"],selection_mode = "single",default = "Past Year", key = "scDateFilter")
    if scFilterChoice == "All":
        secondaryCurrentFiltered = secondaryCurrent
    elif scFilterChoice in (None,"Past Year"):
        secondaryCurrentFiltered = pd.DataFrame({"Phase":secondaryCurrent["Phase"][-1095:],"current":secondaryCurrent["current"][-1095:],"DateTime":secondaryCurrent["DateTime"][-1095:]})
    elif scFilterChoice =="Past 3 Months":
        secondaryCurrentFiltered = pd.DataFrame({"Phase":secondaryCurrent["Phase"][-270:],"current":secondaryCurrent["current"][-270:],"DateTime":secondaryCurrent["DateTime"][-270:]})
    elif scFilterChoice =="Past Month":
        secondaryCurrentFiltered = pd.DataFrame({"Phase":secondaryCurrent["Phase"][-90:],"current":secondaryCurrent["current"][-90:],"DateTime":secondaryCurrent["DateTime"][-90:]})
    elif scFilterChoice == "Select Dates":
        max_date = secondaryCurrent["DateTime"].iat[-1]
        min_date = secondaryCurrent["DateTime"].iat[0]
        date_start = st.date_input("Select starting date:", value = min_date, max_value = max_date, min_value = min_date, key = "scDateStart")
        date_end = st.date_input("Select ending date:", value = max_date, max_value = max_date, min_value = min_date, key = "scDateEnd")
        for i in range(len(secondaryCurrent["DateTime"])):
            if date_start == secondaryCurrent["DateTime"][i].date():
                index_start = i-3
        for i in range(len(secondaryCurrent["DateTime"])):
            if date_end == secondaryCurrent["DateTime"][i].date():
                index_end = i+1
        secondaryCurrentFiltered = pd.DataFrame({"Phase":secondaryCurrent["Phase"][index_start:index_end],"current":secondaryCurrent["current"][index_start:index_end],"DateTime":secondaryCurrent["DateTime"][index_start:index_end]})
    #sc chart
    singleSelect = alt.selection_point(fields = ["DateTime"], on = "mouseover", nearest = True,empty = "none")
    sc_chart_base = alt.Chart(secondaryCurrentFiltered).mark_line().encode(
        alt.X("DateTime:T").title("Date"),
        alt.Y("current:Q").scale(domain=(min(secondaryCurrentFiltered["current"])-100,max(secondaryCurrentFiltered["current"])+100)).title("Current(A)"),
        color = "Phase:N"
    )
    sc_points = sc_chart_base.transform_filter(singleSelect).mark_circle(size = 65)
    sc_tooltips = alt.Chart(secondaryCurrentFiltered).mark_rule().encode(
        x = "DateTime",
        y = "current",
        opacity = alt.condition(singleSelect,alt.value(0.3),alt.value(0)),
        tooltip=[
            alt.Tooltip("DateTime",title="Date"),
            alt.Tooltip("current",title = "Current"),
            alt.Tooltip("Phase",title = "Phase")
        ]
    ).add_params(singleSelect)
    sc_chart = sc_chart_base + sc_tooltips + sc_points
    st.altair_chart(sc_chart)

    #power factor chart
    st.header("Power Factor")
    pfFilterChoice = st.segmented_control("Filter data:",["All","Past Year", "Past 3 Months", "Past Month","Select Dates"],selection_mode = "single",default = "Past Year", key = "pfFilter")
    if pfFilterChoice == "All":
        powerFactorFiltered = powerFactor
    elif pfFilterChoice in (None,"Past Year"):
        powerFactorFiltered = pd.DataFrame({"Power Factor":powerFactor["Power Factor"][-1095:],"DateTime":powerFactor["DateTime"][-1095:]})
    elif pfFilterChoice =="Past 3 Months":
        powerFactorFiltered = pd.DataFrame({"Power Factor":powerFactor["Power Factor"][-270:],"DateTime":powerFactor["DateTime"][-270:]})
    elif pfFilterChoice =="Past Month":
        powerFactorFiltered = pd.DataFrame({"Power Factor":powerFactor["Power Factor"][-90:],"DateTime":powerFactor["DateTime"][-90:]})
    elif pfFilterChoice == "Select Dates":
        max_date = powerFactor["DateTime"].iat[-1]
        min_date = powerFactor["DateTime"].iat[0]
        date_start = st.date_input("Select starting date:", value = min_date, max_value = max_date, min_value = min_date, key = "pfDateStart")
        date_end = st.date_input("Select ending date:", value = max_date, max_value = max_date, min_value = min_date, key = "pfDateEnd")
        for i in range(len(powerFactor["DateTime"])):
            if date_start == powerFactor["DateTime"][i].date():
                index_start = i
        for i in range(len(powerFactor["DateTime"])):
            if date_end == powerFactor["DateTime"][i].date():
                index_end = i+1
        powerFactorFiltered = pd.DataFrame({"Power Factor":powerFactor["Power Factor"][index_start:index_end],"DateTime":powerFactor["DateTime"][index_start:index_end]})
    pf_chart_base = alt.Chart(powerFactorFiltered).mark_line().encode(
        alt.X("DateTime:T").title("Date"),
        alt.Y("Power Factor:Q").scale(domain=(0,1)).title("PF(%)"),
    )
    pf_points = pf_chart_base.transform_filter(singleSelect).mark_circle(size = 65)
    pf_tooltips = alt.Chart(powerFactorFiltered).mark_rule().encode(
        x = "DateTime",
        y = "Power Factor",
        opacity = alt.condition(singleSelect,alt.value(0.3),alt.value(0)),
        tooltip=[
            alt.Tooltip("DateTime",title="Date"),
            alt.Tooltip("Power Factor",title = "Power Factor")
        ]
    ).add_params(singleSelect)
    pf_chart = pf_chart_base + pf_tooltips + pf_points
    st.altair_chart(pf_chart)








with col2:
    _ = """
    st.header("Lifetime Forecast")
    projected = "Projected End Date: " + end_date
    st.line_chart(lifetimeChart, x = "Time", y = "Lifetime", x_label = projected, y_label = "Lifetime (%)")
"""
    #secondary voltage chart
    st.header("Secondary Voltage")

    #secondary voltage filter
    svFilterChoice = st.segmented_control("Filter data:",["All","Past Year", "Past 3 Months", "Past Month","Select Dates"],selection_mode = "single",default = "Past Year", key = "svFilter")
    if svFilterChoice == "All":
        secondaryVoltageFiltered = secondaryVoltage
    elif svFilterChoice in (None,"Past Year"):
        secondaryVoltageFiltered = pd.DataFrame({"Phase":secondaryVoltage["Phase"][-1095:],"voltage":secondaryVoltage["voltage"][-1095:],"DateTime":secondaryVoltage["DateTime"][-1095:]})
    elif svFilterChoice =="Past 3 Months":
        secondaryVoltageFiltered = pd.DataFrame({"Phase":secondaryVoltage["Phase"][-270:],"voltage":secondaryVoltage["voltage"][-270:],"DateTime":secondaryVoltage["DateTime"][-270:]})
    elif svFilterChoice =="Past Month":
        secondaryVoltageFiltered = pd.DataFrame({"Phase":secondaryVoltage["Phase"][-90:],"voltage":secondaryVoltage["voltage"][-90:],"DateTime":secondaryVoltage["DateTime"][-90:]})
    elif svFilterChoice == "Select Dates":
        max_date = secondaryVoltage["DateTime"].iat[-1]
        min_date = secondaryVoltage["DateTime"].iat[0]
        date_start = st.date_input("Select starting date:", value = min_date, max_value = max_date, min_value = min_date, key = "svDateStart")
        date_end = st.date_input("Select ending date:", value = max_date, max_value = max_date, min_value = min_date, key = "svDateEnd")
        for i in range(len(secondaryVoltage["DateTime"])):
            if date_start == secondaryVoltage["DateTime"][i].date():
                index_start = i-2
        for i in range(len(secondaryVoltage["DateTime"])):
            if date_end == secondaryVoltage["DateTime"][i].date():
                index_end = i+1
        secondaryVoltageFiltered = pd.DataFrame({"Phase":secondaryVoltage["Phase"][index_start:index_end],"voltage":secondaryVoltage["voltage"][index_start:index_end],"DateTime":secondaryVoltage["DateTime"][index_start:index_end]})

    singleSelect = alt.selection_point(fields = ["DateTime"], on = "mouseover", nearest = True,empty = "none")
    sv_chart_base = alt.Chart(secondaryVoltageFiltered).mark_line().encode(
        alt.X("DateTime:T").title("Date"),
        alt.Y("voltage:Q").scale(domain=(min(secondaryVoltageFiltered["voltage"])-5,max(secondaryVoltageFiltered["voltage"])+5)).title("Voltage(V)"),
        color = "Phase:N"
    )
    sv_points = sv_chart_base.transform_filter(singleSelect).mark_circle(size = 65)
    sv_tooltips = alt.Chart(secondaryVoltageFiltered).mark_rule().encode(
        x = "DateTime",
        y = "voltage",
        opacity = alt.condition(singleSelect,alt.value(0.3),alt.value(0)),
        tooltip=[
            alt.Tooltip("DateTime",title="Date"),
            alt.Tooltip("voltage",title = "Voltage"),
            alt.Tooltip("Phase",title = "Phase")
        ]
    ).add_params(singleSelect)
    sv_chart = sv_chart_base + sv_tooltips + sv_points
    st.altair_chart(sv_chart)

    #temperature chart
    st.header("Temperature")

    tempFilterChoice = st.segmented_control("Filter data:",["All","Past Year", "Past 3 Months", "Past Month","Select Dates"],selection_mode = "single",default = "Past Year", key = "tempFilter")
    if tempFilterChoice == "All":
        temperatureFiltered = temperature
    elif tempFilterChoice in (None,"Past Year"):
        temperatureFiltered = pd.DataFrame({"temp":temperature["temp"][-1095:],"Type":temperature["Type"][-1095:],"DateTime":temperature["DateTime"][-1095:]})
    elif tempFilterChoice =="Past 3 Months":
        temperatureFiltered = pd.DataFrame({"temp":temperature["temp"][-270:],"Type":temperature["Type"][-270:],"DateTime":temperature["DateTime"][-270:]})
    elif tempFilterChoice =="Past Month":
        temperatureFiltered = pd.DataFrame({"temp":temperature["temp"][-90:],"Type":temperature["Type"][-90:],"DateTime":temperature["DateTime"][-90:]})
    elif tempFilterChoice == "Select Dates":
        max_date = temperature["DateTime"].iat[-1]
        min_date = temperature["DateTime"].iat[0]
        date_start = st.date_input("Select starting date:", value = min_date, max_value = max_date, min_value = min_date, key = "tempDateStart")
        date_end = st.date_input("Select ending date:", value = max_date, max_value = max_date, min_value = min_date, key = "tempDateEnd")
        for i in range(len(temperature["DateTime"])):
            if date_start == temperature["DateTime"][i].date():
                index_start = i-1
        for i in range(len(temperature["DateTime"])):
            if date_end == temperature["DateTime"][i].date():
                index_end = i+1
        temperatureFiltered = pd.DataFrame({"temp":temperature["temp"][index_start:index_end],"Type":temperature["Type"][index_start:index_end],"DateTime":temperature["DateTime"][index_start:index_end]})

    singleSelect = alt.selection_point(fields = ["DateTime"], on = "mouseover", nearest = True,empty = "none")
    temp_chart_base = alt.Chart(temperatureFiltered).mark_line().encode(
        alt.X("DateTime:T").title("Date"),
        alt.Y("temp:Q").scale(domain=(min(temperatureFiltered["temp"])-10,max(temperatureFiltered["temp"])+10)).title("Temperature(C)"),
        color = "Type:N"
    )
    temp_points = temp_chart_base.transform_filter(singleSelect).mark_circle(size = 65)
    temp_tooltips = alt.Chart(temperatureFiltered).mark_rule().encode(
        x = "DateTime",
        y = "temp",
        opacity = alt.condition(singleSelect,alt.value(0.3),alt.value(0)),
        tooltip=[
            alt.Tooltip("DateTime",title="Date"),
            alt.Tooltip("temp",title = "Temperature"),
            alt.Tooltip("Type",title = "Type")
        ]
    ).add_params(singleSelect)
    temp_chart = temp_chart_base + temp_tooltips + temp_points
    st.altair_chart(temp_chart)
    
