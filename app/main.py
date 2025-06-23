from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
from app.model_utils import train_model, predict_lead

app = FastAPI()

class LeadInput(BaseModel):
    CREATEDBY: str
    CREATED_ON: date
    CITY: str
    DISTRICT_NAME: str
    STATE: str
    LEAD_TYPE: str
    LEAD_SOURCE: str
    LEAD_SOURCE_TYPE: str
    LEAD_STAGE: str
    EXPECTED_QTY: int
    NO_OF_CUSTOMER_VISITS: int
    TOTAL_TIME_SPENT_AT_CUSTOMER: str
    FULLFILLED_BY_DEALER_NAME: str
    NO_OF_DEALER_VISITS: int
    TOTAL_TIME_SPENT_AT_DEALER: str
    TOTAL_SALE_IN_MT: int

@app.post("/predict")
def predict_endpoint(data: LeadInput):
    d = data.dict()
    d["TOTAL_TIME_SPENT_AT_CUSTOMER"] = convert_to_minutes(d["TOTAL_TIME_SPENT_AT_CUSTOMER"])
    d["TOTAL_TIME_SPENT_AT_DEALER"] = convert_to_minutes(d["TOTAL_TIME_SPENT_AT_DEALER"])
    prob = predict_lead(d)
    return {"conversion_probability": prob}
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from fastapi import BackgroundTasks

@app.post("/add_data_and_train")
def add_data(data: LeadInput, background_tasks: BackgroundTasks):
    df = pd.read_csv("app/Lead_Data.csv")
    new_row = pd.DataFrame([data.dict()])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv("app/Lead_Data.csv", index=False)

    background_tasks.add_task(train_model)  # run in background
    return {"message": "✅ Data added. Training started in background."}

def convert_to_minutes(time_str: str):
    try:
        h, m = 0, 0
        if 'h' in time_str:
            parts = time_str.split('h')
            h = int(parts[0].strip())
            if 'm' in parts[1]:
                m = int(parts[1].replace('m', '').strip())
        elif 'm' in time_str:
            m = int(time_str.replace('m', '').strip())
        return h * 60 + m
    except:
        return 0
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from llama_summary import summarize_with_llama

router = APIRouter()

class LogEntry(BaseModel):
    date: str
    created_by: str
    location: str
    lead_name: str
    notes: str
    distance_km: float
    status_update: str

@app.post("/summarize")
def summarize_logs(logs: List[LogEntry]):
    notes = [log.notes for log in logs]
    stats = {
        "Total Visits": len(logs),
        "Total Distance": sum(log.distance_km for log in logs),
        "Leads Met": len(set(log.lead_name for log in logs))
    }

    combined_notes = "\n".join(notes)
    summary = summarize_with_llama(combined_notes, stats)
    return {"summary": summary}
