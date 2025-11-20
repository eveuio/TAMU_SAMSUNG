from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

import datetime
import os
import sys

# -------------------------------------------------------------------
# Path / imports so we can see the shared code
# -------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from DataProcessing.programFiles import Database
from machinelearning.transformer_health_monitor import TransformerHealthMonitor
from machinelearning.forecast_engine import TransformerForecastEngine

# -------------------------------------------------------------------
# Database config - this MUST point at the DB you inspected with sqlite3
# -------------------------------------------------------------------
# You confirmed this file has transformers, HealthScores, ForecastData, etc.
DATABASE_URL = (
    "sqlite:////home/alexbrent25/Capstone-alex/src/alex/TAMU_SAMSUNG/transformerDB.db"
)
DB_FILE_PATH = (
    "/home/alexbrent25/Capstone-alex/src/alex/TAMU_SAMSUNG/transformerDB.db"
)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# -------------------------------------------------------------------
# Helper: reflect any table by name (for *_average_metrics_day, etc.)
# -------------------------------------------------------------------
def get_table_by_name(table_name: str):
    metadata = MetaData()
    metadata.reflect(bind=engine)

    if table_name in metadata.tables:
        return Table(table_name, metadata, autoload_with=engine)
    return None


# -------------------------------------------------------------------
# ORM models for the tables you care about in FastAPI
# -------------------------------------------------------------------
class Transformers(Base):
    __tablename__ = "transformers"

    id = Column(Integer, primary_key=True)
    transformer_name = Column(String, nullable=False)
    kva = Column(Float)
    rated_voltage_HV = Column(Float)
    rated_current_HV = Column(Float)
    rated_voltage_LV = Column(Float)
    rated_current_LV = Column(Float)
    rated_thermal_class = Column(Float)
    rated_avg_winding_temp_rise = Column(Float)
    winding_material = Column(String)
    weight_CoreAndCoil_kg = Column(Float)
    weight_Total_kg = Column(Float)
    rated_impedance = Column(Float)
    manufacture_date = Column(String)
    status = Column(String)


class Transformer(BaseModel):
    transformer_name: str
    kva: float
    rated_voltage_HV: float
    rated_current_HV: float
    rated_voltage_LV: float
    rated_current_LV: float
    rated_thermal_class: float
    rated_avg_winding_temp_rise: float
    winding_material: str
    weight_CoreAndCoil_kg: float
    weight_Total_kg: float
    rated_impedance: float
    manufacture_date: str
    status: str


class HealthScores(Base):
    __tablename__ = "HealthScores"

    id = Column(Integer, primary_key=True)
    transformer_name = Column(String)
    date = Column(String)  # stored as text (YYYY-MM-DD)
    variable_name = Column(String)
    average_value = Column(Float)
    rated_value = Column(Float)
    status = Column(String)
    overall_score = Column(Float)
    overall_color = Column(String)


# -------------------------------------------------------------------
# Shared low-level database object used by health + forecast engines
#   ⚠️ This must expose .conn and .cursor, which your DataProcessing.programFiles.Database does.
# -------------------------------------------------------------------
database = Database(
    db_path=DB_FILE_PATH,
    session_factory=SessionLocal,
    orm_transformers=Transformers,
    engine=engine,
)

health_monitor = TransformerHealthMonitor(database=database)
forecast_engine = TransformerForecastEngine(database)


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI()


# -------------------------------------------------------------------
# GET /transformers/  -> list of all transformers
# -------------------------------------------------------------------
@app.get("/transformers/")
async def read_xfmrs():
    table = get_table_by_name("transformers")
    if table is None:
        raise HTTPException(status_code=404, detail="Table not found")

    with SessionLocal() as db:
        results = db.query(table).all()
        return [row._asdict() for row in results]


# -------------------------------------------------------------------
# GET /transformers/{xfmr_name} -> *_average_metrics_day for plotting
# -------------------------------------------------------------------
@app.get("/transformers/{xfmr_name}")
async def read_xfmr(xfmr_name: str):
    table = get_table_by_name(f"{xfmr_name}_average_metrics_day")
    if table is None:
        raise HTTPException(status_code=404, detail="Transformer not found")

    with SessionLocal() as db:
        results = db.query(table).all()
        return [row._asdict() for row in results]


# -------------------------------------------------------------------
# GET /transformers/status/{xfmr_name}
#   -> latest HealthScores rows for that transformer
# -------------------------------------------------------------------
@app.get("/transformers/status/{xfmr_name}")
def read_xfmr_status(xfmr_name: str):
    with SessionLocal() as db:
        # Get most recent date for this xfmr
        latest_row = (
            db.query(HealthScores)
            .filter_by(transformer_name=xfmr_name)
            .order_by(HealthScores.date.desc())
            .first()
        )

        # If no health scores yet, return empty list (Streamlit can handle this)
        if latest_row is None:
            return []

        latest_date = latest_row.date  # string in DB

        rows = (
            db.query(HealthScores)
            .filter_by(transformer_name=xfmr_name, date=latest_date)
            .all()
        )

        data = [
            {
                "id": r.id,
                "transformer_name": r.transformer_name,
                "date": r.date,
                "variable_name": r.variable_name,
                "average_value": r.average_value,
                "rated_value": r.rated_value,
                "status": r.status,
                "overall_score": r.overall_score,
                "overall_color": r.overall_color,
            }
            for r in rows
        ]

        return data


# -------------------------------------------------------------------
# POST /transformers/ -> create new transformer,
#   create its tables, run health monitoring, run forecast
# -------------------------------------------------------------------
@app.post("/transformers/", response_model=Transformer)
def create_xfmr(xfmr: Transformer):
    with SessionLocal() as db:
        db_item = Transformers(**xfmr.model_dump())

        # Prevent duplicates
        exists = (
            db.query(Transformers)
            .filter_by(transformer_name=db_item.transformer_name)
            .first()
        )
        if exists:
            raise HTTPException(status_code=400, detail="Transformer already exists")

        # Insert into transformers table
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        # Create necessary per-transformer tables
        # (this is your existing logic in DataProcessing.programFiles.Database)
        database.addTransformer()

        # Run health monitoring – this populates HealthScores and other tables
        try:
            health_monitor.run_health_monitoring()
        except Exception as e:
            print(f"[FASTAPI] Health monitor error: {e}")

        # Run lifetime forecast – this should populate ForecastData
        try:
            forecast_engine.forecast_transformer_lifetime(db_item.transformer_name)
        except Exception as e:
            print(f"[FASTAPI] Forecast engine error: {e}")

        return db_item

@app.post("/update-tables/")
def update_tables():
    with SessionLocal() as db:
        database.checkAndUpdateTransformerDataTables()
        db.commit()
    return {"status": "success"}


@app.delete("/transformers/{xfmr_name}")
def delete_xfmr(xfmr_name:str):
    with SessionLocal() as db:
        xfmr_to_delete = db.query(transformers).filter_by(transformer_name = xfmr_name).first()
        if xfmr_to_delete:
            #TODO: add removal of all transformer tables
            database.removeTransformer(xfmr_to_delete.transformer_name)
            db.delete(xfmr_to_delete)
            db.commit()
            return True
        else:
            raise HTTPException(status_code = 404, detail ="Transformer not found")

# -------------------------------------------------------------------
# DELETE /transformers/{xfmr_name} -> delete by name
# -------------------------------------------------------------------
@app.delete("/transformers/{xfmr_name}")
def delete_xfmr(xfmr_name: str):
    with SessionLocal() as db:
        xfmr_to_delete = (
            db.query(Transformers)
            .filter_by(transformer_name=xfmr_name)
            .first()
        )
        if not xfmr_to_delete:
            raise HTTPException(status_code=404, detail="Transformer not found")

        # Also remove per-transformer tables, via your Database helper
        database.removeTransformer(xfmr_to_delete.transformer_name)

        db.delete(xfmr_to_delete)
        db.commit()
        return True


# -------------------------------------------------------------------
# DELETE /transformers/{xfmrid} -> delete by ID (careful: same path
#   as above — if you keep this, consider changing the path to /transformers/id/{xfmrid})
# -------------------------------------------------------------------
@app.delete("/transformers/by-id/{xfmrid}")
def delete_by_id(xfmrid: int):
    with SessionLocal() as db:
        xfmr_to_delete = db.query(Transformers).filter_by(id=xfmrid).first()
        if not xfmr_to_delete:
            raise HTTPException(status_code=404, detail="Transformer not found")

        database.removeTransformer(xfmr_to_delete.transformer_name)
        db.delete(xfmr_to_delete)
        db.commit()
        return True
