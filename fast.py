from sqlalchemy import create_engine, MetaData, Table, Column,Integer,Float,ForeignKey,String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
# Replace with your actual database connection string
DATABASE_URL = "sqlite:///./transformerDB.db" 
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_table_by_name(table_name: str):
    """
    Reflects the database and returns a Table object for the given table name.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    if table_name in metadata.tables:
        return Table(table_name, metadata, autoload_with=engine)
    else:
        return None

# Example usage within a FastAPI endpoint
from fastapi import FastAPI, HTTPException

class transformers(Base):
    __tablename__ = "transformers"
    id = Column(Integer,primary_key = "true",)
    transformer_name = Column(String,nullable = False)
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

class Transformer(BaseModel):
    transformer_name: str
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

app = FastAPI()

@app.get("/transformers/")
async def read_xfmrs():
    table = get_table_by_name("transformers")
    if table == None:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    with SessionLocal() as db:
        # You can now query the reflected table
        # For example, to get all rows:
        results = db.query(table).all() 
        
        # Convert results to a serializable format if needed
        # (e.g., list of dictionaries)
        data = []
        for row in results:
            data.append(row._asdict()) # For Row objects, convert to dict
        return data

@app.get("/transformers/{xfmr_name}")
async def read_xfmr(xfmr_name: str):
    table = get_table_by_name(f"{xfmr_name}_average_metrics_day")
    if table == None:
        raise HTTPException(status_code = 404, detail = "Transformer not found")

    with SessionLocal() as db:
        results = db.query(table).all()

        xfmrData = []
        for row in results:
            xfmrData.append(row._asdict())
        return xfmrData

@app.get("/transformers/status/{xfmr_name}")
async def read_xfmr_status(xfmr_name: str):
    table = get_table_by_name(f"{xfmr_name}_HealthScores")
    if table == None:
        raise HTTPException(status_code = 404, detail = "Transformer not found")

    with SessionLocal() as db:
        results = db.query(table).all()
        xfmr_data = []
        for row in results:
            xfmr_data.append(row._asdict())
        return xfmr_data

@app.post("/transformers/",response_model=Transformer)
async def create_xfmr(xfmr: Transformer):
    with SessionLocal() as db:
        db_item = transformers(**xfmr.model_dump())
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item

@app.delete("/transformers/{xfmr_name}")
async def delete_xfmr(xfmr_name:str):
    with SessionLocal() as db:
        xfmr_to_delete = db.query(transformers).filter_by(transformer_name = xfmr_name).first()
        if xfmr_to_delete:
            db.delete(xfmr_to_delete)
            db.commit()
            return True
        else:
            raise HTTPException(status_code = 404, detail ="Transformer not found")
@app.delete("/temp/")
async def delete(xfmrid: int):
    with SessionLocal() as db:
        xfmr_to_delete = db.query(transformers).filter_by(id = xfmrid).first()
        if xfmr_to_delete:
            db.delete(xfmr_to_delete)
            db.commit()
            return True


