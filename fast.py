from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Replace with your actual database connection string
DATABASE_URL = "sqlite:///./transformerDB.db" 

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



