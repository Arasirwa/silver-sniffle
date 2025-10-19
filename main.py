from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from model_utils import predict_binary, predict_multiclass

app = FastAPI()

# Serve static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-text")
async def predict_text(data: dict):
    text = data.get("text", "")
    binary = predict_binary([text])[0]
    multiclass = predict_multiclass([text])[0]
    return {"binary": binary, "multiclass": multiclass}

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return JSONResponse(status_code=400, content={"error": "CSV must have a 'text' column."})
    texts = df["text"].astype(str).tolist()

    binary_preds = predict_binary(texts)
    multi_preds = predict_multiclass(texts)

    results = []
    for b, m in zip(binary_preds, multi_preds):
        results.append({
            "text": b["text"],
            "binary": b["label"],
            "binary_conf": b["confidence"],
            "multiclass": m["label"],
            "multi_conf": m["confidence"]
        })
    return {"results": results}