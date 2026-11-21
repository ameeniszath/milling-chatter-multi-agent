from fastapi import FastAPI, UploadFile, File
from scipy.io import loadmat
import numpy as np
import io

app = FastAPI()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    contents = await file.read()
    mat = loadmat(io.BytesIO(contents))

    time = mat["time"].flatten().tolist()
    xacc = mat["xtacc"].flatten().tolist()   # use your actual key

    return {
        "time": time,
        "xacc": xacc
    }