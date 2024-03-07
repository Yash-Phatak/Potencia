from fastapi import FastAPI,HTTPException,Request
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from recommend import process
from pydantic import BaseModel
import uvicorn

app = FastAPI()
# Allow all origins, put specific origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class request_body(BaseModel):
    uid: str
    target_muscle: List[str]
    level: str
    type: List[str]

@app.get('/')
def main():
    return {'message':'Potencia'}

# Testing UI
@app.post('/recommendation')
async def recommendation(data:Request):
    input_dict = await data.json()
    input_dict = dict(input_dict)
    print(input_dict)
    answer = process(input_dict)
    return JSONResponse(content=answer)