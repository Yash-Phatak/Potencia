from fastapi import FastAPI,HTTPException,Request
from typing import List
from fastapi.responses import JSONResponse
from recommend import process
from pydantic import BaseModel
import uvicorn

class request_body(BaseModel):
    uid: str
    target_muscle: List[str]
    level: str
    type: List[str]

app = FastAPI()

@app.get('/')
def main():
    return {'message':'Potencia'}

# Testing UI
@app.post('/recommendation')
async def recommendation(data:Request):
    input_dict = await data.json()
    input_dict = dict(input_dict)
    answer = process(input_dict)
    return JSONResponse(content=answer)