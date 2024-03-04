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

# @app.post('/recommendation')
# async def recommendation(request: Request):
#     data = await request.json()
#     text = data.get("user_rec")
#     answer = recommend(text)
#     return JSONResponse(content=answer)


# Testing UI
@app.post('/recommendation')
async def recommendation(data:request_body):
    input_dict = dict(data)
    # text = data.get("user_rec")
    answer = process(input_dict)
    return JSONResponse(content=answer)