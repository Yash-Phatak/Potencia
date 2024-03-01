from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import JSONResponse
from recommend import recommend
from pydantic import BaseModel
import uvicorn

class request_body(BaseModel):
    message: str 

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
    data = data.message
    # text = data.get("user_rec")
    answer = recommend(data)
    return JSONResponse(content=answer)