# server.py

import uvicorn
import socketio
import cv2
import base64
import numpy as np
from fastapi import FastAPI,File,UploadFile
import json
from pose import potencia
from exercises import deadlift,jump
from fastapi.responses import JSONResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from equip_detection.detect import detect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint to test server connectivity
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/equipment")
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    image = Image.open(BytesIO(contents))
    print("Image Found")
    result = detect(image)
    output = 'plan.json'
    with open(output, 'r') as file:
        dictionary = json.load(file)
    return JSONResponse(content=dictionary)

# Socket.IO server setup
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Socket.IO event handlers
@sio.on("connect")
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.on("disconnect")
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.on("deadlift_frame")
async def deadlift_frame(sid,data):

    buffer = base64.b64decode(data)
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), 1)

    # Process frame using socket function
    lean,dist,stage,rep = deadlift(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    response_string = str(lean)+" "+str(dist) + " " + str(stage)+" "+str(rep)
    await sio.emit('message', response_string)
    print(response_string)

@sio.on("jump_frame")
async def jump_frame(sid,data):
    buffer = base64.b64decode(data)
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), 1)

    # Process frame using socket function
    stage,counter = jump(frame)

    # Display the received frame (for debugging)
    # cv2.imshow('Received Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    response_string = str(stage)+" "+str(counter)
    await sio.emit('message', response_string)
    print(response_string)

@sio.on("send_frame")
async def send_frame(sid, data):
    global counter
    if not hasattr(send_frame, 'counter'):
        send_frame.counter = 0

    # Example: Writing data to a file
    with open("data_shubham.txt", 'a') as file:
        file.write(data)

    # Decode base64 and process frame
    buffer = base64.b64decode(data)
    frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), 1)

    # Process frame using socket function
    left_counter,right_counter = potencia(frame)

    # Display the received frame (for debugging)
    # cv2.imshow('Received Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    response_data = {
                'left_count': left_counter,
                'right_count': right_counter
            }
    response_string = str(left_counter)+" "+str(right_counter)
    await sio.emit('message', response_string)
    print(response_data)



if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, lifespan="on", reload=True)
