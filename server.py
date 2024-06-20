# server.py

import uvicorn
import socketio
import cv2
import base64
import numpy as np
from fastapi import FastAPI
from pose import potencia

app = FastAPI()

# Socket.IO server setup
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Endpoint to test server connectivity
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Socket.IO event handlers
@sio.on("connect")
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.on("disconnect")
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

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
    await sio.emit('message', response_data)
    print(response_data)

# @sio.on("send_frame")
# async def send_frame(sid, data):
#     global counter

#     # Example: Writing data to a file
#     with open("data_shubham.txt", 'a') as file:
#         file.write(data)

#     try:
#         buffer = base64.b64decode(data[23:])  # Ensure data is correctly sliced if needed
#         if buffer:
#             frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), 1)
#             if frame is not None:
#                 # Process frame and get counters
#                 left_counter, right_counter = potencia(frame)
#                 print(f"Left Counter: {left_counter}, Right Counter: {right_counter}")

#                 # Emit response data to the client
#                 response_data = {
#                     'left_count': left_counter,
#                     'right_count': right_counter
#                 }
#                 await sio.emit('message', response_data)
#             else:
#                 print("Failed to decode frame from buffer.")
#         else:
#             print("Empty buffer received or decoding failed.")
#     except Exception as e:
#         print(f"Error decoding or processing frame: {e}")

#     # Close OpenCV windows properly
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, lifespan="on", reload=True)
