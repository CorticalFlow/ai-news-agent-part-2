

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage to prevent CUDA warnings when drivers are missing.
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow info and warning messages.
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Use offscreen rendering to bypass display issues

import cv2
import socket
import numpy as np
import struct
from datetime import datetime
from retinaface import RetinaFace  # Added import for face detection
from deepface import DeepFace

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


HOST = os.getenv('HOST', '0.0.0.0')
PORT = os.getenv('PORT', 9999)



def receive_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main():
    print('main')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print("Listening at:", (HOST, PORT))

    client_socket, addr = server_socket.accept()
    print('Connection from:', addr)

    # Removed GUI window initialization since we're running headless
    # cv2.namedWindow('Received', cv2.WINDOW_NORMAL)  # No GUI window initialization needed in headless mode

    try:
        while True:
            message_size = receive_all(client_socket, struct.calcsize(">L"))
            if not message_size:
                break
            message_size = struct.unpack(">L", message_size)[0]
            frame_data = receive_all(client_socket, message_size)
            if not frame_data:
                break

            print('new frame received')

            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # extract text


            # Face detection using RetinaFace and drawing bounding boxes
            faces = RetinaFace.detect_faces(frame)
            if isinstance(faces, dict):
                for face in faces.values():

                    print('found face')
                    print(face)

                    bbox = face["facial_area"]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    # Extract the face region using the bounding box coordinates
                    face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    # Compute vector embedding using DeepFace with the Facenet model
                    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
                    print("Face Embedding:", embedding)

            timestamp = datetime.utcnow().timestamp()
            # save to filesystem
            cv2.imwrite(f'./data/frame_{timestamp}.png', frame)

            # Removed GUI key-check as there's no display for capture
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    except Exception as e:
        print(e)
    finally:
        print('finally')
        cv2.destroyAllWindows()
        client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    print("main call")
    main() 