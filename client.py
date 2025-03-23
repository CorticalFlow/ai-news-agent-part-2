import socket
import cv2
import mss
import numpy as np
import struct
import os

IMAGE_QUALITY = os.getenv('IMAGE_QUALITY', 100)
SERVER_HOST = os.getenv('SERVER_HOST', '127.0.0.1')  

# If you have multiple screens, you can change the screen index to capture from a different screen.
SCREEN_INDEX = os.getenv('SCREEN_INDEX', 1)

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
 
    port = 9999
    client_socket.connect((SERVER_HOST, port))

    with mss.mss() as sct:
        monitor = sct.monitors[SCREEN_INDEX]
        try:
            while True:
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
                message = struct.pack(">L", len(buffer)) + buffer.tobytes()
                client_socket.sendall(message)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main() 