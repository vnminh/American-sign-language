import socket
from datetime import datetime

import numpy as np
import cv2
import os
import mediapipe as mp
import tensorflow as tf
import keras
from queue import Queue
from threading import Thread
import socket
import struct
import time
ESP32_IP = "192.168.92.239"  # Change to your ESP32-CAM IP
ESP32_PORT = 80  # Ensure this matches the port set in your ESP32 code

#GLOBAL VAR
FILE_PATH_FOR_CLASS = os.path.join(os.pardir,'class_name.txt')
CLASS_LIST = [name.strip() for name in open(FILE_PATH_FOR_CLASS,'r').readlines()]
NUM_WORD = len(CLASS_LIST)
SEQ_LEN = 20
IMAGE_CAM_HEIGHT = 480
#SHARED THREAD VAR
STOP = False

raw_image_queue = Queue(10) # receive from cam

display_queue = Queue(10) # For display

sample_queue = Queue(10) # For 20-frame sequence of extracted landmark

result_queue = Queue(10)  # For store predicted words to send to client

# SOCKET CONFIGURATION
SERVER_HOST = "0.0.0.0"  # Update with your actual IP
SERVER_PORT = 6969

#INIT MEDIAPIPE MODEL
mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic.Holistic(static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence = 0.5)
#-----------------------------------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_len, dim, **kwargs):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.get_positional_encoding(seq_len, dim)

    def get_positional_encoding(self, seq_len, dim):
        pos = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
        i = np.arange(dim, dtype=np.float32)[np.newaxis, :]
        angle_rates = 1 / np.power(10000.0, (2 * (i // 2)) / np.float32(dim))
        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return tf.convert_to_tensor(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        return super().get_config()

@keras.saving.register_keras_serializable()
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, head_size=256, num_heads=12, ff_dim=512, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dropout(rate = dropout),
            keras.layers.Dense(head_size, activation="linear"),
            keras.layers.Dropout(rate = dropout),
        ])

    def call(self, inputs):
        norm1 = self.norm1(inputs)
        attn_output = self.attention(norm1, norm1) + norm1
        norm2 = self.norm2(inputs + attn_output)
        ffn_output = self.ffn(norm2) + norm2
        return ffn_output


@keras.saving.register_keras_serializable()
class ViTSignLanguageModel(keras.Model):
    def __init__(self, seq_len, feature_dim, num_classes, **kwargs):
        super(ViTSignLanguageModel, self).__init__()
        self.conv1 = keras.layers.Conv1D(108, kernel_size=3, activation='linear', padding='same')
        self.conv2 = keras.layers.Conv1D(108, kernel_size=3, activation='linear', padding='same')
        self.pos_encoding = PositionalEncoding(seq_len, 108)
        self.class_token = self.add_weight(shape=(1, 1, 108), initializer= keras.initializers.RandomNormal(mean=0.0, stddev=1), trainable=True, name='class_token')
        self.lamda = keras.layers.Lambda(
            lambda tensor: keras.layers.concatenate((keras.ops.repeat(self.class_token, keras.ops.shape(tensor)[0], axis=0), tensor), axis = 1),
            output_shape=(seq_len + 1, 108))
        self.dropout = keras.layers.Dropout(0.2)
        self.encoders = [TransformerEncoder(108, 9, 512) for _ in range(6)]
        self.glo_avg_pool = keras.layers.GlobalAveragePooling1D()
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense3 = keras.layers.Dense(num_classes, activation='softmax')
        

    def call(self, inputs):
        x = self.conv1(inputs) + inputs
        y = self.conv2(x) + x
        x = self.lamda(y)
        x = self.dropout(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x[...,0,:]
        y = self.glo_avg_pool(y)
        z = x + y
        z = self.norm3(z)
        return self.dense3(z)

    def get_config(self):
      return {
          'seq_len':SEQ_LEN,
          'feature_dim':108,
          'num_classes':NUM_WORD
      }

    @classmethod
    def from_config(cls, config):
      return cls(**config)

    def build(self, input_shape):
        x = keras.layers.Input(shape=input_shape)
        return keras.models.Model(inputs=[x], outputs=self.call(x))

    def build_graph(self, input_shape):
        x = keras.layers.Input(shape=input_shape)
        return keras.models.Model(inputs=[x], outputs=self.call(x))
#-----------------------------------------------------------------------------------------------------------------
def drawLandmarks(image, res):
    '''
    Function for draw landmark
    '''
    def drawLandmarksPose(image, pose_landmarks):
        if pose_landmarks:
            mp_draw.draw_landmarks(image,pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)

    def drawLandmarksHand(image, hand_landmarks):
        if hand_landmarks:
            mp_draw.draw_landmarks(image,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)

    drawLandmarksPose(image, res.pose_landmarks)
    drawLandmarksHand(image, res.left_hand_landmarks)
    drawLandmarksHand(image, res.right_hand_landmarks)

def extract_keypoints(res_holistic, frame_size = (480,640)):
    
    def out_bound(left_wrist, right_wrist, x_center, y_center, half_w, half_h):
        left, right = x_center - half_w, x_center + half_w
        top, bottom = y_center - half_h, y_center + half_h
        out = lambda land_mark: land_mark[0] < left or land_mark[0] > right or land_mark[1] < top or land_mark[1] > bottom
        return out(left_wrist) and out(right_wrist)

    def normalize_pose(list_of_landmarks, nose):
        head_metric = np.linalg.norm(list_of_landmarks[0] - list_of_landmarks[1])/2
        y_min, y_max = nose[1] - head_metric, nose[1] + 3.75*head_metric
        x_center = (list_of_landmarks[0,0] + list_of_landmarks[1,0])/2
        y_center = (y_min + y_max)/2
        half_box_size_w = head_metric*3
        half_box_size_h = head_metric*2.375
        if out_bound(list_of_landmarks[4], list_of_landmarks[5], x_center, y_center, half_box_size_w, half_box_size_h):
            return None, True
        return (list_of_landmarks - np.array([x_center, y_center])) / np.array([half_box_size_w, half_box_size_h]), False
    
    def normalize_hand(list_of_landmarks):
        x_min, x_max = np.min(list_of_landmarks[:,0]), np.max(list_of_landmarks[:,0])
        y_min, y_max = np.min(list_of_landmarks[:,1]), np.max(list_of_landmarks[:,1])
        x_center = (x_min + x_max)/2
        y_center = (y_min + y_max)/2
        half_box_size = max(x_max-x_min,y_max-y_min)/2
        return (list_of_landmarks - np.array([x_center, y_center])) / half_box_size
    
    
    if (not res_holistic.pose_landmarks): return np.zeros((12*2 + 42*2,))
    #extract and normalize pose_landmarks (just shoulder and arms 11->22)
    nose = np.array([res_holistic.pose_landmarks.landmark[0].x, res_holistic.pose_landmarks.landmark[0].y]) * np.array([frame_size[1], frame_size[0]])
    pose_landmarks = np.array([[res_holistic.pose_landmarks.landmark[i].x,
                                res_holistic.pose_landmarks.landmark[i].y] for i in range(11,23)]) if res_holistic.pose_landmarks else np.zeros((12,2))
    #scale to absolute coordinate of image
    pose_landmarks = pose_landmarks * np.array([frame_size[1], frame_size[0]])
    pose_landmarks, isOutBound = normalize_pose(pose_landmarks, nose)
    # check wrist out bound of box
    if isOutBound:
        return np.zeros((12*2 + 42*2,))
    #extract and normalize hand_landmarks
    if not res_holistic.left_hand_landmarks and not res_holistic.right_hand_landmarks:
        return np.zeros((12*2 + 42*2,))
    hand_landmarks = {'Left':np.zeros((21,2),dtype=np.double),'Right':np.zeros((21,2),dtype=np.double)}
    if res_holistic.left_hand_landmarks:
        hand_landmarks['Left'] = np.array([[landmark.x,landmark.y] for landmark in res_holistic.left_hand_landmarks.landmark],dtype=np.double)
        #scale to absolute coordinate of image
        hand_landmarks['Left'] = hand_landmarks['Left']* np.array([frame_size[1], frame_size[0]])
        hand_landmarks['Left'] = normalize_hand(hand_landmarks['Left'])

    if res_holistic.right_hand_landmarks:
        hand_landmarks['Right'] = np.array([[landmark.x,landmark.y] for landmark in res_holistic.right_hand_landmarks.landmark],dtype=np.double)
        #scale to absolute coordinate of image
        hand_landmarks['Right'] = hand_landmarks['Right']* np.array([frame_size[1], frame_size[0]])
        hand_landmarks['Right'] = normalize_hand(hand_landmarks['Right'])

    return np.concatenate((pose_landmarks,hand_landmarks['Left'], hand_landmarks['Right']), axis = None)


#   _____    ______    _____   ______   _____  __      __  ______ 
#  |  __ \  |  ____|  / ____| |  ____| |_   _| \ \    / / |  ____|
#  | |__) | | |__    | |      | |__      | |    \ \  / /  | |__   
#  |  _  /  |  __|   | |      |  __|     | |     \ \/ /   |  __|  
#  | | \ \  | |____  | |____  | |____   _| |_     \  /    | |____ 
#  |_|  \_\ |______|  \_____| |______| |_____|     \/     |______|
                                                                
def receive_frame(client_socket):
    """ Receives a single frame from ESP32-CAM via a socket """
    try:
        # Receive frame size (4 bytes)
        size_data = client_socket.recv(4)
        if not size_data:
            return None
        frame_size = struct.unpack('<I', size_data)[0]

        # Receive frame data
        frame_data = bytearray()
        while len(frame_data) < frame_size:
            chunk = client_socket.recv(min(frame_size - len(frame_data), 8192))
            if not chunk:
                return None
            frame_data.extend(chunk)

        # Convert buffer to frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    except Exception as e:
        print(f"Error receiving frame: {e}")
        return None


# -----------------------------------------------------------------------------------------------------------

def receive_from_cam():
    """ Modified function to receive frames from ESP32-CAM """
    global STOP
    client_socket = None

    try:
        print("RECEIVE_FROM_CAM THREAD:", f"Connecting to ESP32-CAM at {ESP32_IP}:{ESP32_PORT}...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ESP32_IP, ESP32_PORT))
        print("RECEIVE_FROM_CAM THREAD:","Connected successfully")


        while not STOP:
            frame = receive_frame(client_socket)
            if frame is None:
                continue

            # Process frame (same as before)
            scale = IMAGE_CAM_HEIGHT / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            frame = cv2.flip(frame, 1)
            raw_image_queue.put([frame])
            time.sleep(0.01)
            

    except Exception as e:
        print("RECEIVE_FROM_CAM THREAD:", f"Error: {e}")

    finally:
        if client_socket:
            client_socket.close()
        raw_image_queue.put(None)
        print("RECEIVE_FROM_CAM THREAD: End")

#   ______  __   __  _______   _____                _____   _______ 
#  |  ____| \ \ / / |__   __| |  __ \      /\      / ____| |__   __|
#  | |__     \ V /     | |    | |__) |    /  \    | |         | |   
#  |  __|     > <      | |    |  _  /    / /\ \   | |         | |   
#  | |____   / . \     | |    | | \ \   / ____ \  | |____     | |   
#  |______| /_/ \_\    |_|    |_|  \_\ /_/    \_\  \_____|    |_|                                                               

def extract():
    global STOP
    time_seq_feature = []
    while not STOP:
        try:
            frame = raw_image_queue.get()
            if not frame:
                break
            frame = frame[0]
            display_queue.put([frame])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = mp_holistic.process(frame_rgb)

            # Extract keypoints
            time_seq_feature.append(extract_keypoints(res, frame_rgb.shape))

            # Display video
            # drawLandmarks(frame, res)

            # Put extracted data into the queue
            if len(time_seq_feature) == SEQ_LEN:
                try:
                    sample_queue.put([np.array([time_seq_feature])])
                except Exception as e:
                    print("EXTRACT THREAD:", e)
                time_seq_feature = []

            
        except Exception as e:
            print("EXTRACT THREAD:",e)
            break

    display_queue.put(None)
    sample_queue.put(None)
    print('EXTRACT THREAD: End')

#   _____    _____    _____   _____    _                  __     __
#  |  __ \  |_   _|  / ____| |  __ \  | |          /\     \ \   / /
#  | |  | |   | |   | (___   | |__) | | |         /  \     \ \_/ / 
#  | |  | |   | |    \___ \  |  ___/  | |        / /\ \     \   /  
#  | |__| |  _| |_   ____) | | |      | |____   / ____ \     | |   
#  |_____/  |_____| |_____/  |_|      |______| /_/    \_\    |_|                 
                                                                                     
def display():
    global STOP
    while not STOP:
        try:
            frame = display_queue.get()
            if not frame: 
                break
            
            frame = frame[0]
            cv2.imshow("ESP32-CAM Feed", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                STOP = True
                break
            
        except Exception as e:
            print("DISPLAY THREAD:", e)
    
    cv2.destroyAllWindows()
    print('DISPLAY THREAD: End')

#   _____    _____    ______   _____    _____    _____   _______ 
#  |  __ \  |  __ \  |  ____| |  __ \  |_   _|  / ____| |__   __|
#  | |__) | | |__) | | |__    | |  | |   | |   | |         | |   
#  |  ___/  |  _  /  |  __|   | |  | |   | |   | |         | |   
#  | |      | | \ \  | |____  | |__| |  _| |_  | |____     | |   
#  |_|      |_|  \_\ |______| |_____/  |_____|  \_____|    |_|   
                                                            
                                                            
def predict(my_model: keras.Model):
    '''
    Task of predict
    '''
    #Loop
    global STOP
    while not STOP:
        try:
            x = sample_queue.get()
        except Exception as e:
            print('PREDICT THREAD:',e)
            continue
        if not x:
            break
        y = my_model.predict(x[0],verbose=0)
        class_id = np.argmax(y)
        if y[0][class_id] < 0.85:
            class_id = len(CLASS_LIST) - 1
        predicted_word = CLASS_LIST[class_id]
        # result_queue.put(predicted_word)
        print('PREDICT THREAD:','predict word','\033[30;31m'+predicted_word+'\033[0m')
    
    # result_queue.put(None)
    
    print('PREDICT THREAD: End')

#    _____   ______   _   _   _____  
#   / ____| |  ____| | \ | | |  __ \ 
#  | (___   | |__    |  \| | | |  | |
#   \___ \  |  __|   | . ` | | |  | |
#   ____) | | |____  | |\  | | |__| |
#  |_____/  |______| |_| \_| |_____/ 
                                   
                                
def send_predictions():
    """ Sends predicted words from `result_queue` to the Android client via socket. """
    global STOP
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(1)
        print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address}")

        while not STOP:
            word = result_queue.get()  # Get prediction from queue
            if word is None:  # Stop condition
                break

            # Get formatted timestamp
            timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            # Send the result to the client
            message = f"{word}|{timestamp}"
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent: {message}")

        client_socket.close()
        server_socket.close()
        print("SEND THREAD: End")
    except Exception as e:
        print(f"Socket Error: {e}")

#   __  __              _____   _   _ 
#  |  \/  |     /\     |_   _| | \ | |
#  | \  / |    /  \      | |   |  \| |
#  | |\/| |   / /\ \     | |   | . ` |
#  | |  | |  / ____ \   _| |_  | |\  |
#  |_|  |_| /_/    \_\ |_____| |_| \_|
                                    
                                    

if __name__=='__main__':
    my_model = keras.models.load_model(os.path.join(os.pardir,'Model','model_22_04_2025_02_10_1745287819_lstm_new_data_treat.keras'))
    receive_from_cam_thread = Thread(target=receive_from_cam)
    extract_thread = Thread(target=extract)
    display_thread = Thread(target=display)
    predict_thread = Thread(target=predict,args=(my_model,))
    # socket_thread = Thread(target=send_predictions)
    
    receive_from_cam_thread.start()
    extract_thread.start()
    display_thread.start()
    predict_thread.start()
    # socket_thread.start()