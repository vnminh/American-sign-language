import numpy as np
import cv2
import os
import mediapipe as mp
import tensorflow as tf
import keras
from queue import Queue
from threading import Thread

#GLOBAL VAR
FILE_PATH_FOR_CLASS = os.path.join(os.pardir,'class_name.txt')
CLASS_LIST = [name.strip() for name in open(FILE_PATH_FOR_CLASS,'r').readlines()]
NUM_WORD = len(CLASS_LIST)
SEQ_LEN = 20
IMAGE_CAM_HEIGHT = 480
#SHARED THREAD VAR
sample_queue = Queue(10) 
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
    def __init__(self, head_size=256, num_heads=12, ff_dim=1024, dropout=0.1, **kwargs):
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
        self.conv1 = keras.layers.Conv1D(128, kernel_size=3, activation='gelu', padding='valid')
        self.conv2 = keras.layers.Conv1D(128, kernel_size=3, activation='gelu', padding='valid')
        self.bi_lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, activation='gelu'),merge_mode="sum")
        self.bi_lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, activation='gelu'),merge_mode="sum")
        self.pos_encoding = PositionalEncoding(seq_len - 2*2, 128)
        self.class_token = self.add_weight(shape=(1, 1, 128), initializer=keras.initializers.zeros(), trainable=True, name='class_token')
        self.lamda = keras.layers.Lambda(
            lambda tensor: keras.layers.concatenate((keras.ops.repeat(self.class_token, keras.ops.shape(tensor)[0], axis=0), tensor), axis = 1),
            output_shape=(seq_len - 2*2 + 1, 128))
        self.dropout0 = keras.layers.Dropout(0.1)
        self.encoders = [TransformerEncoder(128, 8, 512) for _ in range(12)]
        # self.dense1 = keras.layers.Dense(128, activation='linear')
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        # self.dense2 = keras.layers.Dense(1024, activation='linear')
        # self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense3 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bi_lstm1(x)
        x = self.bi_lstm2(x)
        x = self.pos_encoding(x)
        x = self.lamda(x)
        x = self.dropout0(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x[...,0,:]
        # x = self.dense1(x)
        x = self.norm1(x)
        # x = self.dense2(x)
        # x = self.norm2(x)
        return self.dense3(x)

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
    def normalize_pose(list_of_landmarks, nose):
        head_metric = np.linalg.norm(list_of_landmarks[0] - list_of_landmarks[1])/2
        y_min, y_max = nose[1] - head_metric, nose[1] + 6*head_metric
        x_center = (list_of_landmarks[0,0] + list_of_landmarks[1,0])/2
        y_center = (y_min + y_max)/2
        half_box_size_w = head_metric*3
        half_box_size_h = head_metric*3.5
        return (list_of_landmarks - np.array([x_center, y_center])) / np.array([half_box_size_w, half_box_size_h])
    def normalize_hand(list_of_landmarks):
        x_min, x_max = np.min(list_of_landmarks[:,0]), np.max(list_of_landmarks[:,0])
        y_min, y_max = np.min(list_of_landmarks[:,1]), np.max(list_of_landmarks[:,1])
        x_center = (x_min + x_max)/2
        y_center = (y_min + y_max)/2
        half_box_size = max(x_max-x_min,y_max-y_min)/2
        return (list_of_landmarks - np.array([x_center, y_center])) / half_box_size
    
    
    if not res_holistic.pose_landmarks: return np.zeros((12*2 + 42*2,))
    #extract and normalize pose_landmarks (just shoulder and arms 11->22)
    nose = np.array([res_holistic.pose_landmarks.landmark[0].x, res_holistic.pose_landmarks.landmark[0].y]) * np.array([frame_size[1], frame_size[0]])
    pose_landmarks = np.array([[res_holistic.pose_landmarks.landmark[i].x,
                                res_holistic.pose_landmarks.landmark[i].y] for i in range(11,23)]) if res_holistic.pose_landmarks else np.zeros((12,2))
    #scale to absolute coordinate of image
    pose_landmarks = pose_landmarks * np.array([frame_size[1], frame_size[0]])
    pose_landmarks = normalize_pose(pose_landmarks, nose)

    #extract and normalize hand_landmarks
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

def feed():
    '''
    Task of feed thread
    '''
    video_reader =  cv2.VideoCapture(0)
    # video_reader =  cv2.VideoCapture(os.path.join(os.pardir,'dataset','sometimes','d1.mp4'))
    #Init
    time_seq_feature = []
    #Loop    
    while video_reader.isOpened():
        success, frame = video_reader.read()
        if not success:
            break
        #process
        scale = IMAGE_CAM_HEIGHT / frame.shape[0]
        frame = cv2.resize(frame,(int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_holistic.process(frame_rgb)
        #Extract
        time_seq_feature.append(extract_keypoints(res, frame_rgb.shape))
        #display
        drawLandmarks(frame, res)
        cv2.imshow('video',frame)
        #Put to queue
        if len(time_seq_feature) == SEQ_LEN:
            try:
                sample_queue.put([np.array([time_seq_feature])])
            except Exception as e:
                print('FEED THREAD:',e)
            time_seq_feature = time_seq_feature[-20:]
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    #END
    sample_queue.put(None)
    video_reader.release()
    cv2.destroyAllWindows()

def predict(my_model: keras.Model):
    '''
    Task of predict
    '''
    #Loop
    while True:
        try:
            x = sample_queue.get()
        except Exception as e:
            print('PREDICT THREAD:',e)
            continue
        if not x:
            print('PREDICT THREAD: End')
            break
        y = my_model.predict(x[0],verbose=0)
        class_id = np.argmax(y)
        print('PREDICT THREAD:','predict word','\033[30;31m'+CLASS_LIST[class_id]+'\033[0m'+f': {round(y[0][class_id]*100)}')

if __name__=='__main__':
    my_model = keras.models.load_model(os.path.join(os.pardir,'Model','model_05_03_2025_16_14_1741191281.keras'))
    feed_thread = Thread(target=feed)
    predict_thread = Thread(target=predict,args=(my_model,))
    feed_thread.start()
    predict_thread.start()