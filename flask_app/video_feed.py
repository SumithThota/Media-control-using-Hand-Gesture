import sys
import os
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
import utils as ut
import json

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_architecture import model

# Constants
WIDTH = 1028 // 2
HEIGHT = 720 // 2
TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]
SMOOTH_FACTOR = 6

# Use absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GESTURE_RECOGNIZER_PATH = os.path.join(BASE_DIR, 'models', 'model.pth')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'label.csv')
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, 'models', 'shape_predictor_68_face_landmarks.dat')
STATE_PATH = os.path.join(BASE_DIR, 'data', 'player_state.json')

CONF_THRESH = 0.9
ABSENCE_COUNTER_THRESH = 20
SLEEP_COUNTER_THRESH = 20
EAR_THRESH = 0.21

# Initialize variables
PLOCX, PLOCY = 0, 0
CLOX, CLOXY = 0, 0
GEN_COUNTER = 0
ABSENCE_COUNTER = 0
SLEEP_COUNTER = 0
EAR_HISTORY = ut.deque([])
GESTURE_HISTORY = ut.deque([])

# Debugging: Print the absolute path of the model file
print(f"Absolute path of the model file: {GESTURE_RECOGNIZER_PATH}")

# Load models and labels
if os.path.exists(GESTURE_RECOGNIZER_PATH):
    print("Model file exists.")
    model.load_state_dict(ut.torch.load(GESTURE_RECOGNIZER_PATH))
else:
    raise FileNotFoundError(f"Model file not found: {GESTURE_RECOGNIZER_PATH}")

labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

def initialize_camera():
    cap = ut.cv.VideoCapture(0)
    cap.set(ut.cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(ut.cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap

def process_frame(frame):
    frame = ut.cv.flip(frame, 1)
    frame_rgb = ut.cv.cvtColor(frame, ut.cv.COLOR_BGR2RGB)
    return frame, frame_rgb

def detect_gestures(frame_rgb, frame):
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coordinates_list = ut.calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]
            preprocessed = ut.pre_process_landmark(important_points)
            d0 = ut.calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = ut.normalize_distances(d0, ut.get_all_distances(pts_for_distances))
            features = ut.np.concatenate([preprocessed, distances])
            conf, pred = ut.predict(features, model)
            gesture = labels[pred]
            return gesture, conf, coordinates_list
    return None, None, None

def control_youtube_player(gesture, conf, coordinates_list, det_zone, m_zone, frame):
    global PLOCX, PLOCY, CLOX, CLOXY, GEN_COUNTER, GESTURE_HISTORY
    if ut.cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= CONF_THRESH:
        gest_hist = ut.track_history(GESTURE_HISTORY, gesture)
        before_last = gest_hist[-2] if len(gest_hist) >= 2 else gest_hist[0]
        if gesture == 'Move_mouse':
            x, y = ut.mouse_zone_to_screen(coordinates_list[9], m_zone)
            CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
            CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
            ut.pyautogui.moveTo(CLOX, CLOXY)
            PLOCX, PLOCY = CLOX, CLOXY
        elif gesture == 'Right_click' and before_last != 'Right_click':
            ut.pyautogui.rightClick()
        elif gesture == 'Left_click' and before_last != 'Left_click':
            ut.pyautogui.click()
        elif gesture == 'Play_Pause' and before_last != 'Play_Pause':
            ut.pyautogui.press('space')
        elif gesture == 'Vol_up_gen':
            ut.pyautogui.press('volumeup')
        elif gesture == 'Vol_down_gen':
            ut.pyautogui.press('volumedown')
        elif gesture == 'Vol_up_ytb':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                ut.pyautogui.press('up')
        elif gesture == 'Vol_down_ytb':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                ut.pyautogui.press('down')
        elif gesture == 'Forward':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                ut.pyautogui.press('right')
        elif gesture == 'Backward':
            GEN_COUNTER += 1
            if GEN_COUNTER % 4 == 0:
                ut.pyautogui.press('left')
        elif gesture == 'fullscreen' and before_last != 'fullscreen':
            ut.pyautogui.press('f')
        elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
            ut.pyautogui.press('c')
        elif gesture == 'Neutral':
            GEN_COUNTER = 0
        ut.cv.putText(frame, f'{gesture} | {conf:.2f}', (int(WIDTH * 0.05), int(HEIGHT * 0.07)),
                      ut.cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, ut.cv.LINE_AA)

def detect_sleepiness(frame_rgb, frame_gray, frame):
    global SLEEP_COUNTER, EAR_HISTORY
    faces = detector(frame_gray)
    for face in faces:
        landmarks = predictor(frame_gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        leftEAR = ut.eye_aspect_ratio(leftEye)
        rightEAR = ut.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        EAR_HISTORY = ut.track_history(EAR_HISTORY, round(ear, 2), 20)
        mean_ear = sum(EAR_HISTORY) / len(EAR_HISTORY)
        if mean_ear < EAR_THRESH:
            SLEEP_COUNTER += 1
            ps = None
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as json_file:
                    try:
                        player_state = json.load(json_file)
                        if isinstance(player_state, dict):
                            ps = player_state.get('playerState')
                    except json.JSONDecodeError:
                        print("Error decoding JSON from state file")
            if SLEEP_COUNTER > SLEEP_COUNTER_THRESH and SLEEP_COUNTER % 3 == 0 and ps == 1:
                ut.pyautogui.press('space')
        else:
            SLEEP_COUNTER = 0
        leftEyeHull = ut.cv.convexHull(leftEye)
        rightEyeHull = ut.cv.convexHull(rightEye)
        ut.cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        ut.cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        ut.cv.putText(frame, f'EAR: {mean_ear:.2f}', (int(WIDTH * 0.90), int(HEIGHT * 0.08)),
                      ut.cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, ut.cv.LINE_AA)

# def detect_absence(frame_rgb, frame):
#     global ABSENCE_COUNTER
#     results = face_detection.process(frame_rgb)
#     if results.detections is None:
#         ABSENCE_COUNTER += 1
#         ps = None
#         if os.path.exists(STATE_PATH):
#             with open(STATE_PATH) as json_file:
#                 try:
#                     player_state = json.load(json_file)
#                     if isinstance(player_state, dict):
#                         ps = player_state.get('playerState')
#                 except json.JSONDecodeError:
#                     print("Error decoding JSON from state file")
#         if ABSENCE_COUNTER > ABSENCE_COUNTER_THRESH and ABSENCE_COUNTER % 3 == 0 and ps == 1:
#             ut.pyautogui.press('space')
#     else:
#         ABSENCE_COUNTER = 0
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             frame_height, frame_width = frame.shape[:2]
#             x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
#             w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
#             ut.cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

# def detect_absence(frame_rgb, frame):
#     global ABSENCE_COUNTER
#     results = face_detection.process(frame_rgb)
    
#     # Log the detection results
#     print(f"Detection results: {results.detections}")
    
#     if results.detections is None:
#         ABSENCE_COUNTER += 1
#         ps = None
#         if os.path.exists(STATE_PATH):
#             with open(STATE_PATH) as json_file:
#                 try:
#                     player_state = json.load(json_file)
#                     if isinstance(player_state, dict):
#                         ps = player_state.get('playerState')
#                 except json.JSONDecodeError:
#                     print("Error decoding JSON from state file")
        
#         # Log the absence counter and player state
#         print(f"ABSENCE_COUNTER: {ABSENCE_COUNTER}, Player State: {ps}")
        
#         if ABSENCE_COUNTER > ABSENCE_COUNTER_THRESH and ABSENCE_COUNTER % 3 == 0 and ps == 1:
#             ut.pyautogui.press('space')
#             print("Video paused due to absence")
#     else:
#         ABSENCE_COUNTER = 0
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             frame_height, frame_width = frame.shape[:2]
#             x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
#             w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
#             ut.cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)


def detect_absence(frame_rgb, frame):
    global ABSENCE_COUNTER
    results = face_detection.process(frame_rgb)
    
    # Log the detection results
    print(f"Detection results: {results.detections}")
    
    if results.detections is None:
        ABSENCE_COUNTER += 1
        ps = None
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as json_file:
                try:
                    player_state = json.load(json_file)
                    print(f"Player state JSON content: {player_state}")  # Log the JSON content
                    if isinstance(player_state, dict):
                        ps = player_state.get('playerState')
                        print(f"Extracted playerState: {ps}")  # Log the extracted playerState
                except json.JSONDecodeError:
                    print("Error decoding JSON from state file")
        
        # Log the absence counter and player state
        print(f"ABSENCE_COUNTER: {ABSENCE_COUNTER}, Player State: {ps}")
        
        if ABSENCE_COUNTER > ABSENCE_COUNTER_THRESH and ABSENCE_COUNTER % 3 == 0 and ps == 1:
            ut.pyautogui.press('space')
            print("Video paused due to absence")
    else:
        ABSENCE_COUNTER = 0
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            frame_height, frame_width = frame.shape[:2]
            x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
            w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
            ut.cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

def generate_video():
    cap = initialize_camera()
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frame, frame_rgb = process_frame(frame)
        det_zone, m_zone = ut.det_mouse_zones(frame)
        gesture, conf, coordinates_list = detect_gestures(frame_rgb, frame)
        if gesture:
            control_youtube_player(gesture, conf, coordinates_list, det_zone, m_zone, frame)
        frame_gray = ut.cv.cvtColor(frame_rgb, ut.cv.COLOR_RGB2GRAY)
        detect_sleepiness(frame_rgb, frame_gray, frame)
        detect_absence(frame_rgb, frame)
        _, buffer = ut.cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')