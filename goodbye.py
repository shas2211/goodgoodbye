import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import copy

# ---------------- guys download and install the packages in prior ----------------
CENTER_MARGIN = 20   # pixels (try 8–20) i have put this based on my camera margin

FPS = 30
DELAY_SECONDS = 3.1   # change to 15, 20, etc. adjust based on the requirement , well i wanted my gf to dance with 3.1 factor delay .
DELAY_FRAMES = int(FPS * DELAY_SECONDS)
CAM_INDEX = 0

# ---------------------------------------this here might take some time to load , so have patience

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

# ---------- HELPERS ---------- important functions
def mirror_landmarks(landmarks):
    mirrored = copy.deepcopy(landmarks)
    for lm in mirrored.landmark:
        lm.x = 1.0 - lm.x   # CENTER MIRROR lol here the code actually prints ur own locations but mirrored respect to the center ,good idea na!!
    return mirrored


def draw(img, landmarks, connections, line_color, point_color):
    mp_drawing.draw_landmarks(
        img,
        landmarks,
        connections,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=point_color, thickness=4, circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=line_color, thickness=4
        )
    )

#--------------------------coming to the main code now , which is pretty simple 

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

buffer = deque(maxlen=DELAY_FRAMES)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    
    you_canvas = np.zeros_like(frame)
    her_canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)

   
    if result.pose_landmarks:
        buffer.append({
            "pose": result.pose_landmarks,
            "lh": result.left_hand_landmarks,
            "rh": result.right_hand_landmarks
        })

        # YOU – white lines, red points
        # draw(you_canvas, result.pose_landmarks,
        #      mp_holistic.POSE_CONNECTIONS,
        #      (255, 255, 255), (0, 0, 255))

        if result.left_hand_landmarks:
            draw(you_canvas, result.left_hand_landmarks,
                 mp_holistic.HAND_CONNECTIONS,
                 (255, 255, 255), (0, 0, 255))

        if result.right_hand_landmarks:
            draw(you_canvas, result.right_hand_landmarks,
                 mp_holistic.HAND_CONNECTIONS,
                 (255, 255, 255), (0, 0, 255))

    # ---------- DRAW HER with delayed frames ----------
    if len(buffer) == DELAY_FRAMES:
        old = buffer[0]

        pose_m = mirror_landmarks(old["pose"])
        draw(her_canvas, pose_m,
             mp_holistic.POSE_CONNECTIONS,
             (255, 0, 255), (255, 255, 255))  # NEON PINK

        if old["lh"]:
            draw(her_canvas, mirror_landmarks(old["lh"]),
                 mp_holistic.HAND_CONNECTIONS,
                 (255, 0, 255), (255, 255, 255))

        if old["rh"]:
            draw(her_canvas, mirror_landmarks(old["rh"]),
                 mp_holistic.HAND_CONNECTIONS,
                 (255, 0, 255), (255, 255, 255))

    
    mid = w // 2

    you_masked = you_canvas.copy()
    you_masked[:, mid + CENTER_MARGIN:] = 0   # allow overlap

    her_masked = her_canvas.copy()
    her_masked[:, :mid - CENTER_MARGIN] = 0

    # ---------- OVERLAY ----------
    left_full = cv2.addWeighted(frame, 1.0, you_masked, 1.0, 0)
    right_full = cv2.addWeighted(frame, 1.0, her_masked, 1.0, 0)

    # ---------- FINAL SPLIT ----------
    output = np.hstack((
        left_full[:, :w//2],
        right_full[:, w//2:]
    ))

    cv2.putText(output, "YOU", (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(output, f"HER ????)",
                (w//2 + 40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("sahiba or delusional gf", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
