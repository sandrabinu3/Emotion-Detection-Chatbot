import os
import time
import base64
import asyncio
import concurrent.futures
from tracemalloc import start

import cv2
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from hsemotion.facial_emotions import HSEmotionRecognizer
import torch
import traceback


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Using device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MAX_HISTORY = 20

def trim_history(history):
    # Keep system prompt + last N messages
    return [history[0]] + history[-MAX_HISTORY:]
# ==========================================
# APP SETUP
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vision tasks are heavier, so they get their own pool to avoid blocking chat replies.
vision_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
chat_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# HSEmotion — swapped in for DeepFace's built-in FER model, runs on GPU
emotion_recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew", device="cuda")


# ==========================================
# OLLAMA — real multi-turn chat, not one-shot generation
# ==========================================

SYSTEM_PROMPT = (
    """You are a warm, friendly chat companion having a live conversation. 
    Keep replies concise and natural. You can sometimes see who you're 
    talking to and what emotion their face is currently showing; use that 
    naturally when it's relevant, but don't mention it every message. When responding, always continue the existing discussion. Continue the conversation in a friendly, empathetic way, and ask questions to keep it going unless a new person comes into view. Avoid repeating yourself or asking the same questions over and over.
    When emotion information is available, use it as background context. Always end with a relevant question unless the user has asked for a factual answer."""
)


def build_fallback_reply(history):
    last_user = ""
    for item in reversed(history):
        if item.get("role") == "user":
            last_user = item.get("content", "")
            break

    if any(word in last_user.lower() for word in ["sad", "hurt", "alone", "depressed", "upset", "anxious", "stressed", "worried"]):
        return "I’m here with you. Want to tell me what’s been weighing on you?"
    return "I’m here and listening. Tell me what’s on your mind."


def ask_ollama_chat(history):
    """history is a list of {'role': 'system'|'user'|'assistant', 'content': str}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": history,
                "stream": False,
                "options": {"num_predict": 2000}
            },
            timeout=40,
        )
        response.raise_for_status()
        payload = response.json()
        print(response.json())
        content = payload.get("message", {}).get("content", "")

        return content.strip() or build_fallback_reply(history)
    except Exception as e:
        print("Ollama chat error:", repr(e))
        return build_fallback_reply(history)


# ==========================================
# LOAD KNOWN FACES
# ==========================================

print("Loading known faces...")

known_embeddings = []
known_names = []

DATASET = os.path.join(os.path.dirname(__file__), "data")

if os.path.isdir(DATASET):
    for file in os.listdir(DATASET):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(DATASET, file)
            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=True,
                    align=True,
                )
                known_embeddings.append(np.array(embedding[0]["embedding"]))
                known_names.append(os.path.splitext(file)[0])
                print("Loaded:", file)
            except Exception as e:
                print(file, e)

known_matrix = np.vstack(known_embeddings) if known_embeddings else np.empty((0, 512))
RECOGNITION_THRESHOLD = 0.68

print("Faces loaded:", len(known_names))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

DETECTION_WIDTH = 640


# ==========================================
# CORE PROCESSING (runs in threadpool, sync/blocking)
# ==========================================

def decode_frame(b64_jpeg: str) -> np.ndarray:
    header_free = b64_jpeg.split(",")[-1]  # strip "data:image/jpeg;base64," if present
    img_bytes = base64.b64decode(header_free)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_largest_face(frame):

    h0, w0 = frame.shape[:2]
    scale = DETECTION_WIDTH / w0
    small = cv2.resize(frame, (DETECTION_WIDTH, int(h0 * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    x, y, w, h = int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale)
    return (x, y, w, h)


def padded_crop(frame, box):
    x, y, w, h = box
    pad_x, pad_y = int(w * 0.4), int(h * 0.4)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    return frame[y1:y2, x1:x2]


def recognize_face(padded_face) -> str:

    try:
        result = DeepFace.represent(
            img_path=padded_face,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
        )
        current_embedding = np.array(result[0]["embedding"]).reshape(1, -1)

        if known_matrix.shape[0] == 0:
            return "Unknown"

        sims = cosine_similarity(current_embedding, known_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]

        return known_names[best_idx] if best_score > RECOGNITION_THRESHOLD else "Unknown"

    except Exception:
        return "Unknown"


def analyze_emotion(padded_face) -> str:
    try:
        print("Emotion prediction starting")
        rgb_face = cv2.cvtColor(padded_face, cv2.COLOR_BGR2RGB)
        emotion, _scores = emotion_recognizer.predict_emotions(rgb_face, logits=True)
        return emotion.lower()
    except Exception as e:
        print("Emotion error:", repr(e))
        traceback.print_exc()
        return ""



def process_frame(frame):
    """Full pipeline for one frame. Returns dict result. Blocking — run in executor."""

    box = detect_largest_face(frame)
    
    if box is None:
        return {"box": None, "name": "Unknown", "emotion": ""}

    face = padded_crop(frame, box)
    name = recognize_face(face)
    emotion = analyze_emotion(face)

    x, y, w, h = box
    return {"box": {"x": x, "y": y, "w": w, "h": h}, "name": name, "emotion": emotion}


# ==========================================
# WEBSOCKET
# ==========================================
#
# Two kinds of client messages come over the same socket, distinguished by key:
#   { "frame": "<base64 jpeg>" }   -> a webcam tick, runs vision pipeline
#   { "message": "<user text>" }  -> the user typed something in the chat box
#
# Two kinds of server messages go back, distinguished by "kind":
#   { "kind": "vision", box, name, emotion, reply }   reply is null unless
#       the ambient name/emotion-change trigger fired this round
#   { "kind": "chat", reply }   a direct reply to something the user typed

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connect requested")

    await websocket.accept()
    print("WebSocket accepted")

    loop = asyncio.get_event_loop()

    # Conversation memory for this connection — this is what makes it feel
    # like a real chatbot instead of disconnected one-off generations.
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    current_person = None
    current_emotion = None

    face_history = []
    emotion_history = []

    WINDOW_SIZE = 30

    last_name = ""
    last_emotion = ""
    last_chat_time = 0.0
    chat_delay = 30
    emotion_history =[]

    try:
        while True:
            print("Waiting for client message...")
            payload = await websocket.receive_json()
            # print("FULL PAYLOAD:", payload)
            try:
                # ---- typed chat message ----
                if "message" in payload:
                    print("User message received")
                    user_text = payload["message"].strip()
                    print("User text:", user_text)

                    if not user_text:
                        continue

                    history.append({"role": "user", "content": user_text})
                    history = trim_history(history)
                    print("User:", user_text)
                    print("Sending to Ollama")
                    reply = await loop.run_in_executor(chat_executor, ask_ollama_chat, history)
                    print("Ollama reply:", reply)
                    history.append({
                        "role": "assistant",
                        "content": reply
                    })

                    history = trim_history(history)

                    current_person = stable_name
                    current_emotion = stable_emotion

                    last_chat_time = time.time()
                    await websocket.send_json({
                        "kind": "vision",
                        "box": result["box"],
                        "name": stable_name,
                        "emotion": stable_emotion,
                        "reply": reply,
                    })

                # ---- webcam frame tick ----
                if "frame" in payload:
                    
                    frame = decode_frame(payload["frame"])
                    start = time.time()
                    result = await loop.run_in_executor(vision_executor, process_frame, frame)
                    print("Frame processing:", time.time() - start)
                    name = result["name"]
                    emotion = result["emotion"]

                    if name != "Unknown":
                        face_history.append(name)

                    if emotion:
                        emotion_history.append(emotion)

                    face_history = face_history[-WINDOW_SIZE:]
                    emotion_history = emotion_history[-WINDOW_SIZE:]

                    stable_name = (
                        max(set(face_history), key=face_history.count)
                        if face_history else "Unknown"
                    )

                    stable_emotion = (
                        max(set(emotion_history), key=emotion_history.count)
                        if emotion_history else ""
                    )
                    person_changed = (
                        current_person is not None
                        and stable_name != current_person
                    )

                    state_changed = (
                        stable_name != current_person
                        or stable_emotion != current_emotion
                    )
                    if person_changed:
                        history = [
                            {"role": "system", "content": SYSTEM_PROMPT}
                        ]
                    reply = None

                    if (
                        result["box"] is not None
                        and state_changed
                        and (time.time() - last_chat_time > chat_delay)
                    ):
                        last_chat_time = time.time()
                        current_person = stable_name
                        current_emotion = stable_emotion
                        if current_person != "Unknown":
                            history.append(
                                {
                                    "role": "system",
                                    "content": f"The person in view is {current_person} and they are showing {current_emotion} emotion.",
                                }
                            )
                        else:
                            history.append(
                                {
                                    "role": "system",
                                    "content": f"The person in view is unknown and they are showing {current_emotion} emotion.",
                                }
                            )
                        history = trim_history(history)
                        reply = await loop.run_in_executor(chat_executor, ask_ollama_chat, history)
                        history.append({"role": "assistant", "content": reply})
                    await websocket.send_json({
                        "kind": "vision",
                        "box": result["box"],
                        "name": name,
                        "emotion": emotion,
                        "reply": reply,
                    })

            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                # A single bad frame or a hiccup from Ollama/DeepFace should never
                # kill the whole connection — log it and keep the loop alive.
                traceback.print_exc()
                try:
                    await websocket.send_json({"kind": "error", "message": str(e)})
                except Exception:
                    pass

    except WebSocketDisconnect as e:

        print("Client disconnected:", e)