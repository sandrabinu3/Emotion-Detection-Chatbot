# Emotion Chatbot Webapp

Split-screen webapp: left half is your live webcam feed with a face-tracking
overlay, right half is a chat panel showing what the assistant says as it
recognizes you and reads your emotion.

## How it works

- **Frontend (React + Vite)**: captures your webcam via `getUserMedia`,
  grabs a JPEG frame every ~700ms, and sends it to the backend over a
  WebSocket. It draws the bounding box on a canvas overlay and renders
  replies as chat bubbles.
- **Backend (FastAPI)**: receives frames over `/ws`, runs Haar face
  detection, then (in a background thread pool so it never blocks other
  connections) runs ArcFace recognition + emotion analysis, and calls
  Ollama for a reply when your name or emotion changes. Sends the result
  back as JSON.

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your known-face reference photos in `backend/data/` (e.g.
`data/Sandra.jpg`), one clear face photo per person, filename = their name.

Make sure Ollama is running locally with the `llama3.2` model pulled:

```bash
ollama pull llama3.2
```

Run the backend:

```bash
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (usually `http://localhost:5173`). Your browser
will ask for camera permission — allow it.

## Notes

- The backend only processes one frame at a time per connection (frames
  are dropped, not queued, if it's still busy) — this keeps things
  real-time rather than laggy under load.
- If recognition seems off, check the enrollment photos in `backend/data/`
  are clear, front-facing, and well-lit — accuracy depends heavily on
  those reference embeddings.
- CORS is currently locked to `http://localhost:5173` in `main.py` — update
  `allow_origins` there if you serve the frontend elsewhere.