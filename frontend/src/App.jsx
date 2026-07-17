import { useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/ws'
const CAPTURE_INTERVAL_MS = 700

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)      // hidden canvas, used to grab JPEG frames
  const overlayRef = useRef(null)     // visible canvas, draws the bounding box
  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)

  const [connected, setConnected] = useState(false)
  const [name, setName] = useState('Unknown')
  const [emotion, setEmotion] = useState('')
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)

  // --- webcam setup ---
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 960, height: 720 } })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream
      })
      .catch((err) => console.error('Camera error:', err))
  }, [])

  // --- websocket setup ---
  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.kind === 'vision') {
        setName(data.name || 'Unknown')
        setEmotion(data.emotion || '')
        drawBox(data.box, data.name, data.emotion)

        if (data.reply) {
          setSending(false)
          setMessages((prev) => [
            ...prev,
            { from: 'bot', name: data.name, emotion: data.emotion, text: data.reply, ts: Date.now() },
          ])
        }
        return
      }

      if (data.kind === 'chat') {
        setSending(false)
        setMessages((prev) => [
          ...prev,
          { from: 'bot', name: data.name || name, emotion: data.emotion || emotion, text: data.reply, ts: Date.now() },
        ])
      }
    }

    return () => ws.close()
  }, [])

  // --- capture loop: grab a frame, send as base64 JPEG ---
  useEffect(() => {
    const interval = setInterval(() => {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ws = wsRef.current

      if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return
      if (video.videoWidth === 0) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0)

      const frame = canvas.toDataURL('image/jpeg', 0.6)
      ws.send(JSON.stringify({ frame }))
    }, CAPTURE_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [])

  // --- auto-scroll chat to newest message ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function sendMessage() {
    const ws = wsRef.current
    const text = draft.trim()
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return

    setMessages((prev) => [...prev, { from: 'user', text, ts: Date.now() }])
    ws.send(JSON.stringify({ message: text }))
    setDraft('')
    setSending(true)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // --- draw viewfinder-style bracket box on the overlay canvas ---
  function drawBox(box, boxName, boxEmotion) {
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!overlay || !video) return

    overlay.width = video.clientWidth
    overlay.height = video.clientHeight
    const ctx = overlay.getContext('2d')
    ctx.clearRect(0, 0, overlay.width, overlay.height)

    if (!box || video.videoWidth === 0) return

    const scaleX = overlay.width / video.videoWidth
    const scaleY = overlay.height / video.videoHeight
    const x = box.x * scaleX
    const y = box.y * scaleY
    const w = box.w * scaleX
    const h = box.h * scaleY
    const c = 18 // bracket corner length

    ctx.strokeStyle = '#e8a33d'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'

    const corners = [
      [[x, y + c], [x, y], [x + c, y]],
      [[x + w - c, y], [x + w, y], [x + w, y + c]],
      [[x, y + h - c], [x, y + h], [x + c, y + h]],
      [[x + w - c, y + h], [x + w, y + h], [x + w, y + h - c]],
    ]
    corners.forEach((pts) => {
      ctx.beginPath()
      ctx.moveTo(pts[0][0], pts[0][1])
      pts.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]))
      ctx.stroke()
    })

    ctx.font = '600 15px "JetBrains Mono", monospace'
    ctx.fillStyle = '#e8a33d'
    ctx.fillText(`${boxName}${boxEmotion ? '  ·  ' + boxEmotion : ''}`, x, y - 10)
  }

  return (
    <div className="app">
      <div className="video-pane">
        <video ref={videoRef} autoPlay playsInline muted className="video-feed" />
        <canvas ref={overlayRef} className="overlay" />
        <canvas ref={canvasRef} className="hidden-canvas" />

        <div className="status-bar">
          <span className={`status-dot ${connected ? 'live' : ''}`} />
          {connected ? 'LIVE' : 'CONNECTING…'}
        </div>
      </div>

      <div className="chat-pane">
        <header className="chat-header">
          <h1>Reader</h1>
          <p className="chat-sub">
            {name !== 'Unknown' ? name : 'No one recognized'}
            {emotion && <span className="emotion-tag"> — {emotion}</span>}
          </p>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-state">Say hi, or just look at the camera…</p>
          )}
          {messages.map((m) => (
            <div key={m.ts} className={`message ${m.from === 'user' ? 'message-user' : 'message-bot'}`}>
              {m.from === 'bot' && (
                <div className="message-meta">
                  <span className="message-name">{m.name}</span>
                  {m.emotion && <span className="message-emotion">{m.emotion}</span>}
                </div>
              )}
              <p className="message-text">{m.text}</p>
            </div>
          ))}
          {sending && (
            <div className="message message-bot message-typing">
              <p className="message-text">···</p>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <textarea
            className="composer-input"
            placeholder="Type a message…"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button className="composer-send" onClick={sendMessage} disabled={!draft.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  )
}