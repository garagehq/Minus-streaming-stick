# Minus API Reference

This document describes the HTTP API endpoints provided by the Minus Web UI.

**Base URL:** `http://<hostname>:80` (configurable via `--webui-port`)

---

## Status Endpoints

### GET /api/status

Get current system status including blocking state, detection info, and uptime.

**Response:**
```json
{
  "blocking": false,
  "blocking_source": null,
  "paused": false,
  "paused_remaining": 0,
  "uptime": 3600.5,
  "fps": 30.0,
  "hdmi": {
    "signal": true,
    "resolution": "3840x2160@60Hz"
  }
}
```

### GET /api/health

Detailed health check for monitoring systems.

### GET /api/metrics

Prometheus-compatible metrics endpoint.

**Response (text/plain):**
```
# HELP minus_uptime_seconds Time since service start
# TYPE minus_uptime_seconds gauge
minus_uptime_seconds 3600.5

# HELP minus_video_fps Current video FPS
# TYPE minus_video_fps gauge
minus_video_fps 30.0

# HELP minus_blocking_active Whether blocking is active
# TYPE minus_blocking_active gauge
minus_blocking_active 0

# HELP minus_video_restarts_total Total video pipeline restarts
# TYPE minus_video_restarts_total counter
minus_video_restarts_total 0

# HELP minus_audio_running Whether audio is running
# TYPE minus_audio_running gauge
minus_audio_running 1

# HELP minus_audio_muted Whether audio is muted
# TYPE minus_audio_muted gauge
minus_audio_muted 0

# HELP minus_hdmi_signal Whether HDMI signal is present
# TYPE minus_hdmi_signal gauge
minus_hdmi_signal 1

# HELP minus_time_saved_seconds Total time saved by blocking ads
# TYPE minus_time_saved_seconds counter
minus_time_saved_seconds 120.5
```

**Query Parameters:**
- `simple=1` - Return simple `{status, timestamp}` for uptime monitors

**Response:**
```json
{
  "status": "ok",
  "service": "minus",
  "timestamp": 1234567890.123,
  "subsystems": {
    "video": {"status": "ok", "fps": 30.0, "blocking": false, "restart_count": 0},
    "audio": {"status": "ok", "muted": false, "restart_count": 0},
    "vlm": {"status": "ok"},
    "ocr": {"status": "ok"},
    "fire_tv": {"status": "connected"}
  }
}
```

**Status Codes:**
- `200 OK` - Service healthy
- `500 Internal Server Error` - Service error

---

## Control Endpoints

### POST /api/pause/{minutes}

Pause ad blocking for specified duration.

**Parameters:**
- `minutes` (int, required): Duration in minutes (1-60)

**Response:**
```json
{
  "success": true,
  "paused_until": 1234567890,
  "duration_minutes": 5
}
```

**Errors:**
- `400 Bad Request` - Invalid duration (must be 1-60)

### POST /api/resume

Resume ad blocking immediately.

**Response:**
```json
{
  "success": true
}
```

---

## Video Control

### GET /api/video/color

Get current video color balance settings.

**Response:**
```json
{
  "saturation": 1.25,
  "brightness": 0.0,
  "contrast": 1.0,
  "hue": 0.0
}
```

### POST /api/video/color

Set video color balance settings.

**Request Body:**
```json
{
  "saturation": 1.3,
  "brightness": 0.1,
  "contrast": 1.0,
  "hue": 0.0
}
```

**Parameter Ranges:**
- `saturation`: 0.0-2.0 (default 1.0)
- `brightness`: -1.0 to 1.0 (default 0.0)
- `contrast`: 0.0-2.0 (default 1.0)
- `hue`: -1.0 to 1.0 (default 0.0)

**Response:**
```json
{
  "success": true,
  "saturation": 1.3,
  "brightness": 0.1,
  "contrast": 1.0,
  "hue": 0.0
}
```

**Errors:**
- `400 Bad Request` - Invalid parameter values

### POST /api/video/restart

Force restart the video pipeline.

**Response:**
```json
{
  "success": true,
  "message": "Video pipeline restart initiated"
}
```

---

## Detection Testing

### POST /api/ocr/test

Run OCR on current frame (without saving screenshot).

**Response:**
```json
{
  "success": true,
  "texts": ["Skip Ad", "Learn More"],
  "is_ad": true,
  "skip_info": {
    "skippable": true,
    "text": "Skip Ad",
    "countdown": 0
  }
}
```

### POST /api/vlm/test

Run VLM on current frame (without saving screenshot).

**Response:**
```json
{
  "success": true,
  "is_ad": true,
  "response": "Yes",
  "confidence": 0.95,
  "latency_ms": 850
}
```

### POST /api/test/trigger-block

Trigger blocking for testing purposes.

**Request Body:**
```json
{
  "duration": 10,
  "source": "ocr"
}
```

**Parameters:**
- `duration` (int): Seconds to block (1-60, default 10)
- `source` (string): Detection source - `ocr`, `vlm`, `both`, or `default`

**Response:**
```json
{
  "success": true,
  "duration": 10,
  "source": "ocr"
}
```

**Errors:**
- `400 Bad Request` - Invalid duration or source

### POST /api/test/stop-block

Stop blocking immediately (for testing).

**Response:**
```json
{
  "success": true
}
```

---

## Fire TV Control

### GET /api/fire-tv/status

Get Fire TV connection status.

**Response:**
```json
{
  "connected": true,
  "ip_address": "192.168.1.100"
}
```

### POST /api/blocking/skip

Trigger skip button press on Fire TV.

**Response:**
```json
{
  "success": true,
  "action": "skip"
}
```

---

## Audio Control

### GET /api/audio/status

Get audio passthrough status.

**Response:**
```json
{
  "running": true,
  "muted": false,
  "restart_count": 0
}
```

### POST /api/audio/sync-reset

Reset audio/video sync (causes ~300ms audio dropout).

**Response:**
```json
{
  "success": true
}
```

---

## Settings

### GET /api/preview/enabled
### POST /api/preview/enabled

Get/set ad preview window visibility.

**Response/Request:**
```json
{
  "enabled": true
}
```

### GET /api/debug-overlay/enabled
### POST /api/debug-overlay/enabled

Get/set debug overlay visibility.

**Response/Request:**
```json
{
  "enabled": true
}
```

---

### GET /api/contribute/status
### POST /api/contribute/enable
### POST /api/contribute/disable

Anonymous ad-snapshot contribution (opt-in, default off — see [PRIVACY.md](PRIVACY.md)).

`enable` requires an explicit consent body; anything else is refused with 400:

```bash
curl -X POST http://minus.local/api/contribute/enable -H 'Content-Type: application/json' -d '{"consent": true}'
```

`disable` stops uploads and deletes the local queue. Status / response:

```json
{"available": true, "enabled": true, "queued": 2, "uploaded_this_session": 9,
 "retracted_this_session": 1, "last_upload_at": 1760560000.0, "last_error": null,
 "cooldown_minutes": 10.0, "endpoint": "https://minus-ingest-garage.fly.dev/ingest"}
```

## Streaming

### GET /stream

MJPEG video stream (proxied from ustreamer).

**Content-Type:** `multipart/x-mixed-replace; boundary=frame`

### GET /snapshot

Current frame as JPEG (proxied from ustreamer).

**Content-Type:** `image/jpeg`

---

## Detection History

### GET /api/detections

Get recent detection history.

**Query Parameters:**
- `limit` (int): Maximum entries to return (default 50)

**Response:**
```json
{
  "detections": [
    {
      "timestamp": 1234567890.123,
      "source": "ocr",
      "is_ad": true,
      "text": "Skip Ad"
    }
  ]
}
```

---

## Logs

### GET /api/logs

Get recent log entries.

**Query Parameters:**
- `lines` (int): Number of lines to return (default 100)

**Response:**
```json
{
  "logs": [
    "2026-04-02 10:00:00 [INFO] Ad detected via OCR",
    "2026-04-02 10:00:01 [INFO] Blocking started"
  ]
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error description"
}
```

**Common Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `500 Internal Server Error` - Server error

---

*Last updated: 2026-04-02*
