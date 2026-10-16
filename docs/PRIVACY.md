# Privacy

Minus runs every model **on the device**: OCR on the RK3588 NPU, the vision
model on the Axera NPU, speech recognition on the CPU. By default nothing you
watch ever leaves your living room: no frames, no audio, no viewing history,
no telemetry, no account.

## Anonymous ad-snapshot contribution (opt-in, default OFF)

To help improve the model you can opt in from the web UI: **Settings → Help
Improve Minus → Contribute anonymous ad snapshots**. The switch only takes
effect after you accept a dialog that says exactly what is shared; the API
refuses to turn it on without that explicit consent. When it is on, and only
then:

- When Minus **blocks an ad**, up to **3 frames from that ad** (at least 8 s
  apart, near-duplicates skipped) are queued on the box, downscaled so the
  longest side is 960 px.
- Each frame carries only: the vision model's ad probability (empty when only
  OCR fired), which detector fired (`minus-tv/vlm` or `minus-tv/ocr`), the
  image size, `host: "tv"`, `verdict: "ad"`, and a fresh random id. The id
  is new for every frame, so frames can't be linked to each other or to you.
- Frames wait in a **10-minute local cool-down** before upload. If you
  **pause blocking** during that ad (or within a minute of it ending), every
  queued frame from that ad is **deleted and never uploaded**: a pause means
  the block wasn't wanted, and we don't collect what you didn't want blocked.
- Uploads go to the same ingest server as the [Minus Chrome
  extension](https://github.com/garagehq/Minus-chrome-extension)
  (`server/ingest-server.mjs` there is the reference implementation). The
  training dataset is a private Hugging Face dataset owned by the Minus
  project; write credentials live on the ingest server, never on the box.
- **Never sent:** the device's name, IP address, serial number, Wi-Fi details,
  streaming-device type, accounts, the app or channel you're watching, when
  you watch, on-screen text, audio, or any frame that wasn't blocked as an
  ad. No cookies and no identifier of any kind.
- Turning it **off** stops uploads at once and **deletes everything still
  waiting** in the local queue (`~/.minus_contrib_queue/`).

The queue is capped at 200 frames and 7 days, so a box that is offline or
behind a firewall never fills its disk. If the server can't be reached,
Minus retries with backoff (20 s up to 6 h) and blocks ads exactly the same.

Other settings: `MINUS_INGEST_URL`, `MINUS_INGEST_KEY` (point at your own
ingest server, or set the URL empty to disable uploads entirely),
`MINUS_CONTRIB_QUEUE_DIR`. The implementation is in
[`src/contribute.py`](../src/contribute.py).

## Local screenshots

Separately from the above, Minus keeps training screenshots **on the box**
(`screenshots/ads`, `non_ads`, `static`, `vlm_spastic`) for the review screen in the
web UI. Those are never uploaded by Minus; they stay on the device unless you
copy them off yourself.
