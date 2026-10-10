# TOUCH GRASS 🌳🎮
An AI-powered real-world scavenger hunt for solo and multiplayer play.

## What is Touch Grass?
Touch Grass is an outdoor scavenger hunt designed to get players outside and interacting with their environment.

A player can start alone or create a lobby and share a short game code. The AI Game Master (powered by **Gemma**) uses nearby OpenStreetMap evidence to rank supported physical objects before selecting five realistic targets (e.g., Bicycle, Bench, Stop Sign).

Players must physically go outside and find these objects. When they spot one, they take a photo with their phone camera. A local **YOLO** computer vision model verifies if the object is actually in the photo. The first player to verify all five objects wins.

## Why it exists
This project was built for the Hacktoberfest Weekend Challenge, specifically focusing on the theme "Touch Grass". It demonstrates how open-source AI can be the core driver of an interactive, physical-world experience, taking people away from staring at screens all day.

## How the game works
1. **Create/Join**: A player creates a game on their phone and shares the code.
2. **Start**: The host starts the game.
3. **Generation**: The backend uses the host's latest valid location estimate (up to the configured 250 m accuracy) for bounded nearby OpenStreetMap lookups. Retrieved feature categories are ranked against outdoor classes supported by YOLO, then passed to Gemma to select five suitable targets. If useful map context is unavailable, the UI and prompt explicitly identify approximate/general outdoor context instead.
   When the same coarse OpenStreetMap context is used again, classes selected in at least two of its three most recent games are temporarily removed from Gemma's candidate list when five positively ranked supported alternatives remain. This reduces repetitive sets without inventing objects or changing the location evidence.
4. **Hunt**: Players walk around outside to find the objects.
5. **Validation**: When a player takes a picture, the backend requires a recent, valid GPS estimate (within the configured accuracy ceiling) and uses YOLO's matching-class confidence check to verify the object. There is no distance limit from the game's starting location.
6. **Win**: The first player to complete all five targets wins. Movement is not a win condition.

## Why open-source/open-weight AI is central
Open-source AI is not an optional feature here; it is the game's intelligence.

- **Gemma (Game Master)**: Rather than using a rigid, hardcoded list of objects, we use an instruction-tuned Gemma model running locally. Gemma acts as a creative Game Master, generating realistic, contextual targets while avoiding impossible or dangerous objects.
- **YOLO (Visual Verification)**: A local YOLOv8 model acts as the referee, verifying player submissions in real-time.

### Benefits of this architecture:
- **Local Inference**: Both models can run entirely locally without depending on costly proprietary APIs.
- **Privacy**: Player photos and locations are processed by your own backend and AI models. They are never sent to third-party closed APIs. Photos are used only for verification and can be discarded immediately.
- **Transparency**: You control exactly what the AI models are doing and how they evaluate submissions.
- **Cost**: Running this at a community scale requires no per-API-call billing.

## Local Setup

### Requirements
- Python 3.12
- MongoDB Atlas (or local MongoDB)
- Ollama (for running Gemma locally)

### Step-by-step (Windows PowerShell)

1. **Clone & Environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and configure your MongoDB URI.
   ```powershell
   copy .env.example .env
   ```

3. **Install Ollama and Gemma**
   Download [Ollama](https://ollama.com/) and install it. Then pull the Gemma model:
   ```powershell
   ollama run gemma3:1b
   ```
   *Note: `gemma3:1b` is a small model suitable for local development.*

4. **YOLO Setup**
   The `yolov8n.pt` model will be automatically downloaded by the `ultralytics` package the first time you run the backend.
   To verify that the configured model loads and performs real inference without changing game state, run this optional smoke test from the project root:
   ```powershell
   $env:PYTHONPATH = "backend"
   $env:RUN_YOLO_SMOKE_TEST = "1"
   python -m pytest backend/tests/test_yolo_smoke.py
   ```

5. **Run the Backend**
   From the project root:
   ```powershell
   $env:PYTHONPATH = "backend"
   uvicorn app.main:app --reload
   ```

6. **Play**
   Open your browser (or your phone connected to the same network) to `http://localhost:8000`.

*Note*: If you just want to test the UI without loading heavy AI models, set `DEMO_MODE=true` in your `.env` file. The UI will clearly identify demo mode. Target selection and image verification are simulated and do not represent real AI verification.

## Deploy to Render

The root [`render.yaml`](./render.yaml) Blueprint defines the public FastAPI web service and a separate private Ollama service. FastAPI continues to serve the existing `frontend/` directory; the Render health check uses the existing `/api/health` route. The Ollama service is private to the Render network, listens on port `11434`, and stores models on a persistent disk. Its startup entrypoint waits for Ollama, idempotently pulls `gemma3:1b`, and verifies the model before startup completes. The Dockerfile pins the official Ollama image to `0.40.2`.

### Resource and cost assumptions

The Blueprint uses Render's currently documented compute plan IDs (October 2026; see [Render compute pricing](https://render.com/pricing#compute) and [persistent disk details](https://render.com/docs/disks)):

- **FastAPI / YOLO:** `1c-2g` (1 CPU, 2 GB RAM), approximately **$25/month** at continuous operation. This is a paid plan because the Python/PyTorch/Ultralytics runtime and CPU inference need more headroom than a small/free instance; the service is configured for one Uvicorn worker to avoid duplicating the loaded detector in RAM.
- **Ollama / Gemma:** `2c-4g` (2 CPU, 4 GB RAM), approximately **$85/month** at continuous operation. This gives the 1B model additional RAM for the model runtime and inference context while avoiding selecting a larger plan without workload measurements.
- **Ollama model disk:** 5 GB at **$0.25/GB/month**, approximately **$1.25/month**. The 5 GB allocation leaves room for the model and its metadata.

The estimated always-on Render total is **about $111.25/month**, before taxes, build-pipeline usage, data transfer, or MongoDB Atlas charges. Render bills compute prorated by runtime; confirm the plan and current total in the Blueprint review screen before applying. If observed memory or inference latency requires more capacity, resize only after measuring production-like load.

### Prepare services and secrets

1. Push this project to a Git provider repository yourself. The deployment workspace must contain a Git repository for Render to connect; the current local project directory has no `.git` metadata. No code is pushed or deployed automatically by this setup.
2. Create/verify a MongoDB Atlas database and database user. Permit the Render web service's outbound IP ranges in Atlas Network Access; avoid opening the database to all IPs. Keep the Atlas URI out of source control.
3. In Render, select **New → Blueprint**, connect the repository, and select the root `render.yaml`. Both services are placed in `oregon`; if you choose another supported region, set the same region for both services before applying.
4. Review the displayed compute and disk charges. The Blueprint prompts for `MONGODB_URI` and `GEOCONTEXT_USER_AGENT`. Enter the Atlas URI securely in Render, and set the latter to an identifying value such as `TouchGrassGame/1.0 (contact: your-maintainer-email)`. Nominatim/OSM requests need an identifying User-Agent; respect the upstream usage policy and rates.
5. Apply the Blueprint. The Ollama service becomes reachable only over Render's private network. The web service obtains its internal `host:port` through a Blueprint service reference; the app normalizes that to HTTP while retaining `http://localhost:11434` as the local default. The FastAPI build installs `backend/requirements.txt` and downloads/loads `yolov8n.pt` into the deploy artifact, rather than relying on a model file from the developer's machine.
6. After deployment, open the web service URL over HTTPS. Confirm `/api/health` reports `status: ok` and `/api/health/ai` reports Ollama reachable with the model available. These endpoints do not validate MongoDB connectivity: test a real create/start flow after confirming that Atlas networking and credentials work. Then test GPS, camera capture, real target verification, voting, and multiplayer on devices with permissions enabled.

### Environment variables

The Blueprint sets production-specific values and asks for secrets through Render's secure dashboard prompt; no secrets are stored in this repository. Application settings recognized by `backend/app/config.py` are:

| Variable | Deployment value / default |
| --- | --- |
| `MONGODB_URI` | Required secret; set in Render to the Atlas connection URI. |
| `MONGODB_DATABASE` | `touch_grass` |
| `DEMO_MODE` | `false` (required for real Gemma and YOLO). |
| `GEMMA_MODEL` | `gemma3:1b` |
| `YOLO_MODEL` | `yolov8n.pt` (downloaded during the web service build). |
| `OLLAMA_URL` | Render Blueprint reference to the private service `hostport`; local default remains `http://localhost:11434`. |
| `GEOCONTEXT_USER_AGENT` | Required at deployment; set an app-identifying string with a maintainer contact. |
| `LOCATION_MAX_AGE_SECONDS` | `60` |
| `GPS_MAX_ACCURACY_METERS` | `250` |
| `GPS_HIGH_CONFIDENCE_METERS` | `50` |
| `GPS_MOVEMENT_MAX_ACCURACY_METERS` | `250` |
| `GPS_MIN_SEGMENT_METERS` | `25` |
| `GPS_MOVEMENT_NOISE_FLOOR_METERS` | `15` |
| `GPS_MOVEMENT_STABILITY_METERS` | `35` |
| `GPS_MOVEMENT_CONFIRMATIONS` | `2` |
| `GPS_MAX_SPEED_METERS_PER_SECOND` | `12` |
| `MAX_UPLOAD_MB` | `10` |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.60` |
| `TARGET_VOTE_INACTIVE_SECONDS` | `90` |
| `GEOCONTEXT_PROVIDER` | `openstreetmap` |
| `GEOCONTEXT_NOMINATIM_URL` | `https://nominatim.openstreetmap.org/reverse` |
| `GEOCONTEXT_OVERPASS_URL` | `https://overpass-api.de/api/interpreter` |
| `GEOCONTEXT_RADIUS_METERS` | `1200` |
| `GEOCONTEXT_TIMEOUT_SECONDS` | `5` |
| `GEOCONTEXT_RETRIES` | `1` |
| `GEOCONTEXT_CACHE_TTL_SECONDS` | `21600` |
| `GEOCONTEXT_REFRESH_DISTANCE_METERS` | `200` |
| `GEOCONTEXT_REFRESH_COOLDOWN_SECONDS` | `60` |
| `GEOCONTEXT_REFRESH_STABILITY_METERS` | `150` |
| `GEOCONTEXT_REFRESH_CONFIRMATIONS` | `2` |

`PYTHON_VERSION=3.12.10` selects the Render Python build runtime. The separate Ollama container uses `OLLAMA_HOST=0.0.0.0:11434`; do not create a public Ollama web service or publish port `11434`.

## Location context and movement
- OpenStreetMap reverse-geocoding and nearby-feature endpoints are configurable with `GEOCONTEXT_PROVIDER`, `GEOCONTEXT_NOMINATIM_URL`, and `GEOCONTEXT_OVERPASS_URL`. The default provider makes one bounded nearby-feature query and one coarse reverse lookup, retries a failed request once, and caches successful results by a roughly 100 m location cell for six hours. Set a descriptive `GEOCONTEXT_USER_AGENT` when deploying, and follow the public provider's usage policy.
- When OpenStreetMap evidence is used, the UI displays attribution to OpenStreetMap contributors. The map lookup uses rounded coordinates; precise player coordinates remain server-side and are not included in Gemma's prompt or other players' responses.
- The initial game location is not a boundary. The configured `GPS_MAX_ACCURACY_METERS=250` limits accepted location estimates. Stable, validated fixes continue to update OSM context and area-aware targets; GPS is independent of optional motion sensor statistics.
- The game context refreshes after the player is stably at least `GEOCONTEXT_REFRESH_DISTANCE_METERS=200` from the current context anchor. `GEOCONTEXT_REFRESH_STABILITY_METERS=150`, `GEOCONTEXT_REFRESH_CONFIRMATIONS=2`, and `GEOCONTEXT_REFRESH_COOLDOWN_SECONDS=60` limit jitter-driven regeneration and provider traffic. Refreshed context and targets are stored on the shared game document; completed objectives are retained while unfinished objectives are replaced with Gemma-generated supported classes from the new area evidence.
- GPS location is used for current-area context, not as a win condition.

## Optional activity statistics
- On supported phones, select **Start Step Counter** in the active game and grant motion permission when prompted. The browser counts filtered accelerometer peaks as an estimated step count; it is not a medically accurate or OS-certified pedometer. Tracking stops when leaving the game and pauses when the page is backgrounded.
- GPS route distance is shown only from consecutive accepted fixes with accuracy of 15 m or better and plausible travel time; uncertain fixes break the distance segment. It is an estimate, not survey-grade distance.
- Step-derived distance is shown separately as an approximation based on the editable stride-length assumption (default 0.75 m per step). Adjust it to your own stride. It is never merged with GPS distance.
- To test on a phone, serve the app over HTTPS (camera, location and motion APIs require a secure context; localhost is the development exception). Open the HTTPS page on the phone, grant camera and location permissions, then tap **Start Step Counter** and allow motion access. On iOS, use Safari and accept its motion-sensor prompt; on Android, allow site motion/sensor access if prompted. Keep the page foregrounded while counting; backgrounding pauses tracking. Sensor availability, event cadence and thresholds vary by device, so verify against walking by hand—synthetic unit tests do not establish physical-phone accuracy.

## Unanimous target changes
- Players can request replacement only for an unfinished target. The requester votes approve; every teammate with a heartbeat in the last `TARGET_VOTE_INACTIVE_SECONDS` (default 90 seconds) is in the roster. A fresh vote action refreshes that roster so disconnected players cannot block indefinitely. Pending approvals can be changed to rejection, which immediately cancels the proposal.
- Votes are stored on the game record. A unanimous vote asks Gemma for one target using the game's current area context and classes exposed by the active YOLO weights. The server atomically swaps only that target and records the approved proposal; it never awards completion for the swap. If no valid distinct class can be generated, the original target remains and the game reports why.

## Privacy & Location
- The browser Geolocation API is used as an anti-cheat signal.
- Exact player coordinates are NEVER exposed to other players.
- For OpenStreetMap lookups, the backend sends rounded coordinates (roughly 100 m grid precision) to the configured provider. The resulting coarse context and candidate rationale are cached and passed to Gemma; exact coordinates are not sent to Gemma or included in application logs.
- `YOLO_CONFIDENCE_THRESHOLD` configures the minimum matching-class confidence (default `0.60`).
- `YOLO_MODEL` selects the real detector weights; it defaults to `yolov8n.pt`. For an evaluated deployment, set it to `yolo11s.pt`; to return to the current model, set it back to `yolov8n.pt`. No detector default or class threshold is changed without a representative labeled comparison.
- To compare the current detector with `yolo11s.pt`, collect consenting, representative photos into `positive_backpack/` and hard negatives into `hard_negatives/` under one dataset folder (ideally grouped into `suitcase/`, `handbag/`, `person/`, `pet/`, and `unrelated/` subfolders). Include at least 30 backpack photos; avoid near-duplicate frames across the set. Run from the repository root:
  ```powershell
  $env:PYTHONPATH = "backend"
  python -m app.ai.evaluate_yolo_models --dataset "D:\path\to\backpack-evaluation"
  ```
  The script uses CPU inference, 640-pixel inputs and the same low inference floor for both models, then reports backpack recall and hard-negative false-positive rates at 45% and the configured threshold, confidence distributions, latency, and process RSS. The first image is used as a warm-up for timing. Weights not already cached may require a network download. The benchmark is local and does not send images to a third party.
- Targets use canonical COCO class names supported by the configured YOLO weights. The backend logs the decoded upload format and dimensions, model load status, top detections/confidences, target class, threshold, and final decision; it does not log image contents or coordinates. Rejections distinguish no detections, a different detected class, and a matching class below threshold.
- The configured `yolov8n.pt` weights are closed-vocabulary COCO detection: `tree`, `rock`, `leaf`, and `flower` are not classes. Target titles come from the required detector class (not Gemma-generated free text), and real games intersect the task vocabulary with class names exposed by the loaded weights. A detected `sheep`, `banana`, or `sandwich` is only that model's prediction, not evidence that a tree or rock was recognized.
- The current model remains authoritative; confidence matching is unchanged. On the development laptop, the bundled `bus.jpg` real-inference check loaded 80 classes and detected `bus` at 0.8734 against a 0.60 threshold. This sample demonstrates the pipeline, not performance on natural-object camera photos.
- EXIF metadata from uploaded photos is optionally checked as an additional anti-cheat signal, but the game relies primarily on live browser location.

## Camera and verification limitations
- The in-game camera uses `getUserMedia()` and captures a fresh frame into an in-memory image Blob; the game UI does not show a gallery/file picker.
- Browsers require a secure context for camera access. `http://localhost` is normally allowed for local development, but accessing the server from a phone using a LAN address such as `http://192.168.x.x:8000` may block camera access. Use the deployed HTTPS application for a phone demo.
- Removing the gallery picker improves the intended flow but cannot make cheating impossible: users can bypass the UI and call the API directly, and image models can make classification errors. The backend performs the authoritative YOLO class/confidence check and updates completion state itself.
- Real mode (`DEMO_MODE=false`) calls the configured Ollama Gemma model and YOLO model. It does not substitute successful mock results if either service is unavailable.
- The host may start with only one player. Multiplayer lobbies continue to share the same game targets while keeping completion and movement progress separate per player.

## Future Improvements
- WebSockets for real-time leaderboard updates instead of polling.
- More advanced YOLO models or fine-tuned versions for specific outdoor environments.
- Better EXIF tampering detection.
