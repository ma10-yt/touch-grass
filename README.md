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
   To export and run the real ONNX Runtime smoke test used for the Render configuration:
   ```powershell
   pip install -r backend/requirements-render-export.txt
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=False)"
   $env:RUN_YOLO_ONNX_SMOKE_TEST = "1"
   python -m pytest backend/tests/test_yolo_onnx_smoke.py -q
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

The root [`render.yaml`](./render.yaml) defines exactly one **public FastAPI web service on Render's Free plan**. There are no private services, Render databases, persistent disks, or paid add-ons. FastAPI serves the existing `frontend/` directory and `/api/health` is the health check. The service uses MongoDB Atlas for persistent games, OpenRouter for Gemma target generation, and CPU ONNX Runtime for genuine server-side YOLO image verification.

#### Resource limits and important caveat

Render documents its Free web service as **512 MB RAM**, which is a shared-CPU plan. Free services spin down after 15 minutes idle, take about a minute to cold-start, have ephemeral filesystems, and consume a shared **750 free instance-hours/month** per workspace. Free instances are documented as [unsuitable for production applications](https://render.com/docs/free). The ONNX deployment avoids importing PyTorch/Ultralytics at runtime; these are installed only during build to export the real YOLO weights, then removed. Uploaded images and the ONNX model file are ephemeral; games remain in Atlas and the model is regenerated during builds.

I measured the FastAPI app imported with ONNX Runtime loaded and ran two actual inferences locally, without importing Ultralytics/PyTorch at runtime. Peak process RSS was about **176 MB**. The `bus.jpg` inference returned bus class ID `5` at `0.836` confidence; `zidane.jpg` returned person class ID `0` at `0.822`. The corresponding PyTorch results with fixed 640×640 preprocessing were within about 0.01 confidence. This is a Windows/local measurement, not Render's Linux runtime, and excludes simultaneous requests and production traffic. It suggests a single-worker/single-inference process can fit 512 MB in this local test, but **Render's memory fit under load is not proven**; monitor memory and cold-start behavior, and expect possible OOM/restarts or slow CPU inference. Do not treat this as production reliability evidence.

#### Free Gemma API limits

The Blueprint uses OpenRouter model `google/gemma-4-26b-a4b-it:free` (the current free Gemma variant found in the [OpenRouter model catalog](https://openrouter.ai/models), advertising `response_format`; it is not the local Gemma 3 1B model). OpenRouter's documented [free-variant limits](https://openrouter.ai/docs/api_reference/limits) are **20 requests/minute and 50 requests/day** when the account has purchased less than 10 credits. The 1,000/day tier requires at least 10 credits purchased and is outside a strict $0 budget. Generation can retry once after non-rate-limit errors, so worst case one fresh game start can use two requests; a daily cap of 50 therefore allows at most 25 such starts if each exhausts both attempts, before target changes or other API calls. HTTP 429 errors are returned without an immediate retry. Provider availability/capacity limits can also return 429s. Check the OpenRouter key usage endpoint/dashboard for current quota. Game creation/start requires Gemma; if the API key is missing or generation fails, the server returns an error and does **not** substitute fixed demo targets.

#### Deployment steps (no billing setup)

1. Push this project to a Git-provider repository yourself after checking that no `.env`, credentials, `.venv`, `.pt`, or generated `.onnx` files are tracked. The present project directory has no `.git` metadata; this setup does not push or deploy code for you.
2. Create or select a MongoDB Atlas deployment, database user, and database. Configure Atlas network access for the Render web service; do not expose MongoDB broadly unless you accept that risk. Atlas may have costs independent of Render.
3. Create a free OpenRouter account/key for free model use. Do not add paid credits if maintaining a strict $0 budget. Keep the key private.
4. In Render, choose **New → Blueprint**, connect your repository, and select the root `render.yaml`. Verify the review contains one web service with plan `free` and no disk, private service, or paid database. Do not add a payment method or upgrade the plan.
5. Set the prompted `MONGODB_URI`, `OPENROUTER_API_KEY`, and `GEOCONTEXT_USER_AGENT` in Render's secure environment-variable form. Use an identifying OSM User-Agent with a maintainer contact, for example `TouchGrassGame/1.0 (contact: your-maintainer-email)`. Follow Nominatim/Overpass usage policies.
6. Apply the Blueprint. The build installs the app dependencies, downloads the actual `yolov8n.pt` weights through Ultralytics, exports ONNX at 640×640, then removes training/export-only PyTorch packages. Production serves the ONNX model with CPUExecutionProvider. No model weights from the local machine are required.
7. Check `/api/health` and `/api/health/ai`; the latter confirms OpenRouter key connectivity/model listing but cannot guarantee free-model quota or provider availability. Test MongoDB game persistence and a real camera submission after deployment. Grant location and camera permissions over HTTPS; optional step counting remains browser-side.

### Environment variables

The Blueprint sets production-specific values and asks for secrets through Render's secure dashboard prompt; no secret values belong in this repository. Application settings recognized by `backend/app/config.py` include:

| Variable | Deployment value / default |
| --- | --- |
| `MONGODB_URI` | Required secret; set in Render to the Atlas connection URI. |
| `MONGODB_DATABASE` | `touch_grass` |
| `DEMO_MODE` | `false` (required for real Gemma and YOLO). |
| `AI_PROVIDER` | `openrouter` on Render; defaults to `ollama` locally. |
| `OPENROUTER_API_KEY` | Required secret in Render; absent locally unless OpenRouter is selected. |
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it:free` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `GEMMA_MODEL` | `gemma3:1b` for local Ollama development. |
| `OLLAMA_URL` | `http://localhost:11434` local default; unused by Render's OpenRouter provider. |
| `YOLO_MODEL` | `yolov8n.onnx` on Render; local default `yolov8n.pt`. |
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

`PYTHON_VERSION=3.12.10` selects the Render Python build runtime. Production AI failures remain explicit; `DEMO_MODE` must stay `false`.

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
