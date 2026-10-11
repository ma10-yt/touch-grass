# Touch Grass 🌳🎮

**An AI-powered, location-aware outdoor scavenger hunt that turns the real world into a multiplayer game.**

🚀 **[Play Touch Grass — Live Demo](https://touch-grass-api-uwhz.onrender.com/)**

Explore your surroundings, discover AI-generated challenges, capture real-world objects, and compete with friends!

Built with Python, FastAPI, MongoDB, OpenStreetMap, Gemma, Cloudflare Workers AI, and YOLO.

**Repository:** [github.com/ma10-yt/touch-grass](https://github.com/ma10-yt/touch-grass)

---

## Table of Contents

- [What Is Touch Grass?](#what-is-touch-grass)
- [Features](#features)
- [How the Game Works](#how-the-game-works)
- [Project Architecture](#project-architecture)
- [The AI Journey: Ollama to Cloudflare](#the-ai-journey-ollama-to-cloudflare)
- [Location-Aware Target Generation](#location-aware-target-generation)
- [YOLO-Based Photo Verification](#yolo-based-photo-verification)
- [Multiplayer and Target Replacement](#multiplayer-and-target-replacement)
- [Technology Stack](#technology-stack)
- [Run Locally](#run-locally)
- [Environment Variables](#environment-variables)
- [Run Tests](#run-tests)
- [Deployment on Render](#deployment-on-render)
- [Privacy and Security](#privacy-and-security)
- [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## What Is Touch Grass?

Touch Grass is an outdoor scavenger-hunt game for solo players and groups.

Players create a game, share a code with friends, and receive five physical objects to find in their real surroundings. These targets are generated with the help of AI using nearby geographic information from OpenStreetMap.

For example, a player exploring a park might receive targets such as a bench, a bird, or a sports ball. Someone exploring a built-up area might receive different targets, depending on the geographic evidence and the classes supported by the image detector.

When a player finds an object, they use the in-game camera to take a photograph. The backend runs a YOLO object-detection model to determine whether the required object appears in the submitted image.

The first player to verify all five targets wins.

The goal is to combine AI, geographic data, and computer vision into an interactive experience that encourages real-world exploration.

## Features

- **AI-generated scavenger hunts:** Generates five distinct, eligible target classes rather than relying exclusively on a fixed list of missions.
- **Location-aware generation:** Uses OpenStreetMap reverse-geocoding and nearby geographic features to help rank and select appropriate objects.
- **Solo and multiplayer:** Create a game or join a shared lobby using a game code.
- **Real photo verification:** Uses a server-side YOLO model to verify target objects in submitted photographs.
- **Camera-based gameplay:** Players capture fresh images through the in-game camera rather than selecting an image from a gallery.
- **Collaborative target replacement:** Players can request a replacement for an unfinished target. The existing approval rules apply, and the replacement must be a different eligible class.
- **Live area-context updates:** As players move, the game can refresh its geographic context and update unfinished targets.
- **Optional activity statistics:** Supported devices can provide step-count estimates and GPS-route distance statistics.
- **AI provider flexibility:** Supports local Ollama, OpenRouter, and Cloudflare Workers AI through configuration.
- **Graceful AI failure handling:** When configured providers fail, the generator can use a location-ranked fallback built from supported detector classes.
- **Server-side game state:** MongoDB stores persistent game data, while the backend remains responsible for authoritative target verification and completion.

---

## How the Game Works

### 1. Create or Join

A player opens the application and creates a new game or joins an existing multiplayer game using its code.

### 2. Get the Player's Location

The browser requests location permission and obtains a GPS estimate. The backend validates the location according to the application's accuracy and freshness requirements.

The accepted location is used to determine the player's current area. A location estimate alone does not guarantee that geographic context has been retrieved successfully, so the context service separately tracks whether the map data is available, partial, or unavailable.

### 3. Gather Geographic Context

The backend uses OpenStreetMap services to obtain coarse locality information and nearby feature data. Features can indicate whether the surroundings are associated with parks, residential areas, roads, pedestrian paths, public transport, commercial areas, or other mapped amenities.

The application converts these features into a structured geographic context that can inform target selection.

### 4. Generate Five Targets

The target generator ranks eligible object classes according to the geographic context and the classes supported by the configured YOLO detector.

The selected AI provider receives the structured context, candidate classes, and target-generation instructions. It returns candidate targets with short hints.

The backend validates the returned data, including the number of targets, uniqueness, class names, and eligibility for the active detector.

### 5. Explore and Take Photos

Players go outside, find the objects, and capture fresh photographs using the in-game camera.

### 6. Verify the Objects

The backend runs the image through its configured YOLO detector. The submitted target is accepted only when the matching detector class satisfies the application's verification requirements.

AI-generated target text alone never awards completion.

### 7. Win

The first player to verify all five required objects wins. Walking distance and step count are optional statistics, not victory requirements.

---

## Project Architecture

Touch Grass separates the user interface, game logic, geographic context, AI generation, persistence, and image verification.

```text
Browser
  |
  |-- GPS / Camera / Optional Motion Sensors
  |
  v
FastAPI Backend
  |
  |-- Game API and Session Management
  |-- Target Generation and Replacement
  |-- Input Validation and Game Rules
  |
  +-----------------------+
  |                       |
  v                       v
Location Context       AI Target Generator
Service                    |
  |                        +-- Cloudflare Workers AI
  |                        +-- OpenRouter (optional)
  |                        +-- Ollama (local)
  |
  v
OpenStreetMap
  |
  |-- Nominatim Reverse Geocoding
  |-- Overpass Nearby Features
  |
  v
Ranked Geographic Context
  |
  v
YOLO-Supported Candidate Selection
  |
  v
Exactly Five Validated Targets

Fresh Camera Photo
  |
  v
FastAPI Backend
  |
  v
YOLO Object Detection
  |
  v
Server-Side Verification
  |
  v
MongoDB Game State and Completion
```

### Important architectural decisions

- Geographic context helps choose relevant tasks but does not prove that a specific object exists.
- The AI can select only eligible classes supported by the configured detector.
- The target generator validates model output before the game uses it.
- Photo verification takes place on the backend, not in the browser.
- A failed geographic refresh should not erase a game's previously valid targets or completed progress.
- AI provider selection is configured through environment variables rather than hardcoded into the frontend.

---

## The AI Journey: Ollama to Cloudflare

Choosing how to run the Game Master was one of the most important engineering decisions in this project. The local development experience and the deployed application had different requirements.

### Stage 1: Ollama + Gemma 3 1B — Local inference

The project initially used **Gemma 3 1B through Ollama**.

Ollama is a runtime for downloading and running language models locally. Gemma 3 1B is the model whose weights we downloaded and used during development. Its model download was approximately 815 MB in the local setup.

The local workflow was:

```powershell
ollama pull gemma3:1b
ollama run gemma3:1b
```

The backend could communicate with Ollama's local API to generate targets.

**Why this worked for development:** the model ran on the developer's own computer, without requiring a hosted inference provider for each request.

**Why it was not enough for deployment:** a Render server cannot access the Ollama process running on a developer's laptop through `localhost`. The production application needed a remote inference endpoint that remained available independently of the local development machine.

Running an Ollama instance on a separate server was investigated, but hosting eligibility and payment verification became obstacles.

### Stage 2: OpenRouter — Hosted free-model inference

The next approach was to use OpenRouter so that the deployed backend could send prompts to a hosted Gemma model rather than relying on local Ollama.

The project used the free variant:

`google/gemma-4-26b-a4b-it:free`

This made remote inference convenient, but the free model could hit request limits or temporary upstream provider-capacity limits. Those failures could prevent new games from generating their targets.

This exposed an important distinction: **a hosted model being free to use does not mean its inference capacity is unlimited or guaranteed to be available.**

The game needed more than a working API request. It also needed useful error handling so that temporary provider failures did not unnecessarily make the game unplayable.

### Stage 3: Cloudflare Workers AI — The production approach

Touch Grass moved its production target-generation requests to **Cloudflare Workers AI**, using the hosted model:

`@cf/google/gemma-4-26b-a4b-it`

This is a different model identifier and model family version from the local `gemma3:1b` model. The hosted model is provided by Cloudflare; its weights are not downloaded to the Render application or stored in this repository.

The backend sends prompts to Cloudflare's REST API:

```text
POST
https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}
```

The request uses Bearer-token authentication and structured chat messages.

During integration, the first API test returned an empty message because Gemma used the output-token budget for reasoning. Disabling thinking mode fixed the issue in our test:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

The backend also sets a completion-token limit, checks Cloudflare's success response, parses the generated message, and validates the returned target schema.

### Why Cloudflare?

Cloudflare allowed us to use a managed Gemma inference endpoint without hosting Ollama on a paid virtual machine.

The resulting design keeps the application backend and AI provider separate:

- **Render:** serves the FastAPI application and runs the YOLO verification pipeline.
- **Cloudflare Workers AI:** generates candidate scavenger-hunt targets.
- **MongoDB Atlas:** stores persistent game data.
- **OpenStreetMap:** provides geographic context.
- **Ollama:** remains available for local development.
- **OpenRouter:** remains supported as an optional provider where configured.

Cloudflare still has model-availability and usage limits. It is not an unlimited private server, and temporary capacity errors or exhausted free allocations can still occur. That is why error handling and a safe local target-selection fallback remain important parts of the architecture.

### Provider configuration

The provider is selected using `AI_PROVIDER`.

| Environment | Typical configuration |
|---|---|
| Local development with Ollama | `AI_PROVIDER=ollama` |
| Production with Cloudflare | `AI_PROVIDER=cloudflare` |
| Optional hosted backup | Set `AI_FALLBACK_PROVIDER` to a different configured provider |

`AI_FALLBACK_PROVIDER=none` disables provider-to-provider fallback by default. The generator's own location-ranked local fallback is separate from this setting.

Only configure a fallback provider when its credentials and endpoint are genuinely available to the running backend. For example, `OLLAMA_URL=http://localhost:11434` refers to the machine running the backend; it does not magically connect a deployed Render service to Ollama on a developer's computer.

---

## Location-Aware Target Generation

Location awareness is a core part of Touch Grass. The same universal set of targets should not be treated as equally suitable for every environment.

### OpenStreetMap context

The application uses two types of geographic lookup:

- **Nominatim:** reverse-geocodes the player's coarse location into locality and address information.
- **Overpass API:** searches for nearby mapped features within the configured radius.

The context service classifies relevant map features into categories such as:

- Parks and recreation
- Natural areas
- Residential areas
- Commercial areas
- Roads and junctions
- Pedestrian paths
- Public transport
- Public amenities

The generator uses these categories, feature counts, environment type, and other available context to rank eligible candidate classes.

For example, a park-oriented context can give more weight to classes such as benches and birds, while roads or public transport features can make classes such as cars, buses, bicycles, or traffic-related objects more relevant. These are rankings based on geographic evidence, not guarantees that an object is physically present.

### Geographic context and caching

The current configuration uses:

- Nearby-feature search radius: **1,200 metres**
- OSM request timeout: **10 seconds**
- Configured retry count: **2 retries after the initial attempt**
- Successful context cache lifetime: **30 minutes**
- Context cache key: coordinates rounded to four decimal places
- Stable movement distance for refreshing area context: approximately **200 metres**, subject to the application's movement and cooldown checks

These settings help reduce repeated network requests while allowing the game to refresh its context as the player explores.

When OSM responses are unavailable, the context service records that geographic context is partial or unavailable. In that case, the application can use broad outdoor context instead of pretending it has reliable local evidence.

### Candidate filtering

The AI does not receive unrestricted permission to invent object classes.

Before a response becomes part of the game, the backend checks the target data against the candidate pool and the classes supported by the detector configuration. It also enforces uniqueness and the required target count.

When provider generation fails, the local fallback ranks supported candidates using the available location scores and varies selection between game sessions. It does not need to generate new language-model text to choose a different valid combination.

This distinction is important: location-aware scoring and fallback selection are part of the application's deterministic logic, while Gemma provides contextual target selection and hints.

---

## YOLO-Based Photo Verification

Touch Grass uses a YOLO object-detection model to check whether a photograph contains the target object.

### Local development

The development setup uses the Ultralytics YOLO model file:

`yolov8n.pt`

The `ultralytics` package can download the weights when the model is first loaded.

### Render deployment

The Render build exports the model into ONNX format:

`yolov8n.onnx`

The production application uses ONNX Runtime for CPU inference. This reduces the runtime footprint compared with importing the full PyTorch and Ultralytics stack during production requests.

The repository's Render build configuration handles model export and removes packages that are only required for building and exporting the weights.

### Why target classes are restricted

The configured YOLO model uses a fixed COCO class vocabulary. It cannot recognize arbitrary natural-language descriptions as independent classes.

For example, **tree, rock, leaf, and flower are not classes in the standard YOLOv8 COCO detector used here**.

Asking a language model to produce targets outside the detector's vocabulary would create challenges that players could not reliably complete through the application's verification pipeline.

To avoid this, the target generator uses canonical detector class names and restricts candidate selections to supported classes. Target titles come from those class names rather than unrestricted AI-generated labels.

The detector may still misclassify real-world images. A model prediction is not perfect proof of object identity, so verification confidence thresholds and server-side validation remain necessary.

---

## Multiplayer and Target Replacement

Touch Grass supports both solo and shared games.

Players can join a game through its code and work with the shared set of objectives. Player completion state is tracked separately.

### Unanimous replacement voting

Players can request a replacement for an unfinished target under the game's existing voting rules. A replacement is applied only after the required approval process succeeds.

The backend protects the game state by:

- Restricting replacement requests to unfinished targets.
- Excluding the class being replaced from replacement candidates.
- Excluding other active target classes and required completed classes.
- Validating the new class against detector-supported candidates.
- Replacing only the requested target.
- Preserving the other targets and their completion state.
- Returning an explicit failure if no valid distinct replacement is available.

The model is not trusted to enforce these restrictions by itself. The backend validates the outcome.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend API | Python, FastAPI |
| Data validation | Pydantic |
| Database | MongoDB Atlas / MongoDB |
| Local language-model runtime | Ollama |
| Local development model | Gemma 3 1B |
| Production language-model inference | Cloudflare Workers AI |
| Production hosted model | Gemma 4 26B A4B Instruct |
| Optional AI provider | OpenRouter |
| Geographic data | OpenStreetMap |
| Reverse geocoding | Nominatim |
| Nearby feature queries | Overpass API |
| Image verification | YOLOv8n |
| Production inference runtime | ONNX Runtime |
| Hosting | Render |
| Automated testing | pytest, pytest-asyncio |

---

## Run Locally

### Requirements

- Python 3.12
- Git
- MongoDB locally or a MongoDB Atlas connection
- Ollama for local Gemma inference, or valid Cloudflare Workers AI credentials
- Internet access for OpenStreetMap requests and any hosted AI provider
- Node.js only if you want to run the frontend-specific JavaScript checks

### 1. Clone the repository

```powershell
git clone https://github.com/ma10-yt/touch-grass.git
cd touch-grass
```

### 2. Create a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

If PowerShell blocks the activation script, use the Python executable inside `.venv` directly instead of changing system-wide execution policy.

### 3. Configure environment variables

Copy the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and configure a working MongoDB URI.

The template defaults to Ollama and has demo mode enabled. For real local gameplay, set:

```dotenv
DEMO_MODE=false
AI_PROVIDER=ollama
GEMMA_MODEL=gemma3:1b
OLLAMA_URL=http://localhost:11434
```

Make sure MongoDB is running or replace `MONGODB_URI` with your Atlas connection string.

### 4. Install the local Gemma model

Install [Ollama](https://ollama.com/) and pull the model:

```powershell
ollama pull gemma3:1b
ollama list
```

Start Ollama if it is not already running. The local backend uses Ollama's API at `http://localhost:11434` by default.

Alternatively, to use Cloudflare locally, set `AI_PROVIDER=cloudflare` and supply `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in your ignored `.env` file.

Do not commit `.env` or expose real credentials.

### 5. Run the backend

From the project root:

```powershell
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --reload
```

Open:

[http://localhost:8000](http://localhost:8000)

The application serves the frontend through the backend.

For phone testing, use the deployed HTTPS site or another appropriately secured development setup. Browsers may block camera and location features on a phone accessing a plain LAN HTTP address.

### Demo mode

`DEMO_MODE=true` enables the application's demo behaviour. Demo target selection and verification are simulated and should not be presented as genuine AI generation or photo verification.

For actual games using the configured AI and detector, use:

```dotenv
DEMO_MODE=false
```

---

## Environment Variables

Configure these through the local `.env` file or the hosting provider's secure environment-variable settings. Keep credentials out of source control.

### Application and AI configuration

| Variable | Default / example | Purpose |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DATABASE` | `touch_grass` | Database name |
| `DEMO_MODE` | `true` in `.env.example` | Enables or disables demo behaviour |
| `AI_PROVIDER` | `ollama` | Primary AI provider |
| `AI_FALLBACK_PROVIDER` | `none` | Optional secondary provider |
| `GEMMA_MODEL` | `gemma3:1b` | Local Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `CLOUDFLARE_ACCOUNT_ID` | Empty by default | Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | Empty by default | Cloudflare Workers AI API token |
| `CLOUDFLARE_AI_MODEL` | `@cf/google/gemma-4-26b-a4b-it` | Cloudflare inference model |
| `OPENROUTER_API_KEY` | Empty by default | Optional OpenRouter credential |
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it:free` | OpenRouter model identifier |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |

Supported values for `AI_PROVIDER` are `ollama`, `openrouter`, and `cloudflare`. The fallback setting also supports `none`.

### Geographic context

| Variable | Default | Purpose |
|---|---|---|
| `GEOCONTEXT_PROVIDER` | `openstreetmap` | Geographic context provider |
| `GEOCONTEXT_NOMINATIM_URL` | `https://nominatim.openstreetmap.org/reverse` | Reverse-geocoding endpoint |
| `GEOCONTEXT_OVERPASS_URL` | `https://overpass-api.de/api/interpreter` | Nearby-feature endpoint |
| `GEOCONTEXT_USER_AGENT` | `TouchGrassGame/1.0 (location context)` | Identifying application user agent |
| `GEOCONTEXT_RADIUS_METERS` | `1200` | Nearby-feature search radius |
| `GEOCONTEXT_TIMEOUT_SECONDS` | `10` | Context request timeout |
| `GEOCONTEXT_RETRIES` | `2` | Retries after the initial request |
| `GEOCONTEXT_CACHE_TTL_SECONDS` | `1800` | Context cache lifetime |
| `GEOCONTEXT_REFRESH_DISTANCE_METERS` | `200` | Movement threshold for context refresh |
| `GEOCONTEXT_REFRESH_COOLDOWN_SECONDS` | `60` | Minimum refresh cooldown |
| `GEOCONTEXT_REFRESH_STABILITY_METERS` | `150` | Movement stability threshold |
| `GEOCONTEXT_REFRESH_CONFIRMATIONS` | `2` | Required stable movement confirmations |

Use a descriptive `GEOCONTEXT_USER_AGENT` with a valid maintainer contact for deployment. Follow the published usage policies of Nominatim and Overpass.

### Location, verification, and gameplay

| Variable | Default | Purpose |
|---|---|---|
| `LOCATION_MAX_AGE_SECONDS` | `60` | Maximum accepted location age |
| `GPS_MAX_ACCURACY_METERS` | `250` | Maximum allowed GPS accuracy radius |
| `GPS_HIGH_CONFIDENCE_METERS` | `50` | High-confidence GPS threshold |
| `GPS_MOVEMENT_MAX_ACCURACY_METERS` | `250` | Accuracy ceiling for movement |
| `GPS_MIN_SEGMENT_METERS` | `25` | Minimum GPS segment distance |
| `GPS_MOVEMENT_NOISE_FLOOR_METERS` | `15` | Movement noise filter |
| `GPS_MOVEMENT_STABILITY_METERS` | `35` | Movement stability threshold |
| `GPS_MOVEMENT_CONFIRMATIONS` | `2` | Stable movement confirmations |
| `GPS_MAX_SPEED_METERS_PER_SECOND` | `12` | Plausible movement-speed ceiling |
| `MAX_UPLOAD_MB` | `10` | Image upload size limit |
| `YOLO_MODEL` | `yolov8n.pt` locally; `yolov8n.onnx` on Render | Detector model file |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.60` | Matching-class confidence threshold |
| `TARGET_VOTE_INACTIVE_SECONDS` | `90` | Multiplayer voter inactivity threshold |

These are the application's current defaults, but production settings may override them. Check `backend/app/config.py` and the live environment when diagnosing a deployment.

---

## Run Tests

The backend test suite uses pytest and mocked external requests so that ordinary tests do not consume Cloudflare or OpenRouter inference allocations.

From the project root in Windows PowerShell:

```powershell
$env:PYTHONPATH = "backend"

& ".\.venv\Scripts\python.exe" -m pytest backend\tests -q
```

To check whitespace errors before committing:

```powershell
git diff --check
```

If you change frontend JavaScript, run the existing frontend checks where applicable:

```powershell
node --check frontend\app.js
node --test frontend\tests\app-state.test.js
```

Optional real YOLO inference smoke test:

```powershell
$env:PYTHONPATH = "backend"
$env:RUN_YOLO_SMOKE_TEST = "1"
python -m pytest backend/tests/test_yolo_smoke.py
```

For the current Render-style ONNX inference smoke test:

```powershell
pip install -r backend/requirements-render-export.txt

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=False)"

$env:RUN_YOLO_ONNX_SMOKE_TEST = "1"
python -m pytest backend/tests/test_yolo_onnx_smoke.py -q
```

The real-inference smoke tests are optional and require the appropriate model files and dependencies.

**Recent development verification:** the backend test suite reported 157 passed and 2 skipped after the location-awareness changes.

---

## Deployment on Render

The application is deployed as a FastAPI web service on Render, with MongoDB Atlas as the persistent database and Cloudflare Workers AI as the production target-generation provider.

### Production configuration

For the existing deployed backend service, configure these variables in the Render Dashboard under **Environment**:

| Variable | Production value |
|---|---|
| `MONGODB_URI` | Your MongoDB Atlas connection string |
| `MONGODB_DATABASE` | `touch_grass` |
| `DEMO_MODE` | `false` |
| `AI_PROVIDER` | `cloudflare` |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | A scoped Workers AI API token |
| `CLOUDFLARE_AI_MODEL` | `@cf/google/gemma-4-26b-a4b-it` |
| `GEOCONTEXT_USER_AGENT` | An identifying value that includes maintainer contact details |
| `YOLO_MODEL` | `yolov8n.onnx` |

Set `CLOUDFLARE_API_TOKEN` directly in Render's secure environment settings. Do not place it in the repository, frontend code, screenshots, or public logs.

If you configure an optional secondary AI provider, also configure its credentials and verify that the running backend can reach it.

### Important Blueprint note

The current `render.yaml` contains the original OpenRouter default for `AI_PROVIDER`. The deployed service can override that value through its Render Environment settings, which is how the current Cloudflare deployment is configured.

If creating a new service or reapplying the Blueprint, review the provider setting and Cloudflare variables before deployment. Do not assume a new Blueprint automatically inherits the manually configured production provider and secrets.

### Deployment details

The Render Blueprint defines a single public web service in the Oregon region on the Free plan. Its build process installs the production dependencies, exports the YOLO weights to ONNX, and removes build-only PyTorch/Ultralytics dependencies before runtime.

Render's free service has resource, cold-start, and filesystem limitations. The ONNX model file can be regenerated during the build; persistent game data is stored in MongoDB Atlas rather than in the service's ephemeral filesystem.

### After deployment

1. Confirm the Render deployment finishes successfully.
2. Check the `/api/health` endpoint.
3. Create a real game and check the Render logs for AI provider and geographic-context diagnostics.
4. Verify that five unique target classes are generated and supported by the active YOLO model.
5. Test a photo submission through the camera.
6. If testing location relevance, compare areas with genuinely different mapped features and verify that the logs show different context categories or candidate rankings where appropriate.
7. Test the replacement flow to confirm that a replacement is a different eligible class and that the other targets remain unchanged.

Successful deployment alone does not prove that OpenStreetMap has returned meaningful context or that the free AI provider will always have capacity.

---

## Privacy and Security

Touch Grass uses location and photos because its gameplay depends on real-world exploration and object verification. These inputs should be handled with care.

- The browser requests location permission and sends location information to the application backend.
- The location-context service rounds coordinates to four decimal places for its cache key and OSM lookup flow. This is coarse precision, not a guarantee of anonymity.
- The backend uses the resulting geographic context for target ranking and the AI prompt. Exact GPS coordinates are not included in the Gemma prompt.
- The browser sends submitted camera images to the backend for YOLO verification. The photos are not sent to Cloudflare as part of the target-generation request.
- Cloudflare receives the target-generation prompt and the geographic context needed to select appropriate objects. Because the current production provider is hosted, this data is processed by a third-party inference service.
- API tokens and database credentials must remain in secure server-side environment variables.
- The backend validates AI output and performs authoritative image verification; the frontend does not decide whether a target is complete.
- OpenStreetMap data is subject to its provider's usage policies and attribution requirements. The application displays attribution where map data is used.

Location and computer-vision checks help support the intended game mechanics, but they cannot make every form of client-side cheating impossible. Browser APIs can be bypassed, GPS estimates can be inaccurate, and object detectors can misclassify photos.

---

## Known Limitations

### Geographic data quality

OSM coverage varies between locations. Rural areas, small parks, newly developed areas, and poorly mapped places may have incomplete nearby-feature data. When lookups fail or return insufficient context, the game must use broader outdoor context rather than claim precise local relevance.

### Detector vocabulary

The configured YOLOv8n COCO detector supports a fixed set of classes. It cannot reliably verify every physical object that a person might encounter outdoors. Targets must remain within the supported vocabulary.

### Language-model availability

Cloudflare Workers AI can experience temporary provider-capacity errors and free-allocation exhaustion. Optional provider fallback can help when configured, and the local selection fallback can keep a game playable when sufficient supported candidates remain. No cloud inference provider is guaranteed to be continuously available without limits.

### Render free-tier resources

The deployed application uses CPU-based ONNX inference on Render's free web service. Cold starts, CPU performance, memory usage, and service availability can affect response times. The current configuration is suitable for a small project demonstration, but sustained production-scale reliability has not been established.

### Location and camera permissions

Camera and location permissions must be granted in the browser. Mobile browsers generally require a secure context such as HTTPS, with localhost being a common development exception. Sensor support and motion-based step estimates vary between devices.

---

## Future Improvements

- Improve location ranking with richer geographic evidence and more detailed environment-specific scoring.
- Evaluate additional object detectors or fine-tuned models for outdoor objects not present in the current COCO class list.
- Add more robust geographic-context monitoring and diagnostics.
- Improve real-time multiplayer updates and leaderboard responsiveness.
- Evaluate target selection quality over a broader set of locations and real-world photographs.
- Improve observability and deployment reliability as the project grows.

---

## Acknowledgements

Touch Grass brings together several open-source tools, open-weight models, and hosted services:

- [Google Gemma](https://ai.google.dev/gemma) — language-model family used for game-master generation.
- [Ollama](https://ollama.com/) — local model runtime for development.
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) — hosted production inference.
- [OpenRouter](https://openrouter.ai/) — an alternative hosted model provider.
- [Ultralytics YOLO](https://docs.ultralytics.com/) — object detection.
- [MongoDB Atlas](https://www.mongodb.com/atlas) — persistent game data.
- [OpenStreetMap](https://www.openstreetmap.org/) and [Nominatim](https://nominatim.org/) — geographic information and reverse geocoding.
- [Render](https://render.com/) — application hosting.

Touch Grass was built for the Hacktoberfest Weekend Challenge with the idea that AI can do more than generate text on a screen: it can help turn the physical world into a game.

**Touch Grass. Explore your surroundings. Find your targets.** 🌳