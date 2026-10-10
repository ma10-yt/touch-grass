const API_URL = '/api/games';
let gameState = {
    id: null,
    joinCode: null,
    playerId: null,
    targets: [],
    completed: [],
    targetChangeVote: null,
    leaderboard: [],
    lastAnnouncedVoteId: null,
    justVerifiedTargetIds: new Set(),
    isCreator: false,
    demoMode: false,
    cameraFacingMode: 'environment',
    cameraStream: null
};
let pollInterval = null;
let locationWatchId = null;
let locationTrackingGeneration = 0;
let locationObservationSequence = 0;
let locationRequestPending = false;
let pendingLocationPosition = null;
let currentTargetId = null;
let cameraCancelHandlerRegistered = false;
let cameraRequestGeneration = 0;
let locationIssue = null;
let lastLocationFix = null;
let creatingGame = false;
let stepCount = 0;
let stepListener = null;
let stepTrackingGeneration = 0;
let stepTrackingActive = false;
let stepSensorTimeout = null;
let stepLastCountedAt = -Infinity;
let stepPeakStarted = false;
let stepPeakMagnitude = 0;
let stepFilteredMagnitude = 0;
let stepGravityBaseline = null;
let lastGpsDistanceFix = null;
let gpsDistanceMeters = 0;
const STEP_PEAK_THRESHOLD = 1.2;
const STEP_MIN_INTERVAL_MS = 350;
const GPS_DISTANCE_MAX_ACCURACY_METERS = 15;
const GPS_DISTANCE_NOISE_FLOOR_METERS = 5;

if (document.addEventListener) {
    document.addEventListener('visibilitychange', handleStepCounterVisibilityChange);
}

async function loadRuntimeMode() {
    const banner = document.getElementById('runtime-mode');
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        gameState.demoMode = Boolean(health.demo_mode);
        banner.innerText = gameState.demoMode
            ? 'DEMO MODE — AI targets and image verification are simulated.'
            : 'REAL AI MODE — Gemma targets and YOLO image verification.';
    } catch (error) {
        banner.innerText = 'Game mode unavailable.';
    }
}

function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

async function createGame() {
    if (creatingGame) return;
    const name = document.getElementById('create-name').value;
    if(!name) return alert("Enter a name");

    creatingGame = true;
    const createButton = document.getElementById('btn-create');
    createButton.disabled = true;
    resetClientGameState({ clearSavedSession: true });
    const createWithLocation = (lat, lon) => doCreate(name, lat, lon)
        .catch(error => {
            console.error('Game creation failed', error);
            document.getElementById('create-status').innerText =
                'Could not create a new game. Check the server connection and retry.';
        })
        .finally(() => {
            creatingGame = false;
            createButton.disabled = false;
        });

    if (!navigator.geolocation) {
        await createWithLocation(null, null);
        return;
    }
    navigator.geolocation.getCurrentPosition(
        pos => createWithLocation(pos.coords.latitude, pos.coords.longitude),
        () => createWithLocation(null, null),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
    );
}

async function doCreate(name, lat, lon) {
    const res = await fetch(API_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({creator_name: name, lat, lon})
    });
    const data = await res.json();
    if (!res.ok || !data.id || !data.creator_id) {
        throw new Error(data.detail || `Failed to create game (HTTP ${res.status}).`);
    }

    resetClientGameState({ clearSavedSession: true });
    gameState.id = data.id;
    gameState.joinCode = data.join_code;
    gameState.playerId = data.creator_id;
    gameState.isCreator = true;
    saveGameSession();

    document.getElementById('lobby-code').innerText = data.join_code;
    document.getElementById('btn-start').style.display = 'block';
    
    showScreen('screen-lobby');
    startPolling();
    startLocationTracking();
}

async function joinGame() {
    const name = document.getElementById('join-name').value;
    const code = document.getElementById('join-code').value.toUpperCase();
    if(!name || !code) return alert("Enter name and code");

    const res = await fetch(`${API_URL}/join`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({display_name: name, join_code: code})
    });
    
    if(!res.ok) return alert("Failed to join game. Check code.");
    const data = await res.json();
    resetClientGameState({ clearSavedSession: true });
    
    gameState.id = data.game_id;
    gameState.joinCode = code;
    gameState.playerId = data.id;
    gameState.isCreator = false;
    saveGameSession();

    document.getElementById('lobby-code').innerText = code;
    
    showScreen('screen-lobby');
    startPolling();
    startLocationTracking();
}

async function startGame() {
    const button = document.getElementById('btn-start');
    if (!gameState.id || !gameState.playerId || !gameState.isCreator) {
        document.getElementById('lobby-status').innerText =
            'Only the host can start this game.';
        return;
    }
    button.disabled = true;
    button.innerText = 'Preparing your targets…';
    try {
        const res = await fetch(`${API_URL}/${gameState.id}/start?creator_id=${encodeURIComponent(gameState.playerId)}`, { method: 'POST' });
        if (!res.ok) {
            const error = await res.json();
            button.disabled = false;
            button.innerText = 'Start the game →';
            document.getElementById('lobby-status').innerText =
                error.detail || `Failed to start game (HTTP ${res.status}).`;
            return;
        }
        const result = await res.json();
        renderLocationContext(result);
        if (result.context_notice) {
            document.getElementById('upload-status').innerText = result.context_notice;
        }
        await pollGameState();
    } catch (error) {
        button.disabled = false;
        button.innerText = 'Start the game →';
        document.getElementById('lobby-status').innerText =
            'Could not reach the server to start the game. Please try again.';
    }
}

function saveGameSession() {
    try {
        localStorage.setItem('touch-grass-session', JSON.stringify({
            id: gameState.id,
            joinCode: gameState.joinCode,
            playerId: gameState.playerId,
            isCreator: gameState.isCreator
        }));
    } catch (error) {
        document.getElementById('context-status').innerText =
            'Session could not be saved locally; movement remains stored on the server.';
        document.getElementById('context-status').hidden = false;
    }
}

function readSavedGameSession() {
    let saved;
    try {
        saved = JSON.parse(localStorage.getItem('touch-grass-session') || 'null');
    } catch (error) {
        localStorage.removeItem('touch-grass-session');
        return null;
    }
    return saved && saved.id && saved.playerId ? saved : null;
}

function showResumeOption() {
    document.getElementById('btn-resume').hidden = !readSavedGameSession();
}

function restoreGameSession() {
    const saved = readSavedGameSession();
    if (!saved) {
        showResumeOption();
        return;
    }
    resetClientGameState();
    gameState.id = saved.id;
    gameState.joinCode = saved.joinCode;
    gameState.playerId = saved.playerId;
    gameState.isCreator = Boolean(saved.isCreator);
    document.getElementById('lobby-code').innerText = gameState.joinCode || '';
    document.getElementById('btn-start').style.display =
        gameState.isCreator ? 'block' : 'none';
    showScreen('screen-lobby');
    startPolling();
    startLocationTracking();
    pollGameState();
    document.getElementById('btn-resume').hidden = true;
}

function leaveGame() {
    resetClientGameState({ clearSavedSession: true });
    document.getElementById('btn-resume').hidden = true;
    document.getElementById('lobby-code').innerText = '...';
    document.getElementById('copy-status').innerText = '';
    document.getElementById('lobby-players').innerHTML = '';
    document.getElementById('leaderboard-list').innerHTML = '';
    document.getElementById('target-list').innerHTML = '';
    document.getElementById('score').innerText = 'Score: 0';
    document.getElementById('progress').innerText = 'Progress: 0%';
    renderActivityStats();
    document.getElementById('loc-status').innerText = 'Location: finding...';
    document.getElementById('upload-status').innerText = '';
    setFeedbackState(document.getElementById('upload-status'), '');
    document.getElementById('create-status').innerText = '';
    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').innerText = 'Start the game →';
    document.getElementById('lobby-status').innerText = 'Share your code and bring your people along.';
    showScreen('screen-landing');
}

function resetClientGameState({ clearSavedSession = false } = {}) {
    stopPolling();
    stopLocationTracking();
    stopStepCounter({ reset: true });
    const cameraModal = document.getElementById('camera-modal');
    if (gameState.cameraStream || cameraModal.open || !cameraModal.hidden) {
        stopCamera();
    }
    if (clearSavedSession) localStorage.removeItem('touch-grass-session');

    gameState.id = null;
    gameState.joinCode = null;
    gameState.playerId = null;
    gameState.targets = [];
    gameState.completed = [];
    gameState.targetChangeVote = null;
    gameState.leaderboard = [];
    gameState.lastAnnouncedVoteId = null;
    gameState.justVerifiedTargetIds = new Set();
    gameState.isCreator = false;
    locationIssue = null;
    lastLocationFix = null;
    locationRequestPending = false;
    gpsDistanceMeters = 0;
    lastGpsDistanceFix = null;
    document.getElementById('loc-status').innerText = 'Location: finding...';
    document.getElementById('upload-status').innerText = '';
    document.getElementById('create-status').innerText = '';
    document.getElementById('context-status').hidden = true;
    document.getElementById('osm-attribution').hidden = true;
    document.getElementById('lobby-code').innerText = '...';
    document.getElementById('copy-status').innerText = '';
    document.getElementById('lobby-players').innerHTML = '';
    document.getElementById('leaderboard-list').innerHTML = '';
    document.getElementById('target-list').innerHTML = '';
    document.getElementById('score').innerText = 'Score: 0';
    document.getElementById('progress').innerText = 'Progress: 0%';
    renderActivityStats();
    document.getElementById('btn-start').style.display = 'none';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').innerText = 'Start the game →';
    document.getElementById('lobby-status').innerText = 'Share your code and bring your people along.';
}

function startPolling() {
    stopPolling();
    pollInterval = setInterval(pollGameState, 3000);
}

function stopPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = null;
}

async function pollGameState() {
    if(!gameState.id) return;
    const requestGameId = gameState.id;
    const requestPlayerId = gameState.playerId;
    void fetch(`${API_URL}/${requestGameId}/players/${requestPlayerId}/heartbeat`, {
        method: 'POST'
    }).then(response => {
        if (!response.ok) {
            console.warn('Player activity heartbeat rejected', {
                status: response.status
            });
        }
    }).catch(error => {
        console.warn('Player activity heartbeat failed', {
            message: error instanceof Error ? error.message : String(error)
        });
    });
    const res = await fetch(`${API_URL}/${requestGameId}`);
    if (gameState.id !== requestGameId || gameState.playerId !== requestPlayerId) return;
    if(!res.ok) return;
    const game = await res.json();
    
    const lbRes = await fetch(`${API_URL}/${requestGameId}/leaderboard`);
    if (gameState.id !== requestGameId || gameState.playerId !== requestPlayerId) return;
    const leaderboard = await lbRes.json();
    
    const me = leaderboard.find(p => p.id === gameState.playerId);
    gameState.leaderboard = leaderboard;
    if(me) {
        const previouslyCompleted = new Set(gameState.completed);
        gameState.completed = me.completed_targets;
        gameState.justVerifiedTargetIds = new Set(
            gameState.completed.filter(targetId => !previouslyCompleted.has(targetId))
        );
        document.getElementById('score').innerText = `Score: ${me.score}`;
    }
    renderLocationContext(game.location_context);

    if(game.status === 'ACTIVE') {
        gameState.targets = game.targets;
        gameState.targetChangeVote = game.target_change_vote || null;
        renderProgress(getCompletedTargetCount(), gameState.targets.length);
        renderTargets();
        gameState.justVerifiedTargetIds.clear();
        announceTargetChange();
        if(document.getElementById('screen-lobby').classList.contains('active')) {
            showScreen('screen-active');
        }
    } else if (game.status === 'FINISHED') {
        const winner = leaderboard.find(player =>
            game.targets.every(target => player.completed_targets.includes(target.id))
        ) || leaderboard[0];
        document.getElementById('winner-name').innerText =
            `${winner.display_name} won with ${winner.score} verified targets!`;
        const targetList = document.getElementById('winner-targets');
        targetList.innerHTML = '';
        game.targets.forEach(target => {
            const item = document.createElement('li');
            item.innerText = `✓ ${target.name}`;
            targetList.appendChild(item);
        });
        document.getElementById('winner-activity').innerText =
            `Your activity: ${stepCount} estimated steps · ` +
            (gpsDistanceMeters > 0
                ? `${Math.round(gpsDistanceMeters)} m GPS route estimate`
                : 'GPS route estimate unavailable');
        showScreen('screen-winner');
        stopPolling();
        stopLocationTracking();
        stopStepCounter({ status: 'Game finished; step counter stopped.' });
    }

    renderLeaderboard(leaderboard, game.creator_id);
}

async function copyGameCode() {
    const code = document.getElementById('lobby-code').innerText;
    const status = document.getElementById('copy-status');
    if (!code || code === '...') return;
    try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) {
            throw new Error('Clipboard is unavailable in this browser.');
        }
        await navigator.clipboard.writeText(code);
        status.innerText = 'Game code copied. Share it with your crew.';
    } catch (error) {
        status.innerText = `Copy unavailable — share this code: ${code}`;
    }
}

function renderLocationContext(context) {
    const status = document.getElementById('context-status');
    const attribution = document.getElementById('osm-attribution');
    if (!context) {
        status.hidden = true;
        attribution.hidden = true;
        return;
    }
    const locality = context.locality
        ? ` near ${context.locality}`
        : ' in an approximate area';
    if (context.context_mode === 'contextual') {
        const environment = context.environment_type.replaceAll('_', ' ');
        status.innerText = `AI Game Master: ${environment} context${locality}`;
    } else {
        status.innerText = `AI Game Master: General outdoor context${locality}`;
    }
    if (context.notice) status.innerText += `. ${context.notice}`;
    status.hidden = false;
    attribution.hidden = context.context_provider !== 'openstreetmap'
        || context.context_status === 'unavailable';
}

function renderProgress(completed, total) {
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
    document.getElementById('progress').innerText = `Progress: ${completed}/${total} (${percent}%)`;
    const fill = document.querySelector
        ? document.querySelector('.progress-fill')
        : null;
    if (fill) fill.style.width = `${percent}%`;
}

function resetStepSignal() {
    stepPeakStarted = false;
    stepPeakMagnitude = 0;
    stepFilteredMagnitude = 0;
    stepGravityBaseline = null;
    stepLastCountedAt = -Infinity;
}

async function startStepCounter() {
    if (!gameState.id || !gameState.playerId) {
        updateStepStatus('Join or create a game before starting the step counter.');
        return;
    }
    if (stepTrackingActive) return;
    if (typeof DeviceMotionEvent === 'undefined') {
        updateStepStatus('Step counter unsupported: this browser does not provide motion sensors.');
        return;
    }

    const startGeneration = stepTrackingGeneration;
    const gameId = gameState.id;
    const playerId = gameState.playerId;
    const requestPermission = DeviceMotionEvent.requestPermission;
    if (typeof requestPermission === 'function') {
        try {
            const permission = await requestPermission.call(DeviceMotionEvent);
            if (
                permission !== 'granted'
                || startGeneration !== stepTrackingGeneration
                || gameState.id !== gameId
                || gameState.playerId !== playerId
            ) {
                if (permission !== 'granted') {
                    updateStepStatus('Motion permission denied. Allow motion access in browser settings and retry.');
                }
                return;
            }
        } catch (error) {
            updateStepStatus('Motion permission could not be granted. Check browser settings and retry.');
            return;
        }
    }

    if (
        startGeneration !== stepTrackingGeneration
        || gameState.id !== gameId
        || gameState.playerId !== playerId
    ) return;

    stepTrackingGeneration += 1;
    const trackingGeneration = stepTrackingGeneration;
    resetStepSignal();
    stepTrackingActive = true;
    stepListener = event => {
        if (
            !stepTrackingActive
            || trackingGeneration !== stepTrackingGeneration
            || gameState.id !== gameId
            || gameState.playerId !== playerId
        ) return;
        handleMotionSample(event, event.timeStamp || Date.now());
    };
    if (typeof addEventListener !== 'function') {
        stopStepCounter({ status: 'Step counter unsupported: motion event listeners are unavailable.' });
        return;
    }
    addEventListener('devicemotion', stepListener);
    updateStepStatus('Waiting for motion sensor data…');
    stepSensorTimeout = setTimeout(() => {
        if (trackingGeneration === stepTrackingGeneration && stepTrackingActive) {
            updateStepStatus('No motion data received. Check sensor support and browser permission.');
        }
    }, 5000);
}

function stopStepCounter({ reset = false, status = 'Step counter stopped.' } = {}) {
    stepTrackingGeneration += 1;
    stepTrackingActive = false;
    if (stepListener && typeof removeEventListener === 'function') {
        removeEventListener('devicemotion', stepListener);
    }
    stepListener = null;
    if (stepSensorTimeout !== null) clearTimeout(stepSensorTimeout);
    stepSensorTimeout = null;
    resetStepSignal();
    if (reset) stepCount = 0;
    updateStepStatus(status);
    renderActivityStats();
}

function handleStepCounterVisibilityChange() {
    if (document.visibilityState === 'hidden' && stepTrackingActive) {
        stopStepCounter({
            status: 'Step tracking paused in the background. Tap Start Step Counter to resume.'
        });
    }
}

function updateStepStatus(message) {
    const status = document.getElementById('step-status');
    if (status) status.innerText = message;
    const button = document.getElementById('btn-step-counter');
    if (button) button.innerText = stepTrackingActive ? 'Stop Step Counter' : 'Start Step Counter';
}

function toggleStepCounter() {
    if (stepTrackingActive) {
        stopStepCounter();
    } else {
        void startStepCounter();
    }
}

function handleMotionSample(event, timestamp) {
    const acceleration = event.acceleration;
    const gravityAcceleration = event.accelerationIncludingGravity;
    let x;
    let y;
    let z;
    if (hasValidVector(acceleration)) {
        ({ x, y, z } = acceleration);
    } else if (hasValidVector(gravityAcceleration)) {
        if (!stepGravityBaseline) {
            stepGravityBaseline = { x: gravityAcceleration.x, y: gravityAcceleration.y, z: gravityAcceleration.z };
            updateStepStatus('Motion sensor ready; hold phone normally and begin walking.');
            return;
        }
        const alpha = 0.08;
        x = gravityAcceleration.x - stepGravityBaseline.x;
        y = gravityAcceleration.y - stepGravityBaseline.y;
        z = gravityAcceleration.z - stepGravityBaseline.z;
        stepGravityBaseline.x += alpha * x;
        stepGravityBaseline.y += alpha * y;
        stepGravityBaseline.z += alpha * z;
    } else {
        updateStepStatus('Motion sensor data is unavailable. Check device sensor permissions.');
        return;
    }

    const magnitude = Math.sqrt(x * x + y * y + z * z);
    if (!Number.isFinite(magnitude)) return;
    stepFilteredMagnitude = stepFilteredMagnitude === 0
        ? magnitude
        : 0.35 * magnitude + 0.65 * stepFilteredMagnitude;
    if (stepFilteredMagnitude >= STEP_PEAK_THRESHOLD) {
        if (!stepPeakStarted) {
            stepPeakStarted = true;
            stepPeakMagnitude = stepFilteredMagnitude;
        } else {
            stepPeakMagnitude = Math.max(stepPeakMagnitude, stepFilteredMagnitude);
        }
    } else if (
        stepPeakStarted
        && stepPeakMagnitude - stepFilteredMagnitude >= 0.3
    ) {
        if (
            stepPeakMagnitude >= STEP_PEAK_THRESHOLD
            && timestamp - stepLastCountedAt >= STEP_MIN_INTERVAL_MS
        ) {
            stepCount += 1;
            stepLastCountedAt = timestamp;
            renderActivityStats();
        }
        stepPeakStarted = false;
        stepPeakMagnitude = 0;
    }
    if (stepTrackingActive) {
        updateStepStatus(`Counting real motion events. ${stepCount} estimated steps so far.`);
    }
}

function hasValidVector(vector) {
    return vector
        && Number.isFinite(vector.x)
        && Number.isFinite(vector.y)
        && Number.isFinite(vector.z);
}

function recordGpsDistance(position, accuracyMeters) {
    if (
        !Number.isFinite(accuracyMeters)
        || accuracyMeters > GPS_DISTANCE_MAX_ACCURACY_METERS
    ) {
        lastGpsDistanceFix = null;
        renderActivityStats();
        return;
    }
    const current = {
        lat: position.coords.latitude,
        lon: position.coords.longitude,
        accuracy: accuracyMeters,
        timestamp: position.timestamp
    };
    if (lastGpsDistanceFix) {
        const elapsedSeconds = (current.timestamp - lastGpsDistanceFix.timestamp) / 1000;
        const distance = haversineDistanceMeters(
            lastGpsDistanceFix.lat,
            lastGpsDistanceFix.lon,
            current.lat,
            current.lon
        );
        const uncertaintyAndNoise =
            lastGpsDistanceFix.accuracy + current.accuracy + GPS_DISTANCE_NOISE_FLOOR_METERS;
        if (
            elapsedSeconds <= 0
            || elapsedSeconds > 120
            || distance / elapsedSeconds > 12
        ) {
            lastGpsDistanceFix = current;
        } else if (distance > uncertaintyAndNoise) {
            gpsDistanceMeters += distance - uncertaintyAndNoise;
            lastGpsDistanceFix = current;
        }
    } else {
        lastGpsDistanceFix = current;
    }
    renderActivityStats();
}

function haversineDistanceMeters(lat1, lon1, lat2, lon2) {
    const radians = value => value * Math.PI / 180;
    const latitudeDelta = radians(lat2 - lat1);
    const longitudeDelta = radians(lon2 - lon1);
    const a = Math.sin(latitudeDelta / 2) ** 2
        + Math.cos(radians(lat1)) * Math.cos(radians(lat2))
        * Math.sin(longitudeDelta / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function renderActivityStats() {
    const steps = document.getElementById('step-count');
    const gpsDistance = document.getElementById('gps-distance');
    const stepDistance = document.getElementById('step-distance');
    if (!steps || !gpsDistance || !stepDistance) return;
    steps.innerText = `${stepCount} estimated steps`;
    gpsDistance.innerText = gpsDistanceMeters > 0
        ? `GPS route estimate: ${Math.round(gpsDistanceMeters)} m (high-confidence fixes only)`
        : `GPS distance: unavailable (need consecutive fixes ≤${GPS_DISTANCE_MAX_ACCURACY_METERS} m accuracy)`;
    const strideInput = document.getElementById('stride-length');
    const strideMeters = strideInput ? Number(strideInput.value) : 0.75;
    stepDistance.innerText = Number.isFinite(strideMeters) && strideMeters > 0
        ? `Step-distance estimate: ~${Math.round(stepCount * strideMeters)} m ` +
            `(using ${strideMeters.toFixed(2)} m/step; adjust stride above)`
        : 'Step-distance estimate: set a valid stride length.';
}

function getCompletedTargetCount() {
    const targetIds = new Set(gameState.targets.map(target => target.id));
    return new Set(gameState.completed.filter(targetId => targetIds.has(targetId))).size;
}

function renderTargets() {
    const list = document.getElementById('target-list');
    list.innerHTML = '';
    const currentTarget = gameState.targets.find(
        target => !gameState.completed.includes(target.id)
    );
    gameState.targets.forEach((t, index) => {
        const isDone = gameState.completed.includes(t.id);
        const li = document.createElement('li');
        if (isDone) li.classList.add('done');
        if (!isDone && currentTarget && currentTarget.id === t.id) {
            li.classList.add('target-focus');
        }
        if (gameState.justVerifiedTargetIds.has(t.id)) {
            li.classList.add('celebrate-target');
        }
        const details = document.createElement('div');
        details.className = 'target-card-content';
        const number = document.createElement('span');
        number.className = 'target-number';
        number.innerText = `FIND ${String(index + 1).padStart(2, '0')}`;
        const name = document.createElement('strong');
        name.innerText = t.name;
        const hint = document.createElement('small');
        hint.innerText = t.hint;
        const targetStatus = document.createElement('span');
        targetStatus.className = 'target-status';
        targetStatus.innerText = isDone
            ? 'Verified'
            : gameState.targetChangeVote
                && gameState.targetChangeVote.target_id === t.id
                && gameState.targetChangeVote.status === 'pending'
                ? 'Awaiting approval'
                : currentTarget && currentTarget.id === t.id
                    ? 'Current target · To find'
                    : 'To find';
        details.append(number, name, hint);
        li.append(details);
        li.append(targetStatus);
        const vote = gameState.targetChangeVote;
        if (
            vote
            && vote.target_id === t.id
            && vote.status === 'pending'
        ) {
            const eligible = vote.eligible_player_ids || [];
            const approved = vote.approved_player_ids || [];
            const approvals = document.createElement('small');
            const voterNames = eligible.map(playerId => {
                const player = gameState.leaderboard.find(item => item.id === playerId);
                const displayName = player ? player.display_name : 'A teammate';
                return `${approved.includes(playerId) ? '✓' : '…'} ${displayName}`;
            });
            approvals.innerText =
                `Waiting for unanimous approval (${approved.length}/${eligible.length}): ` +
                voterNames.join(', ');
            li.append(approvals);
            const myVote = approved.includes(gameState.playerId);
            const approveButton = document.createElement('button');
            approveButton.className = 'secondary';
            approveButton.type = 'button';
            approveButton.innerText = myVote
                ? 'Approval recorded — recheck active roster'
                : 'Approve replacement';
            approveButton.addEventListener('click', () => {
                void castTargetChangeVote(vote.proposal_id, true);
            });
            li.append(approveButton);
            const rejectButton = document.createElement('button');
            rejectButton.className = 'secondary';
            rejectButton.type = 'button';
            rejectButton.innerText = 'Reject and cancel vote';
            rejectButton.addEventListener('click', () => {
                void castTargetChangeVote(vote.proposal_id, false);
            });
            li.append(rejectButton);
        } else if (
            vote
            && vote.target_id === t.id
            && vote.status !== 'pending'
            && vote.message
        ) {
            const voteMessage = document.createElement('small');
            voteMessage.innerText = vote.message;
            li.append(voteMessage);
        }
        if (!isDone) {
            const cameraButton = document.createElement('button');
            cameraButton.className = 'camera-btn';
            cameraButton.type = 'button';
            cameraButton.innerText = 'TAKE PHOTO 📸';
            cameraButton.addEventListener('click', () => openCamera(t.id));
            li.append(cameraButton);
            if ((!vote || vote.status !== 'pending') && !isDone) {
                const requestButton = document.createElement('button');
                requestButton.className = 'secondary';
                requestButton.type = 'button';
                requestButton.innerText = 'Request Target Change';
                requestButton.addEventListener('click', () => {
                    void requestTargetChange(t.id);
                });
                li.append(requestButton);
            }
        }
        list.appendChild(li);
    });
}

async function requestTargetChange(targetId) {
    if (!gameState.id || !gameState.playerId) return;
    const status = document.getElementById('upload-status');
    status.innerText = 'Requesting a team vote to replace this unfinished target…';
    setFeedbackState(status, 'pending');
    try {
        const response = await fetch(
            `${API_URL}/${gameState.id}/targets/${targetId}/replacement-vote`,
            {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ player_id: gameState.playerId })
            }
        );
        const result = await response.json();
        if (!response.ok) {
            status.innerText = typeof result.detail === 'string'
                ? result.detail
                : `Could not request a target change (HTTP ${response.status}).`;
            setFeedbackState(status, 'error');
            return;
        }
        status.innerText = result.target_change_vote?.message
            || 'Target-change vote started. Every active teammate must approve.';
        setFeedbackState(
            status,
            result.target_change_vote?.status === 'approved' ? 'success' : 'pending'
        );
        await pollGameState();
    } catch (error) {
        status.innerText = 'Could not reach the server to request a target change.';
        setFeedbackState(status, 'error');
    }
}

async function castTargetChangeVote(proposalId, approve) {
    if (!gameState.id || !gameState.playerId) return;
    const status = document.getElementById('upload-status');
    status.innerText = approve ? 'Recording your approval…' : 'Cancelling the vote…';
    setFeedbackState(status, 'pending');
    try {
        const response = await fetch(
            `${API_URL}/${gameState.id}/target-change-vote/${proposalId}`,
            {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    player_id: gameState.playerId,
                    approve
                })
            }
        );
        const result = await response.json();
        if (!response.ok) {
            status.innerText = typeof result.detail === 'string'
                ? result.detail
                : `Could not record your vote (HTTP ${response.status}).`;
            setFeedbackState(status, 'error');
            return;
        }
        status.innerText = result.target_change_vote?.message
            || 'Vote recorded. The replacement requires unanimous approval.';
        setFeedbackState(
            status,
            result.target_change_vote?.status === 'approved' ? 'success' :
                result.target_change_vote?.status === 'rejected'
                    || result.target_change_vote?.status === 'unavailable'
                    ? 'error'
                    : 'pending'
        );
        await pollGameState();
    } catch (error) {
        status.innerText = 'Could not reach the server to record your vote.';
        setFeedbackState(status, 'error');
    }
}

function setFeedbackState(element, state) {
    if (!element || !element.classList) return;
    element.classList.remove('is-error', 'is-success', 'is-pending');
    if (state) element.classList.add(`is-${state}`);
}

function announceTargetChange() {
    const vote = gameState.targetChangeVote;
    if (
        vote
        && vote.status === 'approved'
        && vote.proposal_id !== gameState.lastAnnouncedVoteId
    ) {
        document.getElementById('upload-status').innerText = vote.message;
        gameState.lastAnnouncedVoteId = vote.proposal_id;
    }
}

function renderLeaderboard(lb, creatorId = null) {
    const list = document.getElementById('lobby-players');
    list.innerHTML = '';
    lb.forEach(p => {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.innerText = `${p.display_name} · ${p.score} verified`;
        li.appendChild(name);
        if (creatorId && p.id === creatorId) {
            const badge = document.createElement('span');
            badge.className = 'host-badge';
            badge.innerText = 'HOST';
            li.appendChild(badge);
        }
        list.appendChild(li);
    });
    const count = document.getElementById('lobby-player-count');
    if (count) count.innerText = `${lb.length} ${lb.length === 1 ? 'player' : 'players'}`;
    const lobbyStatus = document.getElementById('lobby-status');
    if (lobbyStatus && document.getElementById('screen-lobby').classList.contains('active')) {
        lobbyStatus.innerText = lb.length === 1
            ? 'You can start solo, or wait for friends to join.'
            : `${lb.length} players are ready to head out.`;
    }

    const lbList = document.getElementById('leaderboard-list');
    lbList.innerHTML = list.innerHTML;
}

function startLocationTracking() {
    stopLocationTracking();
    locationObservationSequence = 0;
    const trackingGeneration = locationTrackingGeneration;
    const trackedGameId = gameState.id;
    const trackedPlayerId = gameState.playerId;
    if (!trackedGameId || !trackedPlayerId) return;
    if (!navigator.geolocation || !navigator.geolocation.watchPosition) {
        reportLocationIssue('GPS unavailable: this browser does not support continuous geolocation.');
        return;
    }

    document.getElementById('loc-status').innerText = 'Requesting a fresh high-accuracy GPS fix…';
    locationWatchId = navigator.geolocation.watchPosition(position => {
        if (!isCurrentLocationSession(trackingGeneration, trackedGameId, trackedPlayerId)) return;

        const observationSequence = ++locationObservationSequence;
        lastLocationFix = {
            timestamp: position.timestamp,
            accuracy: position.coords.accuracy
        };
        document.getElementById('loc-status').innerText =
            `Fresh GPS fix received (${Math.round(position.coords.accuracy)} m accuracy); validating…`;
        if (locationRequestPending) {
            pendingLocationPosition = { position, observationSequence };
            return;
        }
        return submitLocationPosition(
            position,
            trackingGeneration,
            trackedGameId,
            trackedPlayerId,
            observationSequence
        );
    }, error => {
        if (!isCurrentLocationSession(trackingGeneration, trackedGameId, trackedPlayerId)) return;
        locationObservationSequence += 1;
        const messages = {
            1: 'GPS permission denied. Allow location access in your browser settings.',
            2: 'GPS unavailable. Check device location services and try again.',
            3: 'GPS request timed out. Move to an open area and retry.'
        };
        console.warn('Browser geolocation failed', {
            code: error.code,
            message: error.message
        });
        reportLocationIssue(
            `${messages[error.code] || 'GPS unavailable. Check device location services and retry.'} ` +
            `(code ${error.code}: ${error.message})`
        );
    }, { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 });
}

function stopLocationTracking() {
    locationTrackingGeneration += 1;
    if (locationWatchId !== null && navigator.geolocation) {
        navigator.geolocation.clearWatch(locationWatchId);
    }
    locationWatchId = null;
    locationRequestPending = false;
    pendingLocationPosition = null;
}

async function submitLocationPosition(
    position,
    trackingGeneration,
    gameId,
    playerId,
    observationSequence
) {
    if (!isCurrentLocationSession(trackingGeneration, gameId, playerId)) return;
    locationRequestPending = true;
    document.getElementById('loc-status').innerText =
        `Submitting fresh GPS fix (${Math.round(position.coords.accuracy)} m accuracy)…`;
    try {
        const response = await fetch(`${API_URL}/${gameId}/players/${playerId}/location`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                lat: position.coords.latitude,
                lon: position.coords.longitude,
                accuracy_m: position.coords.accuracy,
                timestamp: new Date(position.timestamp).toISOString()
            })
        });
        const result = await response.json();
        if (!isCurrentLocationSession(trackingGeneration, gameId, playerId)) return;
        if (!response.ok) {
            console.warn('Location API request rejected', {
                status: response.status,
                detail: summarizeLocationDetail(result.detail)
            });
            if (observationSequence === locationObservationSequence) {
                reportLocationIssue(formatLocationApiError(response.status, result.detail));
            }
            return;
        }
        if (!result.accepted) {
            console.warn('Fresh GPS fix rejected by backend', {
                accuracy: position.coords.accuracy,
                reason: result.message
            });
            if (observationSequence === locationObservationSequence) {
                reportLocationIssue(
                    `Fresh GPS fix received (${Math.round(position.coords.accuracy)} m), ` +
                    `but rejected: ${result.message}`
                );
            }
            return;
        }

        if (observationSequence === locationObservationSequence) {
            locationIssue = null;
            renderLocationContext(result.location_context);
            const precision = result.location_quality === 'approximate'
                ? 'Approximate location'
                : 'Higher-confidence location';
            const locality = result.location_context && result.location_context.locality
                ? ` near ${result.location_context.locality}`
                : '';
            const movementStatus = result.movement_accepted
                ? ''
                : `; movement not added (${result.message})`;
            const contextStatus = result.context_refreshed
                ? '; nearby tasks refreshed'
                : '';
            document.getElementById('loc-status').innerText =
                `${precision} ±${Math.round(result.location_accuracy_m)} m${locality}` +
                `${movementStatus}${contextStatus}`;
        }
        recordGpsDistance(position, result.location_accuracy_m);
        if (result.context_refreshed) await pollGameState();
        if (result.context_refresh_notice) {
            document.getElementById('upload-status').innerText = result.context_refresh_notice;
        }
    } catch (error) {
        if (!isCurrentLocationSession(trackingGeneration, gameId, playerId)) return;
        console.warn('Location API request failed', {
            message: error instanceof Error ? error.message : String(error)
        });
        if (observationSequence === locationObservationSequence) {
            reportLocationIssue('Location API request failed; check your connection and retry.');
        }
    } finally {
        if (!isCurrentLocationSession(trackingGeneration, gameId, playerId)) return;
        locationRequestPending = false;
        const queuedFix = pendingLocationPosition;
        pendingLocationPosition = null;
        if (queuedFix) {
            void submitLocationPosition(
                queuedFix.position,
                trackingGeneration,
                gameId,
                playerId,
                queuedFix.observationSequence
            );
        }
    }
}

function isCurrentLocationSession(generation, gameId, playerId) {
    return generation === locationTrackingGeneration
        && gameState.id === gameId
        && gameState.playerId === playerId;
}

function retryLocationTracking() {
    if (!gameState.id || !gameState.playerId) return;
    lastLocationFix = null;
    document.getElementById('loc-status').innerText = 'Retrying: requesting a fresh high-accuracy GPS fix…';
    startLocationTracking();
}

function reportLocationIssue(message) {
    locationIssue = message;
    document.getElementById('loc-status').innerText = message;
}

function summarizeLocationDetail(detail) {
    if (!Array.isArray(detail)) return detail;
    return detail.map(issue => ({
        location: issue.loc,
        type: issue.type,
        message: issue.msg
    }));
}

function formatLocationApiError(status, detail) {
    if (status === 422 && Array.isArray(detail)) {
        const problems = detail.map(issue => {
            const field = Array.isArray(issue.loc)
                ? issue.loc.filter(part => part !== 'body').join('.')
                : 'request';
            return `${field}: ${issue.msg}`;
        });
        return `Location API validation error (422): ${problems.join('; ')}`;
    }
    const message = typeof detail === 'string' ? detail : 'Request rejected.';
    return `Location API error (HTTP ${status}): ${message}`;
}

async function openCamera(targetId) {
    currentTargetId = targetId;
    const target = gameState.targets.find(item => item.id === targetId);
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-preview');
    const switchButton = document.getElementById('camera-switch');
    document.getElementById('camera-target').innerText = target
        ? `${target.name}: ${target.hint}`
        : 'Target';
    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.innerText = '';
    setFeedbackState(uploadStatus, '');
    const requestGeneration = ++cameraRequestGeneration;
    modal.hidden = false;
    if (!cameraCancelHandlerRegistered) {
        modal.addEventListener('cancel', event => {
            event.preventDefault();
            cancelCamera();
        });
        cameraCancelHandlerRegistered = true;
    }
    if (typeof modal.showModal === 'function') {
        modal.showModal();
    } else {
        modal.setAttribute('open', '');
        modal.hidden = false;
    }

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera access requires a supported browser and secure context.');
        }
        gameState.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: gameState.cameraFacingMode } },
            audio: false
        });
        if (requestGeneration !== cameraRequestGeneration || !gameState.id) {
            gameState.cameraStream.getTracks().forEach(track => track.stop());
            gameState.cameraStream = null;
            return;
        }
        video.srcObject = gameState.cameraStream;
        await video.play();
        const devices = await navigator.mediaDevices.enumerateDevices();
        switchButton.hidden = devices.filter(device => device.kind === 'videoinput').length < 2;
    } catch (error) {
        if (requestGeneration !== cameraRequestGeneration) return;
        cancelCamera();
        uploadStatus.innerText =
            error.message || 'Unable to open the camera. Check permission and use HTTPS or localhost.';
        setFeedbackState(uploadStatus, 'error');
    }
}

async function switchCamera() {
    gameState.cameraFacingMode = gameState.cameraFacingMode === 'environment' ? 'user' : 'environment';
    const targetId = currentTargetId;
    stopCamera();
    await openCamera(targetId);
}

function capturePhoto() {
    const video = document.getElementById('camera-preview');
    if (!video.videoWidth || !video.videoHeight || !currentTargetId) {
        const status = document.getElementById('upload-status');
        status.innerText = 'Camera preview is not ready.';
        setFeedbackState(status, 'error');
        return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const targetId = currentTargetId;
    canvas.toBlob(async blob => {
        stopCamera();
        if (!blob) {
            const status = document.getElementById('upload-status');
            status.innerText = 'Could not capture an image.';
            setFeedbackState(status, 'error');
            return;
        }
        await uploadCapturedPhoto(blob, targetId);
    }, 'image/jpeg', 0.9);
}

function stopCamera() {
    cameraRequestGeneration += 1;
    if (gameState.cameraStream) {
        gameState.cameraStream.getTracks().forEach(track => track.stop());
        gameState.cameraStream = null;
    }
    document.getElementById('camera-preview').srcObject = null;
    const modal = document.getElementById('camera-modal');
    if (modal.open && typeof modal.close === 'function') {
        modal.close();
    } else {
        modal.removeAttribute('open');
    }
    modal.hidden = true;
    currentTargetId = null;
}

function cancelCamera() {
    stopCamera();
}

async function uploadCapturedPhoto(blob, targetId) {
    const statusEl = document.getElementById('upload-status');
    statusEl.innerText = 'Checking captured photo…';
    setFeedbackState(statusEl, 'pending');
    const formData = new FormData();
    formData.append("player_id", gameState.playerId);
    formData.append("target_id", targetId);
    formData.append("image", blob, `capture-${Date.now()}.jpg`);

    try {
        const res = await fetch(`${API_URL}/${gameState.id}/submit`, {
            method: 'POST',
            body: formData
        });
        const result = await res.json();
        if (!res.ok) {
            statusEl.innerText = formatSubmissionError(res.status, result.detail);
            setFeedbackState(statusEl, 'error');
            return;
        }

        renderProgress(result.completed_targets, result.total_targets);
        if (result.success) {
            statusEl.innerText = result.message;
            setFeedbackState(statusEl, 'success');
        } else {
            statusEl.innerText = result.message === 'Location not available'
                ? locationIssue
                    ? `Photo verification needs a saved GPS fix. Last location error: ${locationIssue}`
                    : 'No accepted GPS fix is saved for this player. Wait for a reliable fix, then retry the photo.'
                : result.message || 'Target not found in image.';
            setFeedbackState(statusEl, 'error');
        }
        await pollGameState();
    } catch(e) {
        statusEl.innerText = "Error uploading photo.";
        setFeedbackState(statusEl, 'error');
    }
}

function formatSubmissionError(status, detail) {
    if (typeof detail === 'string' && detail === 'Location not available') {
        return locationIssue
            ? `Photo verification needs a saved GPS fix. Last location error: ${locationIssue}`
            : 'No accepted GPS fix is saved for this player. Wait for a reliable fix, then retry the photo.';
    }
    return typeof detail === 'string' ? detail : `Verification failed (HTTP ${status}).`;
}

loadRuntimeMode();
showResumeOption();
