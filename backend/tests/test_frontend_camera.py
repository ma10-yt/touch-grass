from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_camera_uses_media_devices_and_uploads_captured_blob():
    assert "navigator.mediaDevices.getUserMedia" in APP_JS
    assert "function capturePhoto()" in APP_JS
    assert "canvas.toBlob" in APP_JS
    assert 'formData.append("image", blob,' in APP_JS


def test_active_game_has_no_gallery_file_picker():
    assert '<input type="file"' not in INDEX_HTML
    assert 'id="camera-preview"' in INDEX_HTML
    assert 'id="camera-capture"' in INDEX_HTML
    assert 'id="camera-switch"' in INDEX_HTML
    assert 'id="runtime-mode"' in INDEX_HTML
    assert "DEMO MODE" in APP_JS


def test_progress_is_rendered_as_completed_targets_over_total():
    assert "getCompletedTargetCount()" in APP_JS
    assert "completed / total" in APP_JS
    assert "Progress: ${completed}/${total} (${percent}%)" in APP_JS
    assert "95%" not in APP_JS


def test_optional_activity_stats_and_actual_location_context_are_visible():
    assert 'id="step-count"' in INDEX_HTML
    assert 'id="gps-distance"' in INDEX_HTML
    assert 'id="step-distance"' in INDEX_HTML
    assert "context.environment_type.replaceAll('_', ' ')" in APP_JS
    assert "result.message" in APP_JS
    assert "accuracy_m: position.coords.accuracy" in APP_JS
    assert "new Date(position.timestamp).toISOString()" in APP_JS
    assert "touch-grass-session" in APP_JS
    assert "\nfunction renderLocationContext(context)" in APP_JS
    assert "\nfunction renderActivityStats()" in APP_JS


def test_location_tracking_reports_gps_and_api_errors():
    assert "accuracy_m: position.coords.accuracy" in APP_JS
    assert "GPS permission denied." in APP_JS
    assert "GPS unavailable." in APP_JS
    assert "GPS request timed out." in APP_JS
    assert "formatLocationApiError(response.status, result.detail)" in APP_JS
    assert "status: response.status" in APP_JS
    assert "detail: summarizeLocationDetail(result.detail)" in APP_JS
    assert "code: error.code" in APP_JS
    assert "message: error.message" in APP_JS
    assert "Location API validation error (422)" in APP_JS
    assert "No accepted GPS fix is saved for this player." in APP_JS
    assert "Last location error: ${locationIssue}" in APP_JS
    assert "but rejected: ${result.message}" in APP_JS
    assert "observationSequence === locationObservationSequence" in APP_JS


def test_game_resume_is_intentional_and_leave_returns_to_landing():
    assert "restoreGameSession();" not in APP_JS
    assert "showResumeOption();" in APP_JS
    assert 'id="btn-resume" class="secondary" onclick="restoreGameSession()" hidden' in INDEX_HTML
    assert "function leaveGame()" in APP_JS
    assert "localStorage.removeItem('touch-grass-session')" in APP_JS
    assert "showScreen('screen-landing')" in APP_JS
    assert "location.reload()" not in APP_JS
    assert 'onclick="leaveGame()">Leave Game</button>' in INDEX_HTML


def test_new_game_uses_backend_identity_and_resets_game_specific_state():
    assert "gameState.id = data.id;" in APP_JS
    assert "gameState.playerId = data.creator_id;" in APP_JS
    assert "resetClientGameState({ clearSavedSession: true });" in APP_JS
    assert "locationIssue = null;" in APP_JS
    assert "lastLocationFix = null;" in APP_JS
    assert "gameState.completed = [];" in APP_JS
    assert "gameState.targets = [];" in APP_JS


def test_location_watch_uses_fresh_high_accuracy_fixes_and_guards_old_callbacks():
    assert "navigator.geolocation.watchPosition" in APP_JS
    assert "navigator.geolocation.clearWatch(locationWatchId)" in APP_JS
    assert "maximumAge: 0" in APP_JS
    assert "enableHighAccuracy: true" in APP_JS
    assert "isCurrentLocationSession(trackingGeneration, trackedGameId, trackedPlayerId)" in APP_JS
    assert "if (!result.accepted) {" in APP_JS
    assert "function retryLocationTracking()" in APP_JS


def test_osm_attribution_and_host_solo_start_button_are_present():
    assert 'id="osm-attribution"' in INDEX_HTML
    assert "OpenStreetMap contributors" in INDEX_HTML
    assert 'id="btn-start"' in INDEX_HTML
