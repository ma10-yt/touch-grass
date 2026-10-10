const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appSource = fs.readFileSync(
    path.join(__dirname, '..', 'app.js'),
    'utf8'
);

function createApp({
    savedSession = null,
    fetchHandler = async () => response({}),
    motionSupported = true,
    motionPermission = 'granted'
} = {}) {
    const elements = new Map();
    const makeElement = id => {
        const names = new Set(id === 'screen-landing' ? ['active'] : []);
        const element = {
            id,
            hidden: id === 'camera-modal',
            innerText: id === 'stride-length' ? '0.75' : '',
            value: id === 'stride-length' ? '0.75' : '',
            disabled: false,
            style: {},
            children: [],
            append(...children) { this.children.push(...children); },
            appendChild(child) { this.children.push(child); },
            addEventListener() {},
            classList: {
                get active() { return names.has('active'); },
                set active(value) {
                    if (value) names.add('active');
                    else names.delete('active');
                },
                add(...classNames) { classNames.forEach(name => names.add(name)); },
                remove(...classNames) { classNames.forEach(name => names.delete(name)); },
                contains(name) { return names.has(name); }
            }
        };
        let innerHTML = '';
        Object.defineProperty(element, 'innerHTML', {
            get() { return innerHTML; },
            set(value) {
                innerHTML = value;
                if (value === '') element.children = [];
            }
        });
        return element;
    };
    const screenIds = [
        'screen-landing',
        'screen-create',
        'screen-join',
        'screen-lobby',
        'screen-active',
        'screen-leaderboard',
        'screen-winner'
    ];
    for (const id of [
        ...screenIds,
        'runtime-mode', 'context-status', 'osm-attribution', 'create-name',
        'create-status', 'btn-create', 'btn-resume', 'lobby-code',
        'lobby-players', 'leaderboard-list', 'target-list', 'score',
        'progress', 'step-count', 'step-distance', 'gps-distance', 'stride-length',
        'step-status', 'btn-step-counter', 'loc-status', 'upload-status',
        'btn-start', 'camera-preview', 'camera-modal', 'lobby-status',
        'lobby-player-count', 'copy-status', 'winner-name', 'winner-targets',
        'winner-activity', 'btn-copy-code'
    ]) {
        elements.set(id, makeElement(id));
    }
    const progressFill = makeElement('progress-fill');

    const storage = new Map();
    if (savedSession) storage.set('touch-grass-session', JSON.stringify(savedSession));
    const watches = new Map();
    const clearedWatches = [];
    const fetchCalls = [];
    const motionListeners = new Map();
    const documentListeners = new Map();
    const timeouts = new Map();
    let nextWatchId = 1;
    let nextTimerId = 1;
    class MockDeviceMotionEvent {}
    if (motionSupported && motionPermission !== null) {
        MockDeviceMotionEvent.requestPermission = async () => motionPermission;
    }
    const sandbox = {
        console: { warn() {}, error() {} },
        document: {
            visibilityState: 'visible',
            addEventListener(name, callback) { documentListeners.set(name, callback); },
            getElementById(id) {
                if (!elements.has(id)) throw new Error(`Missing test element: ${id}`);
                return elements.get(id);
            },
            querySelectorAll(selector) {
                return selector === '.screen'
                    ? screenIds.map(id => elements.get(id))
                    : [];
            },
            querySelector(selector) {
                return selector === '.progress-fill' ? progressFill : null;
            },
            createElement(tagName) { return makeElement(tagName); }
        },
        ...(motionSupported ? { DeviceMotionEvent: MockDeviceMotionEvent } : {}),
        addEventListener(name, callback) {
            motionListeners.set(name, [...(motionListeners.get(name) || []), callback]);
        },
        removeEventListener(name, callback) {
            motionListeners.set(
                    name,
                    (motionListeners.get(name) || []).filter(listener => listener !== callback)
            );
        },
        navigator: {
            geolocation: {
                watchPosition(success, error, options) {
                    const id = nextWatchId++;
                    watches.set(id, { success, error, options });
                    return id;
                },
                clearWatch(id) {
                    clearedWatches.push(id);
                    watches.delete(id);
                }
            }
        },
        localStorage: {
            getItem(key) { return storage.get(key) || null; },
            setItem(key, value) { storage.set(key, value); },
            removeItem(key) { storage.delete(key); }
        },
        fetch: async (...args) => {
            fetchCalls.push(args);
            return fetchHandler(...args);
        },
        setInterval() { return nextTimerId++; },
        clearInterval() {},
        setTimeout(callback) {
            const id = nextTimerId++;
            timeouts.set(id, callback);
            return id;
        },
        clearTimeout(id) { timeouts.delete(id); },
        alert() {},
        Date,
        Set,
        Map,
        URLSearchParams,
        FormData: class {},
        location: { reload() { throw new Error('Leave must not reload the page'); } }
    };
    vm.createContext(sandbox);
    vm.runInContext(appSource, sandbox, { filename: 'app.js' });
    return {
        sandbox,
        elements,
        progressFill,
        storage,
        watches,
        motionListeners,
        documentListeners,
        timeouts,
        clearedWatches,
        fetchCalls,
        state(expression) {
            return vm.runInContext(expression, sandbox);
        }
    };
}

function response(body, status = 200) {
    return {
        ok: status >= 200 && status < 300,
        status,
        async json() { return body; }
    };
}

function gpsPosition({ accuracy = 5, timestamp = Date.now() } = {}) {
    return {
        timestamp,
        coords: { latitude: 43, longitude: -79, accuracy }
    };
}

test('stored game is offered for intentional resume, not automatically restored', () => {
    const app = createApp({
        savedSession: { id: 'old-game', playerId: 'old-player', joinCode: 'OLD' }
    });

    test('progress and target cards reflect verified and current objectives', () => {
        const app = createApp();
        vm.runInContext(
            "gameState.targets = [" +
                "{ id: 'target-1', name: 'Oak tree', hint: 'Look for a broad canopy.' }," +
                "{ id: 'target-2', name: 'Bench', hint: 'A place to rest.' }" +
            "]; gameState.completed = ['target-2']; renderProgress(1, 2); renderTargets();",
            app.sandbox
        );

        assert.equal(app.elements.get('progress').innerText, 'Progress: 1/2 (50%)');
        assert.equal(app.progressFill.style.width, '50%');
        const cards = app.elements.get('target-list').children;
        assert.equal(cards.length, 2);
        assert.equal(cards[0].classList.contains('target-focus'), true);
        assert.equal(cards[0].children[1].innerText, 'Current target · To find');
        assert.equal(cards[1].classList.contains('done'), true);
        assert.equal(cards[1].children[1].innerText, 'Verified');
    });

    assert.equal(app.state('gameState.id'), null);
    assert.equal(app.elements.get('btn-resume').hidden, false);
    assert.equal(app.watches.size, 0);
    assert.equal(app.fetchCalls.length, 1);
});

test('target-change request and ballot use server-persisted vote endpoints', async () => {
    const app = createApp({
        fetchHandler: async () => response({
            target_change_vote: {
                proposal_id: 'proposal-1',
                status: 'pending',
                eligible_player_ids: ['player-1', 'player-2'],
                approved_player_ids: ['player-1']
            }
        })
    });
    app.sandbox.pollGameState = async () => {};
    vm.runInContext(
        "gameState.id = 'game-1'; gameState.playerId = 'player-1';",
        app.sandbox
    );

    await app.sandbox.requestTargetChange('target-1');
    await app.sandbox.castTargetChangeVote('proposal-1', true);

    const voteCalls = app.fetchCalls.filter(([url]) => url.startsWith('/api/games/'));
    assert.equal(
        voteCalls[0][0],
        '/api/games/game-1/targets/target-1/replacement-vote'
    );
    assert.deepEqual(JSON.parse(voteCalls[0][1].body), {
        player_id: 'player-1'
    });
    assert.equal(
        voteCalls[1][0],
        '/api/games/game-1/target-change-vote/proposal-1'
    );
    assert.deepEqual(JSON.parse(voteCalls[1][1].body), {
        player_id: 'player-1',
        approve: true
    });
});

test('new game uses the fresh IDs returned by the create API and resets prior state', async () => {
    let createCount = 0;
    const app = createApp({
        fetchHandler: async url => {
            if (url === '/api/health') return response({ demo_mode: false });
            createCount += 1;
            return response({
                id: `new-game-${createCount}`,
                creator_id: `new-host-${createCount}`,
                join_code: `CODE${createCount}`
            });
        }
    });

    await app.sandbox.doCreate('Walker', null, null);
    const oldWatchId = Array.from(app.watches.keys())[0];
    app.sandbox.reportLocationIssue('old game GPS error', 3);
    app.state('stepCount = 17; gpsDistanceMeters = 42;');
    await app.sandbox.doCreate('Walker', null, null);

    assert.equal(app.state('gameState.id'), 'new-game-2');
    assert.equal(app.state('gameState.playerId'), 'new-host-2');
    assert.equal(app.state('gameState.joinCode'), 'CODE2');
    assert.equal(app.state('locationIssue'), null);
    assert.equal(app.state('lastLocationFix'), null);
    assert.equal(app.state('stepCount'), 0);
    assert.equal(app.state('gpsDistanceMeters'), 0);
    assert.equal(app.elements.get('step-count').innerText, '0 estimated steps');
    assert.equal(app.storage.get('touch-grass-session').includes('new-game-2'), true);
    assert.equal(app.clearedWatches.includes(oldWatchId), true);
});

test('leaving clears local session and client state without a backend delete request', () => {
    const app = createApp({
        savedSession: { id: 'active-game', playerId: 'active-player', joinCode: 'JOIN' }
    });
    vm.runInContext(
        "gameState.id = 'active-game'; gameState.playerId = 'active-player'; startLocationTracking();",
        app.sandbox
    );
    const watchId = Array.from(app.watches.keys())[0];
    app.sandbox.leaveGame();

    assert.equal(app.storage.has('touch-grass-session'), false);
    assert.equal(app.state('gameState.id'), null);
    assert.equal(app.state('gameState.playerId'), null);
    assert.equal(app.clearedWatches.includes(watchId), true);
    assert.equal(app.elements.get('screen-landing').classList.active, true);
    assert.equal(app.fetchCalls.length, 1);
});

test('callbacks from an old location watch cannot update the new game', async () => {
    const app = createApp({
        fetchHandler: async url => url === '/api/health'
            ? response({ demo_mode: false })
            : response({ accepted: true, movement_meters: 10 })
    });
    vm.runInContext(
        "gameState.id = 'old-game'; gameState.playerId = 'old-player'; startLocationTracking();",
        app.sandbox
    );
    const oldWatch = app.watches.get(1);
    vm.runInContext(
        "gameState.id = 'new-game'; gameState.playerId = 'new-player'; startLocationTracking();",
        app.sandbox
    );
    const locationStatus = app.elements.get('loc-status').innerText;

    await oldWatch.success(gpsPosition());

    assert.equal(app.fetchCalls.length, 1);
    assert.equal(app.elements.get('loc-status').innerText, locationStatus);
});

test('rejected GPS reading shows the fresh accuracy and never renders movement', async () => {
    const app = createApp({
        fetchHandler: async url => url === '/api/health'
            ? response({ demo_mode: false })
            : response({
                accepted: false,
                message: 'GPS accuracy is 213 m; wait for a fix within 30 m.',
                movement_meters: 500,
            })
    });
    vm.runInContext(
        "gameState.id = 'game'; gameState.playerId = 'player'; startLocationTracking();",
        app.sandbox
    );
    const watch = app.watches.get(1);

    await watch.success(gpsPosition({ accuracy: 213 }));

    assert.match(app.elements.get('loc-status').innerText, /Fresh GPS fix received \(213 m\)/);
    assert.match(app.elements.get('loc-status').innerText, /rejected/);
    assert.equal(watch.options.enableHighAccuracy, true);
    assert.equal(watch.options.maximumAge, 0);
});

test('a fresh accepted fix clears an earlier GPS error', async () => {
    const app = createApp({
        fetchHandler: async url => url === '/api/health'
            ? response({ demo_mode: false })
            : response({
                accepted: true,
                message: 'Location updated.',
                movement_meters: 0,
                movement_accepted: true,
                location_accuracy_m: 8,
                location_quality: 'higher-confidence',
                location_context: {
                    locality: 'Test District',
                    context_mode: 'contextual',
                    environment_type: 'urban',
                    context_provider: 'openstreetmap',
                    context_status: 'available'
                },
            })
    });
    vm.runInContext(
        "gameState.id = 'game'; gameState.playerId = 'player'; " +
        "reportLocationIssue('old poor-accuracy fix', 3); startLocationTracking();",
        app.sandbox
    );

    await app.watches.get(1).success(gpsPosition({ accuracy: 8 }));

    assert.equal(app.state('locationIssue'), null);
    assert.match(app.elements.get('loc-status').innerText, /Higher-confidence location ±8 m near Test District/);
    assert.match(app.elements.get('context-status').innerText, /near Test District/);
});

test('step detection filters small acceleration noise', async () => {
    const app = createApp();
    vm.runInContext("gameState.id = 'game'; gameState.playerId = 'player';", app.sandbox);
    await app.sandbox.startStepCounter();
    const listener = app.motionListeners.get('devicemotion')[0];

    for (let index = 0; index < 30; index += 1) {
        listener({
            acceleration: { x: index % 2 ? 0.6 : 0.4, y: 0, z: 0 },
            timeStamp: 1000 + index * 50
        });
    }

    assert.equal(app.state('stepCount'), 0);
    assert.match(app.elements.get('step-status').innerText, /estimated steps so far/);
});

test('step detector counts distinct peaks and suppresses duplicates within the minimum interval', async () => {
    const app = createApp();
    vm.runInContext("gameState.id = 'game'; gameState.playerId = 'player';", app.sandbox);
    await app.sandbox.startStepCounter();
    const listener = app.motionListeners.get('devicemotion')[0];
    const pulse = start => [0, 0, 5, 0, 0].forEach((x, index) => listener({
        acceleration: { x, y: 0, z: 0 },
        timeStamp: start + index * 50
    }));

    pulse(1000);
    pulse(2300);
    pulse(3200);

    assert.equal(app.state('stepCount'), 2);
    assert.equal(app.elements.get('step-count').innerText, '2 estimated steps');
});

test('unsupported motion sensors and denied permission show useful status', async () => {
    const unsupported = createApp({ motionSupported: false });
    vm.runInContext("gameState.id = 'game'; gameState.playerId = 'player';", unsupported.sandbox);
    await unsupported.sandbox.startStepCounter();
    assert.match(unsupported.elements.get('step-status').innerText, /unsupported/);
    assert.equal(unsupported.motionListeners.size, 0);

    const denied = createApp({ motionPermission: 'denied' });
    vm.runInContext("gameState.id = 'game'; gameState.playerId = 'player';", denied.sandbox);
    await denied.sandbox.startStepCounter();
    assert.match(denied.elements.get('step-status').innerText, /permission denied/);
    assert.equal(denied.motionListeners.size, 0);
});

test('step listeners are unique while active, removable on stop, and report unavailable data', async () => {
    const app = createApp();
    vm.runInContext("gameState.id = 'game'; gameState.playerId = 'player';", app.sandbox);
    await app.sandbox.startStepCounter();
    await app.sandbox.startStepCounter();
    assert.equal(app.motionListeners.get('devicemotion').length, 1);

    const timer = Array.from(app.timeouts.values())[0];
    timer();
    assert.match(app.elements.get('step-status').innerText, /No motion data received/);

    app.sandbox.stopStepCounter();
    assert.equal(app.motionListeners.get('devicemotion').length, 0);
    assert.equal(app.elements.get('btn-step-counter').innerText, 'Start Step Counter');
});

test('backgrounding suspends sensor tracking and old sensor callbacks cannot alter a new game', async () => {
    let createCount = 0;
    const app = createApp({
        fetchHandler: async url => url === '/api/health'
            ? response({ demo_mode: false })
            : response({
                id: `new-game-${++createCount}`,
                creator_id: `player-${createCount}`,
                join_code: `CODE${createCount}`
            })
    });
    await app.sandbox.doCreate('Walker', null, null);
    await app.sandbox.startStepCounter();
    const oldCallback = app.motionListeners.get('devicemotion')[0];
    const pulse = start => [0, 0, 5, 0, 0].forEach((x, index) => oldCallback({
        acceleration: { x, y: 0, z: 0 },
        timeStamp: start + index * 50
    }));
    pulse(1000);
    pulse(2300);
    assert.ok(app.state('stepCount') > 0);

    app.sandbox.document.visibilityState = 'hidden';
    app.documentListeners.get('visibilitychange')();
    assert.match(app.elements.get('step-status').innerText, /paused in the background/);

    await app.sandbox.doCreate('Walker', null, null);
    const newGameStatus = app.elements.get('step-status').innerText;
    oldCallback({
        acceleration: { x: 5, y: 0, z: 0 },
        timeStamp: 1000
    });

    assert.equal(app.state('stepCount'), 0);
    assert.equal(app.elements.get('step-status').innerText, newGameStatus);
});

test('GPS distance is separate and does not bridge across inaccurate fixes', () => {
    const app = createApp();
    app.sandbox.recordGpsDistance(
        { coords: { latitude: 43, longitude: -79 }, timestamp: 1000 },
        8
    );
    app.sandbox.recordGpsDistance(
        { coords: { latitude: 43, longitude: -78.998 }, timestamp: 60000 },
        8
    );
    const gpsLabel = app.elements.get('gps-distance').innerText;
    assert.match(gpsLabel, /GPS route estimate/);

    app.sandbox.recordGpsDistance(
        { coords: { latitude: 43, longitude: -78.997 }, timestamp: 90000 },
        213
    );
    assert.equal(app.elements.get('gps-distance').innerText, gpsLabel);
    app.sandbox.recordGpsDistance(
        { coords: { latitude: 43, longitude: -78.996 }, timestamp: 120000 },
        8
    );
    assert.equal(app.elements.get('gps-distance').innerText, gpsLabel);
    app.sandbox.recordGpsDistance(
        { coords: { latitude: 43, longitude: -78.995 }, timestamp: 150000 },
        8
    );
    assert.notEqual(app.elements.get('gps-distance').innerText, gpsLabel);
    assert.match(app.elements.get('step-distance').innerText, /Step-distance estimate/);
    assert.match(app.elements.get('step-distance').innerText, /0\.75 m\/step/);
});
