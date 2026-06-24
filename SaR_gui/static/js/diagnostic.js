/**
 * Diagnostic Script for Multi-Player Game Testing
 *
 * HOW TO USE:
 * 1. Open browser console (F12 → Console)
 * 2. Paste this command:
 *    fetch('/static/js/diagnostic.js').then(r => r.text()).then(eval)
 * 3. Run tests with:
 *    runAllDiagnostics()  // Run all checks
 * 4. Or specific tests:
 *    testServerConnectivity()
 *    testJoinSession()
 *    testGameState()
 */

console.log("=== DIAGNOSTIC SCRIPT LOADED ===");

// Color output for better readability
const log = {
    success: (msg) => console.log(`✅ ${msg}`),
    error: (msg) => console.error(`❌ ${msg}`),
    info: (msg) => console.log(`ℹ️  ${msg}`),
    warning: (msg) => console.warn(`⚠️  ${msg}`),
    header: (msg) => console.log(`\n${'='.repeat(50)}\n${msg}\n${'='.repeat(50)}`),
};

/**
 * Test 1: Server Connectivity
 */
async function testServerConnectivity() {
    log.header("Test 1: Server Connectivity");

    try {
        const response = await fetch('http://localhost:3000/', {
            method: 'GET',
            timeout: 3000
        });

        if (response.ok) {
            log.success(`Server is responding (Status: ${response.status})`);
            return true;
        } else {
            log.error(`Server returned status ${response.status}`);
            return false;
        }
    } catch (error) {
        log.error(`Cannot connect to server: ${error.message}`);
        log.info("Make sure Flask server is running: python main.py --task-type official --condition normal --session-id session1");
        return false;
    }
}

/**
 * Test 2: Join Session Endpoint
 */
async function testJoinSession() {
    log.header("Test 2: Join Session Endpoint");

    const sessionId = 'diagnostic-test';
    log.info(`Attempting to join session: ${sessionId}`);

    try {
        // First player
        const response1 = await fetch(`http://localhost:3000/join-session/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response1.ok) {
            log.error(`Failed to join (Status: ${response1.status})`);
            const error = await response1.json();
            log.error(`Error: ${error.error}`);
            return false;
        }

        const data1 = await response1.json();
        log.success(`Player 1 joined successfully`);
        log.info(`Assigned as: ${data1.agent_id}`);
        log.info(`Session: ${data1.session_id}`);

        // Second player (same session)
        const response2 = await fetch(`http://localhost:3000/join-session/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response2.ok) {
            log.error(`Failed to join (Status: ${response2.status})`);
            return false;
        }

        const data2 = await response2.json();
        log.success(`Player 2 joined successfully`);
        log.info(`Assigned as: ${data2.agent_id}`);

        // Third player (should fail)
        const response3 = await fetch(`http://localhost:3000/join-session/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response3.status === 400) {
            const error = await response3.json();
            log.success(`Third player correctly rejected: "${error.error}"`);
        } else {
            log.warning(`Third player not rejected (got status ${response3.status})`);
        }

        return true;

    } catch (error) {
        log.error(`Join session test failed: ${error.message}`);
        return false;
    }
}

/**
 * Test 3: Game State Endpoint
 */
async function testGameState() {
    log.header("Test 3: Game State Endpoint");

    const sessionId = 'test-session';
    log.info(`Checking game state for session: ${sessionId}`);

    try {
        const response = await fetch(`http://localhost:3000/game-state/${sessionId}`, {
            method: 'GET'
        });

        if (!response.ok) {
            log.error(`Failed to get game state (Status: ${response.status})`);
            return false;
        }

        const data = await response.json();
        log.success(`Game state retrieved successfully`);
        log.info(`Players connected: ${data.players_connected}`);
        log.info(`Both ready: ${data.both_ready}`);
        log.info(`Game started: ${data.game_started}`);
        log.info(`Players: ${JSON.stringify(data.players, null, 2)}`);

        return true;

    } catch (error) {
        log.error(`Game state test failed: ${error.message}`);
        return false;
    }
}

/**
 * Test 4: Ready Endpoint
 */
async function testReadyEndpoint() {
    log.header("Test 4: Ready Endpoint");

    const sessionId = 'ready-test';
    const agentId = 'Human1';

    try {
        // First mark as ready
        const response = await fetch(`http://localhost:3000/ready/${sessionId}/${agentId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            log.error(`Failed to mark ready (Status: ${response.status})`);
            return false;
        }

        const data = await response.json();
        log.success(`Marked ${agentId} as ready`);
        log.info(`Players connected: ${data.players_connected}`);
        log.info(`Both ready: ${data.both_ready}`);

        return true;

    } catch (error) {
        log.error(`Ready endpoint test failed: ${error.message}`);
        return false;
    }
}

/**
 * Test 5: Lobby Variables
 */
async function testLobbyVariables() {
    log.header("Test 5: Lobby Variables Check");

    const checks = [
        {
            name: "lv_agent_id",
            check: () => typeof lv_agent_id !== 'undefined',
            value: () => lv_agent_id
        },
        {
            name: "lv_session_id",
            check: () => typeof lv_session_id !== 'undefined',
            value: () => lv_session_id
        },
        {
            name: "lv_player_id",
            check: () => typeof lv_player_id !== 'undefined',
            value: () => lv_player_id
        },
        {
            name: "initializeLobby function",
            check: () => typeof initializeLobby === 'function',
            value: () => "exists"
        },
        {
            name: "markPlayerReady function",
            check: () => typeof markPlayerReady === 'function',
            value: () => "exists"
        },
        {
            name: "getAgentId function",
            check: () => typeof getAgentId === 'function',
            value: () => "exists"
        },
    ];

    let allPass = true;
    for (const check of checks) {
        const passed = check.check();
        if (passed) {
            log.success(`${check.name}: ${check.value()}`);
        } else {
            log.error(`${check.name}: NOT FOUND`);
            allPass = false;
        }
    }

    return allPass;
}

/**
 * Test 6: Script Load Order
 */
async function testScriptLoadOrder() {
    log.header("Test 6: Script Load Order Check");

    const scripts = [
        { name: "jQuery", check: () => typeof $ !== 'undefined' },
        { name: "util.js", check: () => typeof get_max_grid_dimensions === 'function' },
        { name: "gen_grid.js", check: () => typeof initialize_grid === 'function' },
        { name: "lobby.js", check: () => typeof initializeLobby === 'function' },
        { name: "loop.js", check: () => typeof world_manager_loop === 'function' },
        { name: "human_agent.js", check: () => typeof send_userinput_to_MATRX === 'function' },
    ];

    let allLoaded = true;
    for (const script of scripts) {
        const loaded = script.check();
        if (loaded) {
            log.success(`${script.name}: Loaded`);
        } else {
            log.error(`${script.name}: Not loaded`);
            allLoaded = false;
        }
    }

    return allLoaded;
}

/**
 * Test 7: URL Parsing
 */
async function testUrlParsing() {
    log.header("Test 7: URL Parsing");

    const path = window.location.pathname;
    const pathParts = path.split('/');
    const sessionId = pathParts[pathParts.length - 1];

    log.info(`Current URL: ${window.location.href}`);
    log.info(`Pathname: ${path}`);
    log.info(`Path parts: ${JSON.stringify(pathParts)}`);
    log.info(`Extracted session ID: ${sessionId}`);

    if (sessionId && sessionId !== '' && sessionId !== 'human-agent') {
        log.success(`Session ID extracted correctly`);
        return true;
    } else {
        log.error(`Session ID extraction failed`);
        return false;
    }
}

/**
 * Test 8: DOM Elements
 */
async function testDomElements() {
    log.header("Test 8: DOM Elements Check");

    const elements = [
        { id: 'lobby-container', name: 'Lobby Container' },
        { id: 'game-container', name: 'Game Container' },
        { id: 'player-name', name: 'Player Name Display' },
        { id: 'agent-id-display', name: 'Agent ID Display' },
        { id: 'lobby-status', name: 'Lobby Status' },
        { id: 'ready-button', name: 'Ready Button' },
        { id: 'lobby-error', name: 'Lobby Error Display' },
        { id: 'grid', name: 'Game Grid' },
    ];

    let allPresent = true;
    for (const elem of elements) {
        const el = document.getElementById(elem.id);
        if (el) {
            log.success(`${elem.name} (#${elem.id}): Present`);
        } else {
            log.error(`${elem.name} (#${elem.id}): Missing`);
            allPresent = false;
        }
    }

    return allPresent;
}

/**
 * Run all diagnostic tests
 */
async function runAllDiagnostics() {
    log.header("STARTING FULL DIAGNOSTIC TEST");

    const results = {};

    results['Server Connectivity'] = await testServerConnectivity();
    results['DOM Elements'] = await testDomElements();
    results['Script Load Order'] = await testScriptLoadOrder();
    results['URL Parsing'] = await testUrlParsing();
    results['Lobby Variables'] = await testLobbyVariables();
    results['Join Session'] = await testJoinSession();
    results['Game State'] = await testGameState();
    results['Ready Endpoint'] = await testReadyEndpoint();

    // Summary
    log.header("DIAGNOSTIC SUMMARY");
    let passCount = 0;
    let failCount = 0;

    for (const [test, passed] of Object.entries(results)) {
        if (passed) {
            log.success(`${test}: PASS`);
            passCount++;
        } else {
            log.error(`${test}: FAIL`);
            failCount++;
        }
    }

    console.log(`\n${'='.repeat(50)}`);
    log.info(`Total: ${passCount} passed, ${failCount} failed`);
    console.log(`${'='.repeat(50)}\n`);

    if (failCount === 0) {
        log.success("ALL TESTS PASSED! System should be ready for multi-player.");
    } else {
        log.warning(`${failCount} test(s) failed. See above for details.`);
    }

    return results;
}

/**
 * Quick test - just check if things are loaded
 */
async function quickTest() {
    log.header("QUICK DIAGNOSTIC TEST");

    const checks = [
        ["Server is up", () => fetch('/', {method: 'GET'}).then(r => r.ok)],
        ["Scripts loaded", () => typeof initializeLobby === 'function'],
        ["DOM ready", () => document.getElementById('lobby-container') !== null],
        ["Session ID extracted", () => {
            const path = window.location.pathname;
            const id = path.split('/')[path.split('/').length - 1];
            return id !== 'human-agent' && id !== '';
        }],
    ];

    for (const [name, check] of checks) {
        try {
            const result = await Promise.resolve(check());
            if (result) log.success(name);
            else log.error(name);
        } catch (e) {
            log.error(`${name}: ${e.message}`);
        }
    }
}

// Export functions
window.diagnostic = {
    testServerConnectivity,
    testJoinSession,
    testGameState,
    testReadyEndpoint,
    testLobbyVariables,
    testScriptLoadOrder,
    testUrlParsing,
    testDomElements,
    runAllDiagnostics,
    quickTest,
};

log.info("Diagnostic functions available:");
log.info("  diagnostic.quickTest()          - Quick 5-second check");
log.info("  diagnostic.runAllDiagnostics()  - Full comprehensive test");
log.info("  diagnostic.testServerConnectivity()");
log.info("  diagnostic.testJoinSession()");
log.info("  diagnostic.testGameState()");
log.info("And more... see source for full list");
