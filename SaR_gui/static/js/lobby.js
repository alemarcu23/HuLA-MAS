// Lobby system for multi-player coordination
// NOTE: lv_agent_id and lv_session_id are set here but NOT declared
// They are globals that will be used by loop.js and other scripts

var game_started = false;  // Track if game has started (not used elsewhere)

/**
 * Initialize lobby: Join the session and get assigned to an agent
 */
async function initializeLobby() {
    console.log("=== LOBBY INITIALIZATION STARTED ===");
    console.log("typeof lv_agent_id:", typeof lv_agent_id);
    console.log("typeof lv_session_id:", typeof lv_session_id);

    // Get session ID from URL path (e.g., "/human-agent/session1" -> "session1")
    const path = window.location.pathname;
    const pathParts = path.split('/');
    lv_session_id = pathParts[pathParts.length - 1] || 'default_session';

    console.log("URL path:", path);
    console.log("Session ID extracted:", lv_session_id);

    // First, verify server is responding
    console.log("=== Step 1: Verifying server connectivity ===");
    const serverOk = await verifyServerConnectivity();
    if (!serverOk) {
        showLobbyError('Cannot connect to game server. Is it running on port 3000?');
        return false;
    }
    console.log("Server connectivity verified!");

    try {
        // Join the session
        console.log("=== Step 2: Joining session ===");
        const joinUrl = `/join-session/${lv_session_id}`;
        console.log("Attempting to fetch:", joinUrl);

        // Create abort controller for timeout (5 second timeout)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.error("Fetch timeout: Server not responding within 5 seconds");
            controller.abort();
        }, 5000);

        const joinResponse = await fetch(joinUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            signal: controller.signal
        });

        clearTimeout(timeoutId);  // Clear timeout if request succeeded

        console.log("Join response status:", joinResponse.status);
        console.log("Join response statusText:", joinResponse.statusText);

        if (!joinResponse.ok) {
            let errorMsg = 'Failed to join session (Status: ' + joinResponse.status + ')';
            try {
                const errorData = await joinResponse.json();
                console.error("Error response data:", errorData);
                if (errorData.error) {
                    errorMsg = errorData.error;
                }
            } catch (e) {
                console.error("Could not parse error response:", e);
            }
            showLobbyError(errorMsg);
            return false;
        }

        console.log("=== Step 3: Parsing response ===");
        const joinData = await joinResponse.json();
        console.log("Join response data:", joinData);

        // Ensure values are assigned to global variables
        lv_agent_id = joinData.agent_id || null;
        lv_player_id = joinData.player_id || null;

        console.log("After assignment - lv_agent_id:", lv_agent_id);
        console.log("After assignment - lv_player_id:", lv_player_id);

        if (!lv_agent_id) {
            console.error("ERROR: agent_id not found in response!");
            showLobbyError('Server error: No agent assigned. Please reload.');
            return false;
        }

        // Update UI with player info
        console.log("=== Step 4: Updating UI ===");
        updateLobbyUI();
        startPollingGameState();
        console.log("=== LOBBY INITIALIZATION COMPLETE ===");
        return true;

    } catch (error) {
        console.error('Lobby initialization exception:', error);
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);

        let userMsg = 'Connection error: ' + error.message;
        if (error.name === 'AbortError') {
            userMsg = 'Server timeout: Game server is not responding. Try reloading the page.';
        } else if (error.message.includes('Failed to fetch')) {
            userMsg = 'Cannot connect to game server. Check that Flask server is running on port 3000.';
        }

        showLobbyError(userMsg);
        return false;
    }
}

/**
 * Verify that the game server is running and responding
 * @returns {Promise<boolean>} true if server is running, false otherwise
 */
async function verifyServerConnectivity() {
    try {
        console.log("Checking server connectivity...");

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
        }, 3000);  // 3 second timeout for connectivity check

        const response = await fetch('/', {
            method: 'GET',
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        console.log("Server check - Status:", response.status);
        return response.ok;

    } catch (error) {
        console.error("Server connectivity check failed:", error.message);
        return false;
    }
}

/**
 * Update the lobby UI with player information
 */
function updateLobbyUI() {
    const playerName = lv_agent_id === 'Human1' ? 'Player 1' : 'Player 2';
    document.getElementById('player-name').textContent = playerName;
    document.getElementById('agent-id-display').textContent = lv_agent_id;
    document.getElementById('lobby-status').textContent = 'Waiting for Player 2 to connect...';
}

/**
 * Mark the current player as ready
 */
async function markPlayerReady() {
    console.log("markPlayerReady() called");
    console.log("Current lv_agent_id value:", lv_agent_id);
    console.log("Current lv_session_id value:", lv_session_id);

    if (!lv_agent_id) {
        console.error("ERROR: lv_agent_id is empty!");
        const msg = 'Lobby not initialized yet. Please wait a moment or reload the page.';
        showLobbyError(msg);
        console.error("Showing error to user:", msg);
        return;
    }

    console.log("Marking player ready:", lv_agent_id);

    try {
        const response = await fetch(`/ready/${lv_session_id}/${lv_agent_id}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            showLobbyError('Failed to mark as ready. Status: ' + response.status);
            return;
        }

        const data = await response.json();
        console.log("Ready response:", data);

        if (data.both_ready) {
            console.log("Both players ready! Starting game...");
            startGame();
        } else {
            const status = data.players_connected === 1
                ? "Waiting for Player 2 to connect..."
                : `${data.players_connected}/2 players ready. Waiting for other player...`;
            document.getElementById('lobby-status').textContent = status;
            document.getElementById('ready-button').disabled = true;
            document.getElementById('ready-button').textContent = 'Waiting for other player...';
        }

    } catch (error) {
        console.error('Error marking player ready:', error);
        showLobbyError('Connection error. Please try again.');
    }
}

/**
 * Poll the game state to update player count and ready status
 */
function startPollingGameState() {
    console.log("Starting game state polling for session:", lv_session_id);

    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/game-state/${lv_session_id}`);

            if (!response.ok) {
                console.error("Game state fetch failed:", response.status);
                return;
            }

            const data = await response.json();
            const playersConnected = data.players_connected;
            const bothReady = data.both_ready;

            console.log("Game state poll:", { playersConnected, bothReady, gameStarted: game_started });

            if (playersConnected === 1) {
                document.getElementById('lobby-status').textContent =
                    '1/2 players connected. Waiting for Player 2...';
            } else if (playersConnected === 2 && !bothReady) {
                document.getElementById('lobby-status').textContent =
                    '2/2 players connected. Click Ready when you\'re prepared.';
                // Re-enable ready button when other player connects
                const readyBtn = document.getElementById('ready-button');
                if (readyBtn && readyBtn.disabled) {
                    readyBtn.disabled = false;
                    readyBtn.textContent = 'I\'m Ready to Start';
                }
            } else if (bothReady && !game_started) {
                console.log("Both players ready! Initiating game start...");
                document.getElementById('lobby-status').textContent =
                    'Both players ready! Starting game...';
                game_started = true;
                clearInterval(pollInterval);
                hideLobbySoon();
            }

        } catch (error) {
            console.error('Error polling game state:', error);
        }
    }, 500);
}

/**
 * Start the game by hiding the lobby
 */
function startGame() {
    game_started = true;
    document.getElementById('lobby-container').classList.add('d-none');
    document.getElementById('game-container').classList.remove('d-none');

    // Unpause the world (trigger world start)
    unpauseWorld();
}

/**
 * Hide the lobby after a short delay to allow server synchronization
 */
function hideLobbySoon() {
    setTimeout(() => {
        startGame();
    }, 1000);
}

/**
 * Display error message in lobby
 */
function showLobbyError(message) {
    document.getElementById('lobby-error').textContent = message;
    document.getElementById('lobby-error').style.display = 'block';
}

/**
 * Check if game has started
 */
function isGameStarted() {
    return game_started;
}

/**
 * Get the current session ID
 */
function getSessionId() {
    return lv_session_id;
}

/**
 * Get the current agent ID (Human1 or Human2)
 */
function getAgentId() {
    return lv_agent_id;
}

/**
 * Unpause the world by triggering a play button click in the God view
 * or directly calling the start in MATRX (if running as server)
 */
function unpauseWorld() {
    // The world is started by the first player clicking play in the God view
    // OR we can trigger it via API call to MATRX when both players are ready
    console.log("Both players ready - game can now be started from God view");

    // Send the unpause/start command to MATRX API
    console.log("Sending start command to MATRX API...");
    var resp = $.ajax({
        method: "GET",
        url: 'http://' + window.location.hostname + ':3001/start',
        contentType: "application/json; charset=utf-8",
        dataType: 'json',
        success: function(data) {
            console.log("MATRX world started successfully:", data);
            // Wait a moment for MATRX to fully transition to running before game loop queries it
            console.log("Waiting for MATRX to fully start (500ms delay)...");
        },
        error: function(error) {
            console.error("ERROR: Failed to start MATRX world:", error);
        }
    });
}
