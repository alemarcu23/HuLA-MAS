import threading
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory

'''
This file holds the code for the MATRX RESTful api.
External scripts can send POST and/or GET requests to retrieve state, tick and other information, and send
userinput or other information to MATRX. The api is a Flask (Python) webserver.

For visualization, see the seperate MATRX visualization folder / package.
'''

debug = True
port = 3000
app = Flask(__name__, template_folder='templates')

# the path to the media folder of the user (outside of the MATRX package)
ext_media_folder = ""

# Multi-player session management
import threading
session_state = {}  # Tracks connected players per session
session_id_global = "default_session"  # Default session ID
multi_human_global = False  # Whether the multi-player (multiple humans) lobby is enabled
session_lock = threading.RLock()  # RLock = Reentrant Lock (allows same thread to acquire multiple times)

# ── Human Supervisor view state ───────────────────────────────────────────
# Cumulative "explored map" memory for the supervisor: obj_id -> the latest
# object dict ever observed by any agent. This persists across ticks so that
# once an agent has seen something it stays on the supervisor's map and never
# disappears (fog-of-war reveal). Reset whenever a new world is detected.
_sup_discovered = {}      # obj_id -> object property dict (last-known)
_sup_world_id = None      # world_ID the explored map currently belongs to
_sup_lock = threading.RLock()

def init_session(session_id):
    """Initialize a new session for multi-player coordination"""
    with session_lock:
        if session_id not in session_state:
            session_state[session_id] = {
                'players': {},  # {agent_id: {'player_id': 'player1', 'ready': False}}
                'game_started': False
            }

#########################################################################
# Visualization server routes
#########################################################################

@app.route('/human-agent/<id>')
def human_agent_view(id):
    """
    Route for HumanAgentBrain

    Parameters
    ----------
    id
        The human agent ID. Is obtained from the URL.

    Returns
    -------
    str
        The template for this agent's view.

    """
    return render_template('human_agent.html', id=id, multi_human=multi_human_global)


# route for agent, get the ID from the URL
@app.route('/agent/<id>')
def agent_view(id):
    """
    Route for AgentBrain

    Parameters
    ----------
    id
        The agent ID. Is obtained from the URL.

    Returns
    -------
    str
        The template for this agent's view.

    """
    return render_template('agent.html', id=id)


@app.route('/god')
def god_view():
    """
    Route for the 'god' view which contains the ground truth of the world without restrictions.

    Returns
    -------
    str
        The template for this view.

    """
    return render_template('god.html')


@app.route('/supervisor')
def supervisor_view():
    """
    Route for the Human Supervisor view. The supervisor oversees the team of
    agents, sees only the part of the map the agents have explored, and can send
    free-form text commands that agents may choose to act on.

    Returns
    -------
    str
        The template for this view.

    """
    return render_template('supervisor.html')


@app.route('/')
@app.route('/start')
def start_view():
    """
    Route for the 'start' view which shows information about the current scenario, including links to all agents.

    Returns
    -------
    str
        The template for this view.

    """
    return render_template('start.html')




@app.route('/shutdown_visualizer', methods=['GET', 'POST'])
def shutdown():
    """ Shuts down the visualizer by stopping the Flask thread

    Returns
        True
    -------
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Unable to shutdown visualizer server. Not running with the Werkzeug Server')
    func()
    print("Visualizer server shutting down...")
    return jsonify(True)


@app.route('/fetch_external_media/<path:filename>')
def external_media(filename):
    """ Facilitate the use of images in the visualization outside of the static folder

    Parameters
    ----------
    filename
        path to the image file in the external media folder of the user.

    Returns
    -------
        Returns the url (relative from the website root) to that file
    """
    return send_from_directory(ext_media_folder, filename, as_attachment=True)


#########################################################################
# Multi-player session management routes
#########################################################################

@app.route('/join-session/<session_id>', methods=['POST'])
def join_session(session_id):
    """
    Join a multi-player session. Assigns the player to an available agent.

    Parameters
    ----------
    session_id
        The session ID to join

    Returns
    -------
    dict
        Contains player_id and assigned agent_id (Human1 or Human2)
    """
    print(f"\n[{session_id}] === JOIN SESSION REQUEST RECEIVED ===", flush=True)
    print(f"[{session_id}] Acquiring lock...", flush=True)

    try:
        # Use a timeout on the lock to prevent infinite waiting
        acquired = session_lock.acquire(timeout=2.0)

        if not acquired:
            print(f"[{session_id}] ERROR: Could not acquire lock within 2 seconds!", flush=True)
            return jsonify({'error': 'Server busy, try again'}), 503

        print(f"[{session_id}] Lock acquired successfully", flush=True)

        try:
            init_session(session_id)
            print(f"[{session_id}] Session initialized", flush=True)

            session = session_state[session_id]
            print(f"[{session_id}] Current players before assignment: {list(session['players'].keys())}", flush=True)

            # Determine which agent to assign (Human1 or Human2)
            if 'Human1' not in session['players']:
                player_id = 'player1'
                agent_id = 'Human1'
                print(f"[{session_id}] Assigning: Player 1 as Human1", flush=True)
            elif 'Human2' not in session['players']:
                player_id = 'player2'
                agent_id = 'Human2'
                print(f"[{session_id}] Assigning: Player 2 as Human2", flush=True)
            else:
                print(f"[{session_id}] ERROR: Session full, rejecting join attempt", flush=True)
                return jsonify({'error': 'Session is full'}), 400

            # Register this player
            session['players'][agent_id] = {
                'player_id': player_id,
                'ready': False
            }

            print(f"[{session_id}] Current players after assignment: {list(session['players'].keys())}", flush=True)
            print(f"[{session_id}] Returning response: agent_id={agent_id}, player_id={player_id}", flush=True)

            response = {
                'player_id': player_id,
                'agent_id': agent_id,
                'session_id': session_id
            }

            print(f"[{session_id}] === JOIN SESSION COMPLETE ===\n", flush=True)
            return jsonify(response)

        finally:
            session_lock.release()
            print(f"[{session_id}] Lock released", flush=True)

    except Exception as e:
        print(f"[{session_id}] EXCEPTION in join_session: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/ready/<session_id>/<agent_id>', methods=['POST'])
def mark_ready(session_id, agent_id):
    """
    Mark a player as ready to start the game.

    Parameters
    ----------
    session_id
        The session ID
    agent_id
        The agent ID (Human1 or Human2)

    Returns
    -------
    dict
        Contains both_ready status
    """
    print(f"[{session_id}] Mark ready request: {agent_id}", flush=True)

    try:
        acquired = session_lock.acquire(timeout=2.0)
        if not acquired:
            print(f"[{session_id}] ERROR: Could not acquire lock for /ready", flush=True)
            return jsonify({'error': 'Server busy'}), 503

        try:
            init_session(session_id)
            session = session_state[session_id]

            if agent_id in session['players']:
                session['players'][agent_id]['ready'] = True
                print(f"[{session_id}] {agent_id} marked as ready", flush=True)
            else:
                print(f"[{session_id}] Warning: {agent_id} not found in session", flush=True)

            # Check if both players are ready
            both_ready = all(p['ready'] for p in session['players'].values()) if len(session['players']) == 2 else False

            print(f"[{session_id}] Status - Players: {len(session['players'])}, Both Ready: {both_ready}", flush=True)

            return jsonify({
                'both_ready': both_ready,
                'players_connected': len(session['players'])
            })
        finally:
            session_lock.release()

    except Exception as e:
        print(f"[{session_id}] EXCEPTION in mark_ready: {str(e)}", flush=True)
        return jsonify({'error': 'Server error'}), 500


@app.route('/game-state/<session_id>', methods=['GET'])
def get_game_state(session_id):
    """
    Get the current game state for a session.

    Parameters
    ----------
    session_id
        The session ID

    Returns
    -------
    dict
        Contains game status and player information
    """
    try:
        acquired = session_lock.acquire(timeout=2.0)
        if not acquired:
            print(f"[{session_id}] ERROR: Could not acquire lock for /game-state", flush=True)
            return jsonify({'error': 'Server busy'}), 503

        try:
            init_session(session_id)
            session = session_state[session_id]

            both_ready = all(p['ready'] for p in session['players'].values()) if len(session['players']) == 2 else False

            return jsonify({
                'players_connected': len(session['players']),
                'both_ready': both_ready,
                'game_started': session['game_started'],
                'players': session['players']
            })
        finally:
            session_lock.release()

    except Exception as e:
        print(f"[{session_id}] EXCEPTION in get_game_state: {str(e)}", flush=True)
        return jsonify({'error': 'Server error'}), 500


#########################################################################
# Human Supervisor view routes
#########################################################################

def _supervisor_agent_list(matrx_api, explored):
    """ Build the sidebar agent roster from the explored state.

    Includes every autonomous (non-human) agent that has been revealed on the
    map. Roles are read from the live agent brains when available — humans are
    excluded because supervisor commands are only consumed by the AI agents.
    """
    agents = []
    gw = getattr(matrx_api, '_gw', None)
    registered = getattr(gw, 'registered_agents', {}) if gw is not None else {}

    for obj_id, obj in explored.items():
        if not (isinstance(obj, dict) and obj.get('isAgent')):
            continue
        if obj.get('is_human_agent'):
            continue

        role = ''
        brain = registered.get(obj_id)
        profile = getattr(brain, 'profile', None) if brain is not None else None
        if profile is not None:
            try:
                role = profile.role_str() or ''
            except Exception:
                role = ''

        agents.append({
            'id': obj_id,
            'role': role,
            'location': obj.get('location'),
        })

    agents.sort(key=lambda a: a['id'])
    return agents


@app.route('/supervisor_filtered_state', methods=['POST'])
def supervisor_filtered_state():
    """ State feed for the Human Supervisor view.

    Returns a god-shaped state filtered down to only what the agent team has
    explored so far (cumulative — discovered objects never disappear), plus the
    sidebar payload (agent roster, inter-agent messages, supervisor commands,
    and private agent->human status entries). The frontend renders the state
    exactly like the god view.
    """
    from matrx.api import api as matrx_api

    states = matrx_api.get_latest_state_dicts()
    god_state = states.get('god')
    if not god_state:
        # MATRX has not produced a tick yet — frontend simply keeps polling.
        return jsonify({
            'states': [],
            'supervisor': {'agents': []},
            'matrx_paused': bool(getattr(matrx_api, 'matrx_paused', False)),
        })

    world = god_state.get('World', {}) or {}
    world_id = world.get('world_ID')

    global _sup_world_id
    with _sup_lock:
        # New world / restart → forget the previous explored map.
        if world_id != _sup_world_id:
            _sup_discovered.clear()
            _sup_world_id = world_id

        # 1) Reveal everything every agent currently senses (accumulates).
        for aid, st in states.items():
            if aid == 'god' or not isinstance(st, dict):
                continue
            for obj_id, obj in st.items():
                if obj_id == 'World' or not isinstance(obj, dict):
                    continue
                _sup_discovered[obj_id] = obj

        # 2) Always keep the team's live positions visible using ground truth,
        #    so the supervisor can track agents even outside any sense range.
        for obj_id, obj in god_state.items():
            if obj_id == 'World' or not isinstance(obj, dict):
                continue
            if obj.get('isAgent'):
                _sup_discovered[obj_id] = obj

        explored = dict(_sup_discovered)

    filtered_state = {'World': world}
    filtered_state.update(explored)

    # ── Sidebar payload ──────────────────────────────────────────────────
    # Inter-agent messages, with supervisor commands split into their own list
    # (they live in the same global log but are rendered separately).
    messages, commands = [], []
    try:
        from matrx.agents.llm.modules.agent_infra import communication_module as _cm
        for m in list(_cm.GLOBAL_MESSAGE_LOG):
            if m.get('message_type') == 'supervisor_command':
                commands.append({
                    'target_agent': m.get('to', 'all'),
                    'text': m.get('text', ''),
                    'ts': m.get('ts', 0),
                })
            else:
                messages.append(dict(m))
    except Exception:
        pass

    # Private agent->human status entries (plans/actions) — never seen by peers.
    agent_status = []
    try:
        from matrx.agents.llm.modules import supervisor_channel as _sup_ch
        agent_status = [dict(s) for s in list(_sup_ch.AGENT_STATUS_LOG)]
    except Exception:
        pass

    return jsonify({
        'states': [{'god': {'state': filtered_state}}],
        'matrx_paused': bool(getattr(matrx_api, 'matrx_paused', False)),
        'supervisor': {
            'agents': _supervisor_agent_list(matrx_api, explored),
            'messages': messages,
            'supervisor_commands': commands,
            'agent_status': agent_status,
        },
    })


@app.route('/supervisor_command', methods=['POST'])
def supervisor_command():
    """ Queue a free-form supervisor command for one agent or all agents.

    The command is delivered to agents through the supervisor channel; it is
    treated as guidance, not a forced action — each agent's reasoning decides
    whether and how to act on it.

    Expected JSON body: ``{"text": "...", "target_agent": "<id>"|"all"}``.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    target = (data.get('target_agent') or 'all').strip() or 'all'

    if not text:
        return jsonify({'ok': False, 'error': 'Empty command.'}), 400

    try:
        from matrx.agents.llm.modules import supervisor_channel as _sup_ch
        from matrx.api import api as matrx_api
        entry = _sup_ch.push_command(
            text=text,
            tick=getattr(matrx_api, '_current_tick', None),
            target_agent=target,
        )
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

    if entry is None:
        return jsonify({'ok': False, 'error': 'Command rejected.'}), 400

    return jsonify({'ok': True, 'command': {'target_agent': target, 'text': text}})


@app.route('/supervisor_pause', methods=['POST'])
def supervisor_pause():
    """ Toggle the MATRX simulation pause from the supervisor view. """
    try:
        from matrx.api import api as matrx_api
        matrx_api.matrx_paused = not bool(getattr(matrx_api, 'matrx_paused', False))
        return jsonify({'ok': True, 'paused': matrx_api.matrx_paused})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


#########################################################################
# Visualization Flask methods
#########################################################################

def _flask_thread():
    """
    Starts the Flask server on localhost:3000
    """

    if not debug:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    print(f"\n{'='*60}", flush=True)
    print(f"Flask app routes registered:", flush=True)
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint} {list(rule.methods)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    print(f"Starting Flask server on http://0.0.0.0:{port}", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

def run_matrx_visualizer(verbose, media_folder, session_id="default_session", vis_port=None, multi_human=False):
    """
    Creates a seperate Python thread in which the visualization server (Flask) is started, serving the JS visualization

    Parameters
    ----------
    verbose
        Whether to enable debug logging
    media_folder
        Path to external media folder
    session_id
        Session ID for multi-player coordination
    vis_port
        Port for the Flask visualization server (overrides the global default of 3000)
    multi_human
        Whether to enable the multi-player (multiple humans) lobby UI. When False
        (default) a single human plays alongside the agents and no multi-player
        session/lobby UI is served.

    Returns
    -------
        MATRX visualization Python thread
    """
    global debug, ext_media_folder, session_id_global, port, multi_human_global
    debug = verbose
    ext_media_folder = media_folder
    session_id_global = session_id
    multi_human_global = multi_human
    if vis_port is not None:
        port = vis_port

    # Only set up multi-player session coordination when enabled
    if multi_human:
        init_session(session_id)

    print("Starting visualization server")
    print(f"Initialized app: {app}")
    print(f"Session ID: {session_id}")
    vis_thread = threading.Thread(target=_flask_thread)
    vis_thread.start()
    return vis_thread

if __name__ == "__main__":
    run_matrx_visualizer()