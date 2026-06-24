/**
 * supervisor_game.js
 *
 * Drives the Human Supervisor view.  Uses gen_grid.js for tile-based rendering
 * (same visual style as god/agent views) but:
 *   - fetches from /supervisor_filtered_state instead of port 3001 directly
 *   - the returned state only contains objects discovered by robots
 *   - adds a sidebar with agent status, per-agent/broadcast command panel,
 *     and a combined agent-messages + supervisor-command log
 *
 * This file MUST be loaded after gen_grid.js and util.js.
 */

(function () {
    'use strict';

    /* ── Variables expected by gen_grid.js ──────────────────────────────── */
    window.lv_agent_id      = 'god';   // so gen_grid renders the god-view state
    window.lv_matrx_version = '2.0.0';
    window.chat_offsets     = {};
    window.object_selected  = false;

    /* ── Stub functions that gen_grid.js / draw_bg_tiles call ────────────── */
    window.add_context_menu       = function () {};
    window.add_selection_listener = function () {};
    window.populate_agent_menu    = function () {};
    window.populate_new_chat_dropdown = function () {};
    window.process_messages       = function () {};
    window.sync_play_button       = function () {};
    window.startDrawErase         = function () {};
    window.stopDrag               = function () {};
    window.drawToggle             = function () {};
    window.eraseToggle            = function () {};
    window.chatToggle             = function () {};
    window.reset_chat             = function () {};

    /* ── Override get_max_grid_dimensions to leave room for sidebar ──────── */
    window.get_max_grid_dimensions = function () {
        var sidebar = document.getElementById('sup-sidebar');
        var sidebarW = sidebar ? (sidebar.offsetWidth + 1) : 0;
        var height = document.documentElement.clientHeight
                     - (window.navbar ? window.navbar.scrollHeight : 49);
        var width  = document.documentElement.clientWidth - sidebarW;
        return [height, width];
    };

    /* ── Loop state ─────────────────────────────────────────────────────── */
    var lv_state            = {};
    var lv_world_settings   = null;
    var lv_current_tick     = 0;
    var lv_open_req         = false;
    var lv_first_tick       = true;
    var lv_last_update      = Date.now();
    var lv_wait_ms          = 0;
    var lv_tick_duration    = 0.5;
    var lv_reinitialize     = false;
    var lv_world_ID         = null;
    var lv_new_world_ID     = null;
    var lv_matrx_paused     = false;

    /* ── Agent colour palette ───────────────────────────────────────────── */
    var PALETTE = [
        '#ff6b6b', '#4ecdc4', '#f9ca24', '#a29bfe',
        '#fd79a8', '#55efc4', '#fdcb6e', '#74b9ff',
    ];
    var agentColors  = {};
    var paletteIdx   = 0;

    function colorFor(id) {
        if (!agentColors[id]) {
            agentColors[id] = PALETTE[paletteIdx % PALETTE.length];
            paletteIdx++;
        }
        return agentColors[id];
    }

    /* Agent IDs are the lowercased agent names: the first rescue bot is
       'rescuebot' (no suffix), the rest are 'rescuebot1', 'rescuebot2', … */
    var AGENT_DISPLAY_NAMES = {
        'rescuebot':  'Medic',
        'rescuebot1': 'Scout',
    };
    function displayName(id) {
        return AGENT_DISPLAY_NAMES[String(id).toLowerCase()] || id;
    }

    /* ── Message log state ──────────────────────────────────────────────── */
    var renderedEntries = 0;

    /* ═══════════════════════════════════════════════════════════════════════
     *  Initialisation
     * ═══════════════════════════════════════════════════════════════════════ */

    $(document).ready(function () {
        /* Adjust #sup-body height dynamically based on the actual toolbar +
           optional paused-banner heights. */
        adjustBodyHeight();
        window.addEventListener('resize', function () {
            adjustBodyHeight();
            if (window.fix_grid_size) { window.fix_grid_size(); }
        });

        /* Wire up gen_grid.js — needs #matrx-toolbar in the DOM. */
        if (window.initialize_grid) { window.initialize_grid(); }

        /* Broadcast row buttons */
        document.getElementById('broadcast-send').addEventListener('click', function () {
            sendCommandToAgent('all',
                document.getElementById('broadcast-input'),
                document.getElementById('broadcast-status'));
        });
        document.getElementById('broadcast-input').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendCommandToAgent('all',
                    document.getElementById('broadcast-input'),
                    document.getElementById('broadcast-status'));
            }
        });

        document.getElementById('sup-pause-btn').addEventListener('click', togglePause);

        worldManagerLoop();
    });

    function adjustBodyHeight() {
        var toolbar = document.getElementById('matrx-toolbar');
        var banner  = document.getElementById('paused-banner');
        var body    = document.getElementById('sup-body');
        if (!toolbar || !body) { return; }
        var used = toolbar.offsetHeight +
                   (banner && banner.classList.contains('visible') ? banner.offsetHeight : 0);
        body.style.height = (window.innerHeight - used) + 'px';
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  World manager loop (keeps running across world restarts)
     * ═══════════════════════════════════════════════════════════════════════ */

    function worldManagerLoop() {
        init();
        setInterval(function () {
            if (lv_reinitialize) { init(); }
        }, 500);
    }

    function init() {
        lv_reinitialize  = false;
        lv_open_req      = false;
        lv_first_tick    = true;

        var hostname = window.location.hostname;
        $.getJSON('http://' + hostname + ':3001/get_info')
            .done(function (data) {
                if (!data || Object.keys(data).length === 0) {
                    lv_reinitialize = true;
                    return;
                }
                lv_tick_duration = data.tick_duration  || 0.5;
                lv_world_ID      = data.world_ID       || null;
                lv_new_world_ID  = null;
                window.lv_matrx_version = data.matrx_version || '2.0.0';
                applyPausedState(!!data.matrx_paused);
                worldLoop();
            })
            .fail(function () {
                document.getElementById('sup-tick').textContent = 'connecting…';
                lv_reinitialize = true;
            });
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Render loop
     * ═══════════════════════════════════════════════════════════════════════ */

    function worldLoop() {
        /* World switched → reinitialise */
        if (lv_new_world_ID && lv_world_ID && lv_new_world_ID !== lv_world_ID) {
            lv_reinitialize = true;
            return;
        }

        var shouldFetch = Date.now() > lv_last_update + lv_wait_ms && !lv_open_req;

        if (shouldFetch) {
            lv_last_update = Date.now();
            lv_open_req    = true;

            $.ajax({
                method: 'POST',
                url: '/supervisor_filtered_state',
                contentType: 'application/json; charset=utf-8',
                dataType: 'json',
                data: JSON.stringify({ chat_offsets: window.chat_offsets }),
                success: function (data) {
                    lv_open_req = false;
                    handleUpdate(data);
                    window.requestAnimationFrame(worldLoop);
                },
                error: function () {
                    lv_open_req     = false;
                    lv_reinitialize = true;
                },
            });
        } else {
            window.requestAnimationFrame(worldLoop);
        }
    }

    function handleUpdate(data) {
        var states = data.states || [];
        if (!states.length) { return; }

        var latest = states[states.length - 1];
        if (!latest['god']) { return; }

        var raw = latest['god'];
        var state = (raw.state !== undefined) ? raw.state : raw;

        lv_state          = state;
        lv_world_settings = state['World'];

        if (!lv_world_settings) { return; }

        var newTick   = lv_world_settings['nr_ticks'];
        lv_new_world_ID  = lv_world_settings['world_ID'];
        lv_tick_duration = lv_world_settings['tick_duration'] || lv_tick_duration;
        lv_wait_ms       = Math.min(lv_tick_duration * 1000 * 0.6, 500);
        lv_current_tick  = newTick;

        if (typeof data.matrx_paused === 'boolean') {
            applyPausedState(data.matrx_paused);
        }

        document.getElementById('sup-tick').textContent =
            'tick ' + newTick + '  │  supervisor view';

        /* Render the tile grid */
        if (window.draw) {
            window.draw(lv_state, lv_world_settings, {}, {}, true);
            /* Inject colour rings+labels AFTER draw so they survive re-renders */
            injectAgentColorBadges(lv_state);
        }

        /* Update sidebar */
        var sup = data.supervisor || {};
        updateAgentStrip(sup.agents || []);
        updateCommandBoxes(sup.agents || []);
        renderLog(
            sup.messages         || [],
            sup.supervisor_commands || [],
            sup.agent_status     || []
        );
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Agent colour badges (injected on top of gen_grid.js tiles)
     *
     *  Because all robots share the same SVG, we overlay a coloured ring and
     *  a small label chip on each agent's tile element every tick.  draw() is
     *  synchronous, so injecting immediately after guarantees the badges are
     *  present before the browser paints — no visible flicker.
     * ═══════════════════════════════════════════════════════════════════════ */

    function injectAgentColorBadges(state) {
        if (!state) { return; }

        for (var objId in state) {
            if (objId === 'World') { continue; }
            var obj = state[objId];
            if (!obj || typeof obj !== 'object' || !obj.isAgent) { continue; }

            var el = document.getElementById(objId);
            if (!el) { continue; }

            var color = colorFor(objId);

            /* Remove stale badges from this element (covers re-render case) */
            var old = el.querySelectorAll('.sup-agent-badge');
            for (var i = 0; i < old.length; i++) { old[i].remove(); }

            /* ── Coloured ring around the tile ── */
            var ring = document.createElement('div');
            ring.className = 'sup-agent-badge sup-agent-ring';
            ring.style.cssText =
                'position:absolute;' +
                'top:0;left:0;right:0;bottom:0;' +
                'border-radius:50%;' +
                'border:3px solid ' + color + ';' +
                'box-sizing:border-box;' +
                'pointer-events:none;' +
                'z-index:20;';
            el.appendChild(ring);

            /* ── Short label chip  (e.g. "R0", "R1") ── */
            var label = shortAgentLabel(objId);
            var chip = document.createElement('div');
            chip.className = 'sup-agent-badge sup-agent-label';
            chip.textContent = label;
            chip.style.cssText =
                'position:absolute;' +
                'bottom:-1px;right:-1px;' +
                'background:' + color + ';' +
                'color:#111;' +
                'font-family:sans-serif;' +
                'font-size:9px;' +
                'font-weight:800;' +
                'line-height:1;' +
                'padding:1px 3px;' +
                'border-radius:3px;' +
                'pointer-events:none;' +
                'z-index:21;' +
                'white-space:nowrap;' +
                'box-shadow:0 0 3px rgba(0,0,0,0.7);';
            el.appendChild(chip);
        }
    }

    function shortAgentLabel(id) {
        /* "RescueBot0" → "R0",  "rescuebot_1" → "R1",  else first 2 chars */
        var m = String(id).match(/^([A-Za-z]+)[_\-]?(\d+)$/);
        if (m) { return m[1][0].toUpperCase() + m[2]; }
        return String(id).slice(0, 2).toUpperCase();
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Sidebar — agent strip
     * ═══════════════════════════════════════════════════════════════════════ */

    function updateAgentStrip(agents) {
        var strip = document.getElementById('agents-strip');
        if (!agents.length) {
            strip.innerHTML = '<span style="color:#555;font-size:11px;">no agents yet</span>';
            return;
        }
        /* Rebuild only when roster changes; otherwise just patch role + pos */
        var currentIds = agents.map(function (a) { return a.id; }).join('|');
        if (strip.dataset.ids === currentIds) {
            agents.forEach(function (a) {
                var roleEl = strip.querySelector('[data-strip-role="' + a.id + '"]');
                if (roleEl) {
                    var r = niceRole(a.role);
                    roleEl.textContent  = r || 'unassigned';
                    roleEl.className    = 'chip-role' + (r ? '' : ' unassigned');
                }
                var posEl = strip.querySelector('[data-strip-pos="' + a.id + '"]');
                if (posEl && a.location) {
                    posEl.textContent = '(' + a.location[0] + ',' + a.location[1] + ')';
                }
            });
            return;
        }
        strip.dataset.ids = currentIds;
        strip.innerHTML = '';
        agents.forEach(function (a) {
            var color   = colorFor(a.id);
            var loc     = a.location ? '(' + a.location[0] + ',' + a.location[1] + ')' : '?';
            var roleStr = niceRole(a.role);
            var chip    = document.createElement('div');
            chip.className = 'agent-chip';
            chip.style.borderLeftColor = color;
            chip.innerHTML =
                '<span class="dot" style="background:' + color + '"></span>' +
                '<span class="chip-name">' + esc(displayName(a.id)) + '</span>' +
                '<span class="chip-role' + (roleStr ? '' : ' unassigned') + '"' +
                      ' data-strip-role="' + esc(a.id) + '">' +
                    esc(roleStr || 'unassigned') + '</span>' +
                '<span class="pos" data-strip-pos="' + esc(a.id) + '">' + loc + '</span>';
            strip.appendChild(chip);
        });
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Sidebar — per-agent command boxes
     *
     *  One input+send row per live agent, labeled with name and role.
     *  Only rebuilt when the set of agent IDs changes so in-progress
     *  typing is not erased on every tick.
     * ═══════════════════════════════════════════════════════════════════════ */

    var _lastAgentIds = '';   // serialised agent-id list from last render

    /* Human-readable role label */
    var ROLE_LABELS = {
        'scout':        'Scout',
        'medic':        'Medic',
        'heavy_lifter': 'Heavy Lifter',
        'support':      'Support',
        'generalist':   'Generalist',
    };
    function niceRole(roleStr) {
        if (!roleStr) { return ''; }
        return roleStr.split(',').map(function (r) {
            var key = r.trim().toLowerCase();
            return ROLE_LABELS[key] || (r.trim().charAt(0).toUpperCase() + r.trim().slice(1));
        }).join(' · ');
    }

    function updateCommandBoxes(agents) {
        var container   = document.getElementById('cmd-boxes');
        var currentIds  = agents.map(function (a) { return a.id; }).join('|');

        /* Only rebuild DOM when the agent roster changes */
        if (currentIds === _lastAgentIds && currentIds !== '') {
            /* Roster unchanged — just refresh roles (may have arrived late) */
            agents.forEach(function (a) {
                var roleEl = container.querySelector('[data-agent-role="' + a.id + '"]');
                if (roleEl && a.role) { roleEl.textContent = niceRole(a.role); }
            });
            return;
        }
        _lastAgentIds = currentIds;

        container.innerHTML = '';

        if (!agents.length) {
            container.innerHTML = '<span style="color:#555;font-size:11px;">no agents yet</span>';
            return;
        }

        agents.forEach(function (a) {
            var color   = colorFor(a.id);
            var roleStr = niceRole(a.role);

            var box = document.createElement('div');
            box.className = 'agent-cmd-box';
            box.style.borderTopColor = color;
            box.style.setProperty('--agent-color', color);

            var taId     = 'cmd-input-' + a.id.replace(/\W/g, '_');
            var statusId = 'cmd-status-' + a.id.replace(/\W/g, '_');

            box.innerHTML =
                '<div class="agent-cmd-header">' +
                    '<span class="agent-cmd-dot" style="background:' + color + '"></span>' +
                    '<span class="agent-cmd-name">' + esc(displayName(a.id)) + '</span>' +
                    '<span class="agent-cmd-role" data-agent-role="' + esc(a.id) + '">' +
                        esc(roleStr) + '</span>' +
                '</div>' +
                '<textarea id="' + taId + '" class="agent-cmd-textarea" rows="3"' +
                          ' placeholder="Message ' + esc(displayName(a.id)) + '…"></textarea>' +
                '<div class="agent-cmd-footer">' +
                    '<span class="agent-cmd-status" id="' + statusId + '"></span>' +
                    '<button class="agent-cmd-send" style="--agent-color:' + color +
                            ';border-color:' + color + ';color:' + color + '">Send →</button>' +
                '</div>';

            /* Wire up send button */
            var btn      = box.querySelector('.agent-cmd-send');
            var inputEl  = box.querySelector('.agent-cmd-textarea');
            var statusEl = box.querySelector('.agent-cmd-status');
            var agentId  = a.id;

            btn.addEventListener('click', function () {
                sendCommandToAgent(agentId, inputEl, statusEl);
            });
            inputEl.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendCommandToAgent(agentId, inputEl, statusEl);
                }
            });

            container.appendChild(box);
        });
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Sidebar — message log
     *
     *  Merges inter-agent messages (from GLOBAL_MESSAGE_LOG) with supervisor
     *  command entries (from supervisor_channel).  We only append new entries
     *  to avoid flickering.
     * ═══════════════════════════════════════════════════════════════════════ */

    function renderLog(msgs, cmds, agentStatus) {
        /* Build a merged, ordered list sorted by timestamp. */
        var all = [];

        msgs.forEach(function (m) {
            all.push({
                from:       m.from  || '?',
                to:         m.to    || 'all',
                text:       m.text  || '',
                type:       m.message_type || 'message',
                motivation: '',
                _sort:      m.ts || 0,
            });
        });

        cmds.forEach(function (c) {
            all.push({
                from:       'SUPERVISOR',
                to:         c.target_agent || 'all',
                text:       c.text || '',
                type:       'supervisor_command',
                motivation: '',
                _sort:      c.ts || 0,
            });
        });

        /* Agent-private human-only entries (plan_human / action_human).
           These are never forwarded to other agents. */
        (agentStatus || []).forEach(function (s) {
            all.push({
                from:       s.from || '?',
                to:         'HUMAN',
                text:       s.text || '',
                type:       s.message_type || 'action_human',
                motivation: s.motivation || '',
                _sort:      s.ts || 0,
            });
        });

        /* Sort so everything appears in chronological order */
        all.sort(function (a, b) { return a._sort - b._sort; });

        var log   = document.getElementById('msg-log');
        var total = all.length;

        if (total < renderedEntries) {
            log.innerHTML   = '';
            renderedEntries = 0;
        }

        var atBottom = (log.scrollTop + log.clientHeight + 12) >= log.scrollHeight;

        for (var i = renderedEntries; i < total; i++) {
            var e   = all[i];
            var row = document.createElement('div');
            row.className = 'msg-row ' + e.type;

            var toStr = (e.to && e.to !== 'all' && e.to !== 'HUMAN')
                ? esc(e.to) : esc(e.to || 'all');

            /* Colour the sender label */
            var fromColor;
            if (e.from === 'SUPERVISOR') {
                fromColor = '#fa6';
            } else if (e.type === 'plan_human') {
                fromColor = '#d4a017';
            } else if (e.type === 'action_human') {
                fromColor = '#7cc4e8';
            } else {
                fromColor = colorFor(e.from);
            }

            /* Label shown in the type chip */
            var typeLabel = e.type;
            if (e.type === 'plan_human')   { typeLabel = '📋 plan'; }
            if (e.type === 'action_human') { typeLabel = '⚙ action'; }

            var html =
                '<div class="msg-header">' +
                    '<span class="msg-from" style="color:' + fromColor + '">' + esc(e.from) + '</span>' +
                    '<span class="msg-to">→ ' + toStr + '</span>' +
                    '<span class="msg-type">' + esc(typeLabel) + '</span>' +
                '</div>' +
                '<div class="msg-body">' + esc(e.text) + '</div>';

            /* Append full motivation block for plan_human and action_human
               (suppressed for action_human when motivation is empty) */
            if (e.motivation) {
                html += '<div class="msg-motivation">' + esc(e.motivation) + '</div>';
            }

            row.innerHTML = html;
            log.appendChild(row);
        }

        renderedEntries = total;
        if (atBottom) { log.scrollTop = log.scrollHeight; }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Send command (per-agent or broadcast)
     *
     *  agentId  — specific agent ID, or 'all' for broadcast
     *  inputEl  — the <input> whose value is the message text
     *  statusEl — the <span> that receives feedback
     * ═══════════════════════════════════════════════════════════════════════ */

    function sendCommandToAgent(agentId, inputEl, statusEl) {
        var text = (inputEl.value || '').trim();
        if (!text) {
            _setStatus(statusEl, 'Type a message first.', true);
            return;
        }

        _setStatus(statusEl, 'sending…', false);
        inputEl.disabled = true;

        fetch('/supervisor_command', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text: text, target_agent: agentId }),
        })
        .then(function (r) {
            return r.json().catch(function () { return {}; })
                   .then(function (d) { return { r: r, d: d }; });
        })
        .then(function (o) {
            if (!o.r.ok || !o.d.ok) { throw new Error(o.d.error || 'HTTP ' + o.r.status); }
            _setStatus(statusEl, agentId === 'all' ? 'sent to all ✓' : 'sent ✓', false);
            inputEl.value = '';
        })
        .catch(function (e) {
            _setStatus(statusEl, '✗ ' + e.message, true);
        })
        .finally(function () {
            inputEl.disabled = false;
        });
    }

    function _setStatus(el, msg, isErr) {
        if (!el) { return; }
        el.textContent = msg;
        el.className   = 'agent-cmd-status' + (isErr ? ' err' : ' ok');
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Pause / resume
     * ═══════════════════════════════════════════════════════════════════════ */

    function applyPausedState(paused) {
        lv_matrx_paused = paused;
        var btn    = document.getElementById('sup-pause-btn');
        var banner = document.getElementById('paused-banner');
        if (paused) {
            btn.textContent = '▶ Resume';
            btn.classList.add('paused');
            if (banner) { banner.classList.add('visible'); }
        } else {
            btn.textContent = '⏸ Pause';
            btn.classList.remove('paused');
            if (banner) { banner.classList.remove('visible'); }
        }
        adjustBodyHeight();
    }

    function togglePause() {
        var btn = document.getElementById('sup-pause-btn');
        btn.disabled = true;
        fetch('/supervisor_pause', { method: 'POST' })
            .then(function (r) { return r.json().catch(function () { return {}; }); })
            .then(function (d) {
                if (d.ok !== false) { applyPausedState(!!d.paused); }
            })
            .catch(function (e) { console.warn('pause toggle failed:', e); })
            .finally(function () { btn.disabled = false; });
    }

    /* ═══════════════════════════════════════════════════════════════════════
     *  Utility
     * ═══════════════════════════════════════════════════════════════════════ */

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

})();
