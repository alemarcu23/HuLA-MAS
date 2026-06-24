"""Lifecycle of the joint-action state machine: request → consent → drive →
fire → deliver → done, for AI–AI and human–AI, plus consent-only and timeouts."""

ROCK = {'object_id': 'rock_5', 'location': (3, 4)}
ROCK_NAV = {'rock_5': {'location': (3, 4), 'is_traversable': False, 'is_collectable': False}}
VICTIM = {'object_id': 'critically injured man 1', 'location': (10, 10), 'severity': 'critical'}
VICTIM_NAV = {'critically injured man 1': {'location': (10, 10), 'is_traversable': True, 'is_collectable': True}}


def _engage_remove_requester(make_agent):
    a = make_agent('RescueBot').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((5, 4))
    a._joint_on_action('RemoveObjectTogether', {'object_id': 'rock_5'})
    return a


# ── request side ────────────────────────────────────────────────────────────
def test_auto_bridge_sends_ask_help_on_tool_call(make_agent):
    a = _engage_remove_requester(make_agent)
    ask = a.last_sent('ask_help')
    assert ask is not None and ask['target_id'] == 'rock_5' and ask['kind'] == 'remove'
    assert a._joint['phase'] == 'await_consent'
    assert a._joint_active() is True


def test_llm_explicit_requires_ask_help_first(make_agent):
    a = make_agent('RescueBot', trigger='llm_explicit').set_world(
        obstacles=[ROCK], nav=dict(ROCK_NAV)).at((5, 4))
    dec, msg = a._joint_on_action('RemoveObjectTogether', {'object_id': 'rock_5'})
    assert dec == 'error' and 'ask_help' in msg
    assert a._joint is None
    assert a.last_sent('ask_help') is None


def test_explicit_ask_help_message_resolves_and_engages(make_agent):
    a = make_agent('RescueBot', trigger='llm_explicit').set_world(
        obstacles=[ROCK], nav=dict(ROCK_NAV)).at((5, 4))
    dec, extra = a._joint_on_outgoing_message(None, 'ask_help', 'need a hand with the rock', {})
    assert dec == 'send'
    assert extra['target_id'] == 'rock_5' and extra['kind'] == 'remove'
    assert a._joint['phase'] == 'await_consent'


# ── consent side ──────────────────────────────────────────────────────────────
def test_consenter_commits_as_positioner(make_agent):
    a = _engage_remove_requester(make_agent)
    b = make_agent('RescueBot1').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV))
    b.receive(a.last_sent('ask_help'), 'RescueBot')
    assert b._known_request and b._known_request['requester'] == 'RescueBot'
    dec, extra = b._joint_on_outgoing_message('RescueBot', 'help', 'yes', {})
    assert dec == 'send' and extra['text'] == 'yes'
    assert b._joint['role'] == 'positioner' and b._joint['phase'] == 'go'
    assert b._joint['partner_is_human'] is False


def test_requester_becomes_firer_on_yes(make_agent):
    a = _engage_remove_requester(make_agent)
    a.receive({'message_type': 'help', 'from': 'RescueBot1', 'text': 'yes'}, 'RescueBot1')
    assert a._joint['role'] == 'firer' and a._joint['phase'] == 'go'
    assert a._joint['partner_id'] == 'RescueBot1'


def test_consent_only_no_keeps_waiting(make_agent):
    a = _engage_remove_requester(make_agent)
    a.receive({'message_type': 'help', 'from': 'RescueBot1', 'text': 'no'}, 'RescueBot1')
    assert a._joint['phase'] == 'await_consent'  # never auto-assigned


def test_already_engaged_declines_other_requests(make_agent):
    b = make_agent('RescueBot1')
    b._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot', 'partner_is_human': False,
                'role': 'positioner', 'phase': 'go', 'started_tick': 0}
    dec, extra = b._joint_on_outgoing_message('Someone', 'help', 'yes', {})
    assert extra['text'] == 'no'


# ── drive + fire + completion ─────────────────────────────────────────────────
def test_firer_fires_remove_when_both_adjacent(make_agent):
    a = make_agent('RescueBot').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((4, 4))
    a._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    a.state_for_navigation['RescueBot1'] = {'location': (3, 5), 'isAgent': True}
    act = a._joint_infra(None)
    assert act[0] == 'RemoveObjectTogether' and act[1]['object_id'] == 'rock_5'
    assert act[1]['partner_name'] == 'RescueBot1'


def test_firer_waits_until_partner_adjacent(make_agent):
    a = make_agent('RescueBot').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((4, 4))
    a._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    a.state_for_navigation['RescueBot1'] = {'location': (9, 9), 'isAgent': True}  # far
    act = a._joint_infra(None)
    assert act[0] == 'Idle'  # does not fire yet


def test_positioner_announces_and_never_fires(make_agent):
    b = make_agent('RescueBot1').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((4, 4))
    b._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot', 'partner_is_human': False,
                'role': 'positioner', 'phase': 'go', 'started_tick': 0}
    act = b._joint_infra(None)
    assert act[0] == 'Idle'
    assert b.last_sent('help')['text'].startswith('In position')


def test_remove_success_finishes_and_broadcasts_done(make_agent):
    a = make_agent('RescueBot').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((4, 4))
    a._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    a.result('RemoveObjectTogether', ok=True)
    a._joint_infra(None)
    assert a._joint is None
    assert a.last_sent('help_done')['target_id'] == 'rock_5'


def test_help_done_clears_partner_engagement(make_agent):
    b = make_agent('RescueBot1')
    b._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot', 'partner_is_human': False,
                'role': 'positioner', 'phase': 'go', 'started_tick': 0}
    b.receive({'message_type': 'help_done', 'from': 'RescueBot', 'target_id': 'rock_5'}, 'RescueBot')
    assert b._joint is None


def test_carry_fires_delivers_and_records_rescue(make_agent):
    a = make_agent('RescueBot').set_world(victims=[VICTIM], nav=dict(VICTIM_NAV)).at((10, 10))
    a._joint = {'kind': 'carry', 'target_id': 'critically injured man 1', 'target_loc': (10, 10),
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    a.state_for_navigation['RescueBot1'] = {'location': (10, 11), 'isAgent': True}
    # fire the carry
    act = a._joint_infra(None)
    assert act[0] == 'CarryObjectTogether'
    # success → deliver phase
    a.result('CarryObjectTogether', ok=True)
    a._joint_infra(None)
    assert a._joint['phase'] == 'deliver'
    # at the drop zone → drop together
    a.at((23, 8))
    act = a._joint_infra(None)
    assert act[0] == 'DropObjectTogether'
    # drop success → rescue recorded cooperatively + done
    a.result('DropObjectTogether', ok=True)
    a._joint_infra(None)
    assert a._joint is None
    rescued = a.shared_memory.retrieve('rescued_victims') or []
    assert any(v['victim_id'] == 'critically injured man 1' and v['method'] == 'cooperative'
               for v in rescued)
    assert a.last_sent('help_done') is not None


# ── timeouts ──────────────────────────────────────────────────────────────────
def test_consent_timeout_abandons_request(make_agent):
    a = _engage_remove_requester(make_agent)
    a.tick(10_000)
    a._joint_infra(None)
    assert a._joint is None and a._pending_help_abandon


def test_goto_timeout_abandons_engagement(make_agent):
    a = make_agent('RescueBot').set_world(obstacles=[ROCK], nav=dict(ROCK_NAV)).at((9, 9))
    a._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': (3, 4),
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    a.tick(10_000)
    a._joint_infra(None)
    assert a._joint is None and a._pending_help_abandon


# ── human–AI ──────────────────────────────────────────────────────────────────
def test_human_consent_makes_ai_positioner_human_fires(make_agent):
    a = make_agent('RescueBot', roster=('RescueBot',)).set_world(  # 'human' not in roster
        victims=[VICTIM], nav=dict(VICTIM_NAV)).at((9, 10))
    a._joint_on_action('CarryObjectTogether', {'object_id': 'critically injured man 1'})
    assert a._joint['phase'] == 'await_consent'
    a.receive({'text': 'Rescue together'}, 'human')  # GUI button press
    assert a._joint['role'] == 'positioner' and a._joint['partner_is_human'] is True


def test_human_initiate_remove_synthesizes_request(make_agent):
    a = make_agent('RescueBot', roster=('RescueBot',)).set_world(
        obstacles=[ROCK],
        nav={**ROCK_NAV, 'human': {'location': (4, 4), 'isAgent': True}})
    a.receive({'text': 'help remove'}, 'human')
    assert a._known_request and a._known_request['kind'] == 'remove'
    assert a._known_request['target_id'] == 'rock_5'
