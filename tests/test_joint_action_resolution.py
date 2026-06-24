"""Target resolution / salvage — turning the weak LLM's vague or wrong requests
(a teammate's name, a generic word, the wrong kind, a bare location) into a real
on-grid object, so joint actions can actually fire."""

VICTIMS = [
    {'object_id': 'critically injured man 1', 'location': (10, 10), 'severity': 'critical'},
    {'object_id': 'healthy woman 2', 'location': (9, 10), 'severity': 'healthy'},
]
OBSTACLES = [{'object_id': 'rock_5', 'location': (3, 3)}]
NAV = {'critically injured man 1': {'location': (10, 10)}, 'rock_5': {'location': (3, 3)}}


def _agent(make_agent):
    return make_agent('RescueBot').at((9, 9)).set_world(
        victims=list(VICTIMS), obstacles=list(OBSTACLES), nav=dict(NAV))


def test_partner_name_id_resolves_to_real_victim(make_agent):
    tid, loc, kind = _agent(make_agent)._resolve_target('RescueBot1', None, 'carry')
    assert tid == 'critically injured man 1' and kind == 'carry'


def test_resolution_skips_healthy_victims(make_agent):
    tid, _, _ = _agent(make_agent)._resolve_target('RescueBot1', None, 'carry')
    assert 'healthy' not in tid


def test_generic_word_resolves_to_real_obstacle(make_agent):
    tid, loc, kind = _agent(make_agent)._resolve_target('rock', None, 'remove')
    assert tid == 'rock_5' and loc == (3, 3) and kind == 'remove'


def test_exact_locatable_id_is_kept(make_agent):
    tid, _, kind = _agent(make_agent)._resolve_target('rock_5', None, None)
    assert tid == 'rock_5' and kind == 'remove'


def test_wrong_kind_self_corrects(make_agent):
    # asking to "carry" something that is not a victim → nearest real victim
    tid, _, kind = _agent(make_agent)._resolve_target('rock', None, 'carry')
    assert tid == 'critically injured man 1' and kind == 'carry'


def test_ambiguous_ask_picks_nearer_object(make_agent):
    # agent at (9,9): victim (10,10) is nearer than obstacle (3,3)
    tid, _, kind = _agent(make_agent)._resolve_ask_target({}, '')
    assert tid == 'critically injured man 1' and kind == 'carry'


def test_location_match_overrides_generic_id(make_agent):
    tid, loc, _ = _agent(make_agent)._resolve_target('rock', (3, 3), 'remove')
    assert tid == 'rock_5' and loc == (3, 3)


def test_live_vision_fallback_when_belief_empty(make_agent):
    a = make_agent('RescueBot').at((4, 4))
    a.WORLD_STATE_GLOBAL = {'victims': [], 'obstacles': []}  # stale/empty belief
    a.state_for_navigation = {'big_stone_9': {'location': (3, 4), 'is_traversable': False,
                                              'is_collectable': False}}
    tid, loc, kind = a._resolve_target('stone', (3, 4), 'remove')
    assert tid == 'big_stone_9' and loc == (3, 4) and kind == 'remove'


def test_walls_are_never_treated_as_removable(make_agent):
    a = make_agent('RescueBot').at((4, 4))
    a.WORLD_STATE_GLOBAL = {'victims': [], 'obstacles': []}
    # an intraversable wall sits exactly where the LLM pointed; no real obstacle nearby
    a.state_for_navigation = {'wall_3_4': {'location': (3, 4), 'is_traversable': False,
                                           'is_collectable': False}}
    tid, _, _ = a._resolve_target('rock', (3, 4), 'remove')
    assert tid != 'wall_3_4'  # walls/doors are not joint-removable


def test_bogus_tool_call_still_bridges_to_real_target(make_agent):
    a = _agent(make_agent)
    dec, _ = a._joint_on_action('CarryObjectTogether', {'object_id': 'RescueBot1'})
    ask = a.last_sent('ask_help')
    assert dec == 'idle'
    assert ask and ask['target_id'] == 'critically injured man 1'
    assert a._joint['target_id'] == 'critically injured man 1'
