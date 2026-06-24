"""
main.py — HuLA-MAS Search & Rescue with LLM agents
"""
import os
import pathlib
import argparse

import requests
import yaml

from matrx.grid_world_creation.WorldBuilder import create_builder
from matrx.agents.llm.modules.agent_infra.async_model_prompting import (
    init_agent_pool, shutdown_agent_pool, configure_sampling,
)
from matrx.agents.llm.modules.memory_module import configure_memory
from matrx.metrics.agent_metrics import ActionFileLogger
from matrx.logger.OutputLogger import output_logger


def _cycle(values, n, default):
    """Return exactly n entries, cycling `values` (or [default]) to fill."""
    values = values or [default]
    return [values[i % len(values)] for i in range(n)]


if __name__ == "__main__":
    fld = os.getcwd()

    parser = argparse.ArgumentParser(description="HuLA-MAS LLM Search & Rescue simulation")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML run config (default: config.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    run_cfg = cfg.get('run', {})
    llm_cfg = cfg.get('llm', {})
    agents_cfg = cfg.get('agents', {})
    world_cfg = cfg.get('world', {})
    server_cfg = cfg.get('server', {})

    map_name = cfg.get('map', 'official')
    num_agents = agents_cfg.get('num_agents', 2)

    # Per-agent settings cycle to num_agents (shorter lists repeat).
    agent_presets = _cycle(agents_cfg.get('capability_presets'), num_agents, 'generalist')
    agent_roles = _cycle(agents_cfg.get('roles'), num_agents, 'generalist')
    reasoning_strategies = _cycle(agents_cfg.get('reasoning_strategies'), num_agents, 'io')
    planning_strategies = _cycle(agents_cfg.get('planning_strategies'), num_agents, 'io')

    # Shared configuration data: capability presets, role goals, task rules.
    presets = agents_cfg.get('presets', {})
    role_goals = agents_cfg.get('role_goals', {})
    game_rules = cfg.get('task', {}).get('game_rules', '')
    joint_action_ask_trigger = agents_cfg.get('joint_action_ask_trigger', 'auto_bridge')

    enable_gui = server_cfg.get('enable_gui', True)
    vis_port = server_cfg.get('vis_port', 3000)
    api_port = server_cfg.get('api_port', 3001)

    print(f"Map: {map_name} | {num_agents} LLM agent(s), no humans | "
          f"model={llm_cfg.get('model')} backend={llm_cfg.get('backend')}")

    # Per-run action log directory.
    log_dir = os.path.join(fld, 'logs', 'llm_run')
    os.makedirs(log_dir, exist_ok=True)
    ActionFileLogger.init(os.path.join(log_dir, 'agent_actions.csv'))

    # Apply LLM sampling defaults from config (no-op for absent/None keys).
    configure_sampling(**(llm_cfg.get('sampling') or {}))

    # Apply memory tuning params from config.
    configure_memory(**(cfg.get('memory') or {}))

    # Size the LLM thread pool to the number of agents.
    init_agent_pool(num_agents, backend=llm_cfg.get('backend', 'ollama_sdk'))

    builder = None
    vis_thread = None
    try:
        builder = create_builder(
            task_type=run_cfg.get('task_type', 'official'),
            condition=run_cfg.get('condition', 'normal'),
            name='rescuebot',
            folder=fld,
            map_name=map_name,
            num_agents=num_agents,
            agent_model=llm_cfg.get('model', 'qwen2.5:3b'),
            api_base=llm_cfg.get('api_base'),
            agent_presets=agent_presets,
            agent_roles=agent_roles,
            capability_knowledge=agents_cfg.get('capability_knowledge', 'informed'),
            reasoning_strategies=reasoning_strategies,
            planning_strategies=planning_strategies,
            presets=presets,
            role_goals=role_goals,
            game_rules=game_rules,
            world_seed=world_cfg.get('seed'),
            log_dir=log_dir,
            joint_action_ask_trigger=joint_action_ask_trigger,
        )

        # Configure the MATRX REST API port before startup.
        from matrx.api import api as matrx_api
        matrx_api.set_api_port(api_port)

        media_folder = pathlib.Path().resolve()
        builder.startup(media_folder=media_folder)

        if enable_gui:
            from SaR_gui import visualization_server
            print("Starting custom visualizer")
            vis_thread = visualization_server.run_matrx_visualizer(
                verbose=False, media_folder=media_folder, vis_port=vis_port
            )

        world = builder.get_world()
        print("Started world...")
        # No human in the loop → don't wait for a browser to un-pause the world.
        builder.api_info['matrx_paused'] = False
        world.run(builder.api_info)
        print("DONE!")

    finally:
        # Shut down the visualizer thread.
        if enable_gui and vis_thread is not None:
            try:
                requests.get(f"http://localhost:{vis_port}/shutdown_visualizer", timeout=5)
                vis_thread.join(timeout=5)
            except Exception:
                pass

        # Close the per-agent action log.
        logger = ActionFileLogger.get()
        if logger:
            logger.close()

        # Shut down the LLM thread pool.
        shutdown_agent_pool()

        # Persist per-agent + aggregate metrics (AgentMetricsTracker.to_dict()).
        try:
            from matrx.metrics.simulation_metrics import SimulationMetrics
            sim_metrics = SimulationMetrics.get()
            if sim_metrics is not None:
                errors = {k: v for k, v in sim_metrics.save_incremental(log_dir).items() if v}
                if errors:
                    print(f"[main] Per-agent metrics partial errors: {errors}")
                try:
                    report = sim_metrics.aggregate(config=cfg)
                    sim_metrics.save(os.path.join(log_dir, 'simulation_report.json'), report)
                except Exception as e:
                    print(f"[main] Aggregate metrics error: {e}")
                print(f"[main] Saved agent metrics to {os.path.join(log_dir, 'agent_metrics')}")
                SimulationMetrics.reset()
        except Exception as e:
            print(f"[main] Metrics save error: {e}")

        # Aggregate per-run output.
        try:
            output_logger(fld, log_dir=log_dir)
        except Exception as e:
            print(f"[main] Output logger error: {e}")

        if builder is not None:
            builder.stop()
