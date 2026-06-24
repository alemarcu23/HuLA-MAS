# HuLA-MAS: Heterogeneity in Human-LLM Multiagent Systems

This is the repository for the HuLA-MAS framework for studying heterogeneity in human-agent teams. Currently, the framework allows the study of mixed teams (1> humans, 1> agents), or fully artificial teams. 

> **See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for more information on the game loop, the LLM agent, and how to extend the framework (new metrics, agents, modules, strategies, tools, backends).


## Search and rescue environment
The repository uses the [MATRX software package](https://matrx-software.com/) to create a simulated search and rescue task in a two-dimensional grid environment. The environment consists of multiple areas, injured victims, and obstacles blocking area entrances.

![environment-chat-1](https://user-images.githubusercontent.com/54837051/204800699-89ed7159-d329-4f95-8441-acb601ff90a5.png)

### Task
The objective of the task is to find eight target victims in the different areas and carry them to the drop zone. Rescuing mildly injured victims (yellow color) adds three points to the total score, rescuing critically injured victims (red color) adds six points to the total score. The world terminates after successfully rescuing all target victims, the corresponding output logs will then be saved in the 'logs' folder.

## Installation
Download or clone this repository and the required dependencies listed in the 'requirements.txt' file. We recommend the use of Python 3.13, and to create a virtual environment for this project. You can use the following step by step installation steps after cloning or downloading this repository:
- Install the required dependencies through 'pip install -r requirements.txt'.

## Running the simulation
The simulation is launched from the command line. The default run pairs **one** human player with RescueBot (and any extra AI agents defined in the config — see [Configuration](#configuration)):

```bash
python main.py
```

`main.py` accepts the following flags:

| Flag | Choices / type | Default | Description |
| --- | --- | --- | --- |
| `--multi-human` | flag | off | Enables the multi-player lobby UI so **several humans** can join the same world. Off by default. See [Multi-human support](#multi-human-support). |
| `--config` | path | `config.yaml` | YAML file describing the team composition. See [Configuration](#configuration). |

After launching:
- Go to http://localhost:3000 and clear your old cache of the page by pressing 'ctrl' + 'F5'.
- Open the 'God' and human agent view. Start the task in the 'God' view with the play icon in the top right of the toolbar. The 'God' view is shown in the image above, cannot be used to control agents, and should only be used for debugging purposes.
- Go to the human agent view to start the task. Open the messaging interface by pressing the chat box icon in the top right of the toolbar. You can now start playing the task.


### Controls
The human agent is controlled with the keyboard (defined in `key_action_map` in `matrx/grid_world_creation/WorldBuilder.py`):

| Key | Action |
| --- | --- |
| Arrow keys | Move (north / east / south / west) |
| `q` | Carry an object (pick up alone) |
| `w` | Drop a carried object |
| `e` | Remove an obstacle alone |
| `a` | Start carrying a victim **together** with RescueBot |
| `s` | Drop a victim that is being carried **together** |
| `d` | Remove an obstacle **together** with RescueBot |

## Configuration
The team composition is described in a small YAML file (`config.yaml` by default). It controls how many agents are in the team and how many of them are humans:

```yaml
# config.yaml
agents_per_team: 2        # total number of agents in the team (humans + AI)
human_agents_per_team: 1  # how many of those are humans
```

Humans are named `Human1`, `Human2`, … and are spawned near the drop zone.

## Multi-human support
This section explains how to run the simulation with multiple humans. 

Run `main_multi_human.py`, which coordinates players through named sessions and starts the world paused until both players are ready. The easiest way to run it for others on the network is via Docker — see [Multi-Laptop Game Setup](#multi-laptop-game-setup) below.

`main_multi_human.py` adds a `--session-id` flag (default `default_session`) used to keep concurrent games isolated from each other:

```bash
python main_multi_human.py --session-id session1
```


# Repo Overview
All simulation/game code now lives under the `matrx/` package.
- `matrx/actions`: Contains the engine actions plus 'CustomActions.py', which defines the various customized actions like 'CarryObjectTogether' and 'DropObjectTogether'.
- `matrx/agents`: Contains the engine agent code plus the brains: 'ArtificialBrain.py' / 'HumanBrain.py' initialize RescueBot and the human agent. The LLM-driven agents live in the 'matrx/agents/llm' subpackage and the rule-based agent is in 'RuleBasedAgent.py'
- `matrx/logger`: Contains the engine loggers plus 'ActionLogger.py' and 'OutputLogger.py'. The action logger saves the actions and locations of both human and RescueBot during every tick of the task. In the MATRX world, all time is measured in ticks instead of seconds, and actions and messages are all executed at a single tick. The tick duration is set at 0.1, which means around 10 ticks are executed in a second. In addition, the output logger creates one output file and line with the time it took to finish the task (in ticks) and the total number of human and agent actions during the task. Finally, the output logger saves the trust belief values to the 'allTrustBeliefs.csv' file mentioned above. It is important to know that the output logger is only called when the task is successfully completed, or when you press the stop button in the 'God' view (the square button next to the play button). 
- `matrx/worlds`: Contains the 'WorldBuilder.py' file defining the search and rescue environment and task. 
- `matrx/helpers` and `matrx/metrics`: Supporting utilities (perception/logic/navigation helpers, TOON serialization) and the simulation/agent metrics trackers used by the agents.


## Multi-Laptop Game Setup
This runs the session-based multi-human server (`main_multi_human.py`) inside Docker so players on different laptops can join the same world over the local network. The task type, condition, and session id are set through environment variables in `docker-compose.yml` (`TASK_TYPE`, `CONDITION`, `SESSION_ID`).

### Starting the Server

1. Navigate to the project directory
2. Start the Docker container:
```bash
docker-compose up -d
```

3. Find your server's IP address:
```bash
ipconfig getifaddr en0
```

### Accessing the Game

Use the server IP address to access the game from external laptops. Replace `192.168.0.122` with your actual IP:

- **God View**: `http://192.168.0.122:3000/god`
- **Human Agent 1**: `http://192.168.0.122:3000/human-agent/Human1`
- **Human Agent 2**: `http://192.168.0.122:3000/human-agent/Human2`

Each human opens their own URL, lands on the lobby screen, and presses *Ready*; the world starts once both players have joined the session.

### Stopping the Server
To stop docker use:
```bash
docker-compose down
```

## Maps and worlds
A "map" is the physical layout of a world: the grid size, the rooms/areas with their doors and doormats, the obstacles blocking entrances, the victims to rescue, the drop zone, and the decorative tiles (roofs, streets, water, signs). Two task types ship with their own map. See how to create custom maps in `maps/` folder