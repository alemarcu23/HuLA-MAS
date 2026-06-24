"""
Tool registry.
"""

from typing import Dict, Any, List, Tuple

from langchain.tools import tool

@tool
def MoveNorth():
    """Move one cell north (decreases y by 1)."""
    return 'MoveNorth', {}


@tool
def MoveSouth():
    """Move one cell south (increases y by 1)."""
    return 'MoveSouth', {}


@tool
def MoveEast():
    """Move one cell east (increases x by 1)."""
    return 'MoveEast', {}


@tool
def MoveWest():
    """Move one cell west (decreases x by 1)."""
    return 'MoveWest', {}


@tool
def MoveTo(x: int, y: int):
    """Navigate to a specific grid coordinate using A* pathfinding.

    Args:
        x: Target column (east-west axis).
        y: Target row (north-south axis).
    """
    return 'MoveTo', {'x': x, 'y': y}

@tool
def MoveToArea(area: int):
    """Navigate to a specific Area using A* pathfinding.

    Args:
        area: the number of the area to navigate to
    """
    return 'MoveToArea', {'area': area}


@tool
def NavigateToDropZone():
    """Navigate to the rescue drop zone to deliver a carried victim.
    Use this after CarryObject or CarryObjectTogether; follow with Drop to score points."""
    return 'NavigateToDropZone', {}


@tool
def CarryObject(object_id: str):
    """Pick up and carry a victim solo. Only valid if your medical capability allows it.
    - medical=high: can carry ALL victims (mild and critical) alone.
    - medical=medium: can carry MILDLY injured victims alone; critical requires CarryObjectTogether.
    - medical=low: CANNOT carry any victim alone; always use CarryObjectTogether.
    You must be adjacent (Chebyshev distance ≤ 1) to the victim. After picking up, use
    NavigateToDropZone then Drop to score points.

    Args:
        object_id: The ID of the victim to carry (from observation.nearby_victims).
    """
    return 'CarryObject', {'object_id': object_id}


@tool
def CarryObjectTogether(object_id: str, partner_id: str):
    """Cooperatively carry a critically injured victim with a partner agent.
    Required when your medical capability is low, or when carrying a critical victim regardless of capability.

    Rendezvous is handled automatically: once you call this, the infrastructure
    will navigate BOTH you and your partner to the victim via A*, fire the
    action when both are adjacent, and then auto-pilot both agents to the
    drop zone and drop the victim cooperatively. You do NOT need to call
    NavigateToDropZone or Drop yourself for a cooperative carry.

    Args:
        object_id: The ID of the victim to carry cooperatively (from observation.nearby_victims).
        partner_id: REQUIRED — the object_id of the teammate from observation.teammates (must match exactly).
    """
    return 'CarryObjectTogether', {'object_id': object_id, 'partner_id': partner_id}


@tool
def Drop():
    """Drop the currently carried object at the current grid position.
    Use this at the drop zone after NavigateToDropZone to score rescue points."""
    return 'Drop', {}


@tool
def RemoveObject(object_id: str):
    """Remove a small stone or fallen tree obstacle solo. Capability constraints apply:
    - strength=high: can remove trees, stones, and rocks alone.
    - strength=medium: can remove trees and small stones alone; big rocks require RemoveObjectTogether.
    - strength=low: can only remove fallen trees alone; stones and rocks require RemoveObjectTogether.
    Note: ONLY the rescue robot (RescueBot) can remove trees; human agents cannot.
    You must be adjacent (Chebyshev distance ≤ 1) to the obstacle.
    Big grey rocks ALWAYS require RemoveObjectTogether regardless of strength.

    Args:
        object_id: The ID of the obstacle to remove (from observation.nearby_obstacles).
    """
    return 'RemoveObject', {'object_id': object_id}


@tool
def RemoveObjectTogether(object_id: str, partner_id: str):
    """Cooperatively remove a big grey rock obstacle with a partner agent.
    Big rocks ALWAYS require both agents — solo removal is never possible regardless of strength.
    BOTH agents must be adjacent (Chebyshev distance ≤ 1) to the rock before calling this.

    Args:
        object_id: The ID of the rock to remove cooperatively (from observation.nearby_obstacles).
        partner_id: REQUIRED — the object_id of the adjacent teammate from observation.teammates.
    """
    return 'RemoveObjectTogether', {'object_id': object_id, 'partner_id': partner_id}

@tool
def SearchArea(area: int):
    """Systematically search all cells inside an area for victims and obstacles.

    THIS IS THE ONLY ACTION THAT GUARANTEES VICTIM DETECTION IN AN AREA.
    MoveTo/MoveToArea do NOT search — they only navigate and will MISS victims.
    Do NOT substitute MoveTo when the plan says "Search area N for victims".

    You must be at the door of the area (Chebyshev distance ≤ 1) before calling
    this action. The agent will visit every inside cell via a serpentine path and
    return to the door.

    If SearchArea stalls (no movement progress across ticks), an obstacle is
    likely blocking the path inside or at the door — clear it first with
    RemoveObject or RemoveObjectTogether, then call SearchArea again.

    Args:
        area: The number of the area to search (1-14).
    """
    return 'SearchArea', {'area': area}


@tool
def SendMessage(message: str, send_to: str, message_type: str = "message"):
    """Send a message to one or all teammates. This uses your action for this tick.
    Use sparingly — keep messages to 1-2 sentences.

    Args:
        message: The message content to send (1-2 sentences max).
        send_to: Agent name for a directed message, or "all" for a broadcast.
        message_type: One of:
            - "ask_help": ask teammate to do the task for you.
            - "help": offer help to an agent.
            - "message": general status update or information sharing.
    """
    return 'SendMessage', {'message': message, 'send_to': send_to,
        'message_type': message_type}


ALL_ACTION_TOOLS = [
    MoveTo, MoveToArea, NavigateToDropZone, SearchArea,
    CarryObject, CarryObjectTogether, Drop, RemoveObject, RemoveObjectTogether, SendMessage
]

def build_tool_schemas() -> Tuple[Dict[str, Any], List[Dict]]:
    import logging
    from langchain_core.utils.function_calling import convert_to_openai_tool

    logger = logging.getLogger('tool_registry')

    tools_by_name: Dict[str, Any] = {t.name: t for t in ALL_ACTION_TOOLS}

    try:
        tool_schemas: List[Dict] = [
            convert_to_openai_tool(t) for t in ALL_ACTION_TOOLS
        ]
    except Exception as exc:
        logger.warning(
            "convert_to_openai_tool failed (%s); using matrx_tool_description fallback", exc
        )

    return tools_by_name, tool_schemas