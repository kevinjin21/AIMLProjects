"""Robot action primitives for symbolic planning.

This module defines the basic actions a robot can take and their effects
on the world state.
"""

from typing import List, Tuple
from .world import WorldState

# Type alias for action results: (new_state, success, message)
ActionResult = Tuple[WorldState, bool, str]

# Valid locations in the environment
VALID_LOCATIONS = ["table", "shelf", "floor"]

# Valid objects that can be manipulated (can be made dynamic later)
VALID_OBJECTS = ["box", "book"]

def move_to(state: WorldState, location: str) -> ActionResult:
    """Move the robot to a specified location.
    
    Args:
        state: Current world state
        location: Target location to move to
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if robot successfully moved
        - message: Description of the result
    """
    location = location.lower()  # Case-insensitive
    if location not in VALID_LOCATIONS:
        return state, False, f"Invalid location: {location}"
    new_state = state.clone()
    new_state.robot.location = location
    new_state.step_count += 1
    return new_state, True, f"Moved to {location}"

def pick(state: WorldState, obj: str) -> ActionResult:
    """Pick up an object at the robot's current location.
    
    Preconditions:
        - Robot must not be holding anything
        - Object must exist in the world
        - Object must be at the robot's current location
    
    Args:
        state: Current world state
        obj: Name of the object to pick up
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if object was picked up
        - message: Description of the result
    """
    obj = obj.lower()  # Case-insensitive
    new_state = state.clone()
    if new_state.robot.holding is not None:
        return state, False, "Already holding something"

    obj_state = new_state.objects.get(obj)
    if obj_state is None:
        return state, False, f"Unknown object: {obj}"

    if obj_state.location != new_state.robot.location:
        return state, False, f"{obj} not at robot location"

    obj_state.location = "robot_hand"
    new_state.robot.holding = obj
    new_state.step_count += 1
    return new_state, True, f"Picked up {obj}"

def place(state: WorldState, location: str) -> ActionResult:
    """Place the currently held object at a SPECIFIED location.
    
    Preconditions:
        - Robot must be holding an object
        - Target location must be valid
    
    Args:
        state: Current world state
        location: Target location to place the object
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if object was placed
        - message: Description of the result
    """
    location = location.lower()  # Case-insensitive
    new_state = state.clone()
    if new_state.robot.holding is None:
        return state, False, "Not holding anything to place"

    if location not in VALID_LOCATIONS:
        return state, False, f"Invalid location: {location}"

    obj = new_state.robot.holding
    new_state.objects[obj].location = location
    new_state.robot.holding = None
    new_state.step_count += 1
    return new_state, True, f"Placed {obj} on {location}"

def put_down(state: WorldState) -> ActionResult:
    """Place the currently held object at the robot's CURRENT location.
    
    Simpler alternative to place() - drops object at robot's position.
    
    Preconditions:
        - Robot must be holding an object
    
    Args:
        state: Current world state
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if object was put down
        - message: Description of the result
    """
    new_state = state.clone()
    if new_state.robot.holding is None:
        return state, False, "Not holding anything to put down"

    obj = new_state.robot.holding
    new_state.objects[obj].location = new_state.robot.location
    new_state.robot.holding = None
    new_state.step_count += 1
    return new_state, True, f"Put down {obj} at {new_state.robot.location}"

def inspect(state: WorldState, obj: str) -> ActionResult:
    """Inspect an object to view its properties and location.
    
    This is a non-destructive action that provides information without
    changing the world state (except step count).
    
    Args:
        state: Current world state
        obj: Name of the object to inspect
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state with incremented step count
        - success: True if object exists
        - message: Description of object properties and location
    """
    obj = obj.lower()  # Case-insensitive
    new_state = state.clone()
    obj_state = new_state.objects.get(obj)
    
    if obj_state is None:
        return state, False, f"Unknown object: {obj}"
    
    new_state.step_count += 1
    
    # Build description
    props_str = ", ".join([f"{k}={v}" for k, v in obj_state.properties.items()])
    if props_str:
        message = f"Inspected {obj}: at {obj_state.location}, properties: {props_str}"
    else:
        message = f"Inspected {obj}: at {obj_state.location}, no special properties"
    
    return new_state, True, message

def open_container(state: WorldState, obj: str) -> ActionResult:
    """Open a container object (e.g., box, drawer, cabinet).
    
    Preconditions:
        - Object must exist and be a container (have 'is_container' property)
        - Object must be at robot's current location
        - Container must not already be open
    
    Args:
        state: Current world state
        obj: Name of the container to open
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if container was opened
        - message: Description of the result
    """
    obj = obj.lower()  # Case-insensitive
    new_state = state.clone()
    obj_state = new_state.objects.get(obj)
    
    if obj_state is None:
        return state, False, f"Unknown object: {obj}"
    
    if not obj_state.properties.get("is_container", False):
        return state, False, f"{obj} is not a container"
    
    if obj_state.location != new_state.robot.location:
        return state, False, f"{obj} not at robot location"
    
    if obj_state.properties.get("is_open", False):
        return state, False, f"{obj} is already open"
    
    obj_state.properties["is_open"] = True
    new_state.step_count += 1
    return new_state, True, f"Opened {obj}"

def close_container(state: WorldState, obj: str) -> ActionResult:
    """Close a container object (e.g., box, drawer, cabinet).
    
    Preconditions:
        - Object must exist and be a container (have 'is_container' property)
        - Object must be at robot's current location
        - Container must be open
    
    Args:
        state: Current world state
        obj: Name of the container to close
        
    Returns:
        Tuple of (new_state, success, message)
        - new_state: Updated world state (unchanged if action failed)
        - success: True if container was closed
        - message: Description of the result
    """
    obj = obj.lower()  # Case-insensitive
    new_state = state.clone()
    obj_state = new_state.objects.get(obj)
    
    if obj_state is None:
        return state, False, f"Unknown object: {obj}"
    
    if not obj_state.properties.get("is_container", False):
        return state, False, f"{obj} is not a container"
    
    if obj_state.location != new_state.robot.location:
        return state, False, f"{obj} not at robot location"
    
    if not obj_state.properties.get("is_open", False):
        return state, False, f"{obj} is already closed"
    
    obj_state.properties["is_open"] = False
    new_state.step_count += 1
    return new_state, True, f"Closed {obj}"

def list_valid_actions() -> List[str]:
    """Get a list of all available action types.
    
    Returns:
        List of action signatures for use in prompts or UI
    """
    return [
        'MOVE_TO(location)',
        'PICK(object)',
        'PLACE(location)',
        'PUT_DOWN()',
        'INSPECT(object)',
        'OPEN(container)',
        'CLOSE(container)',
    ]
