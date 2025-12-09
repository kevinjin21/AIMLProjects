"""World state representation for symbolic robot planning."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple


@dataclass
class ObjectState:
    """Represents the state of an object in the environment.
    
    Attributes:
        name: Unique identifier for the object
        location: Current location (e.g., "table", "shelf", "robot_hand")
        properties: Optional properties (e.g., {"fragile": True, "heavy": False})
    """
    name: str
    location: str
    properties: Dict[str, bool] = field(default_factory=dict)

@dataclass
class RobotState:
    """Represents the robot's current state.
    
    Attributes:
        location: Current location of the robot
        holding: Name of object being held, None if hand is empty
    """
    location: str
    holding: Optional[str] = None

@dataclass
class Goal:
    """Represents a goal condition for task completion.
    
    Attributes:
        type: Goal type ("object_at_location", "robot_at_location", "object_property")
        params: Parameters specific to the goal type
            - object_at_location: {"object": str, "location": str}
            - robot_at_location: {"location": str}
            - object_property: {"object": str, "property": str, "value": str}
    """
    type: str
    params: Dict[str, str]

@dataclass
class WorldState:
    """Complete state of the world including robot, objects, and goals.
    
    Attributes:
        robot: Current robot state
        objects: Dictionary mapping object names to their states
        task_description: Natural language description of the task
        goals: List of goal conditions to be satisfied
        step_count: Number of actions taken so far
        max_steps: Maximum allowed steps before timeout
        action_history: Chronological list of actions taken
    """
    robot: RobotState
    objects: Dict[str, ObjectState]
    task_description: str
    goals: List[Goal] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 10
    action_history: List[str] = field(default_factory=list)
    
    def record_action(self, action: str) -> None:
        """Record an action in the history.
        
        Args:
            action: String description of the action taken
        """
        self.action_history.append(action)

    def is_task_completed(self) -> bool:
        """Check if all goals have been satisfied.
        
        Returns:
            True if all goals are met, False otherwise
        """
        for goal in self.goals:
            if goal.type == "object_at_location":
                obj = self.objects.get(goal.params["object"])
                if not obj or obj.location != goal.params["location"]:
                    return False
            elif goal.type == "robot_at_location":
                if self.robot.location != goal.params["location"]:
                    return False
            elif goal.type == "object_property":
                obj = self.objects.get(goal.params["object"])
                if not obj:
                    return False
                property_name = goal.params["property"]
                expected_value = goal.params["value"] == "true"  # Convert string to bool
                actual_value = obj.properties.get(property_name, False)
                if actual_value != expected_value:
                    return False
        return True

    def clone(self) -> "WorldState":
        """Create a deep copy of the current world state.
        
        Returns:
            Independent copy of this WorldState
        """
        return deepcopy(self)
    
    def is_valid(self) -> Tuple[bool, str]:
        """Validate the consistency of the world state.
        
        Returns:
            Tuple of (is_valid, message) where is_valid is True if state is valid,
            and message describes any validation errors
        """
        in_hand = [obj for obj in self.objects.values() if obj.location == "robot_hand"]
        if len(in_hand) > 1:
            return False, "Multiple objects in hand"
        if self.robot.holding and self.robot.holding not in self.objects:
            return False, f"Holding unknown object: {self.robot.holding}"
        return True, "Valid"
    
    def to_natural_language(self) -> str:
        """Convert the world state to a natural language description.
        
        Returns:
            Human-readable string describing the current state and goals
        """
        lines = [f"Robot is at {self.robot.location}."]
        if self.robot.holding:
            lines.append(f"Robot is holding {self.robot.holding}.")
        else:
            lines.append("Robot's hand is empty.")
        
        for obj in self.objects.values():
            if obj.location != "robot_hand":
                lines.append(f"{obj.name} is on {obj.location}.")
        
        lines.append(f"Goal: {self.task_description}")
        lines.append(f"Steps taken: {self.step_count}/{self.max_steps}")
        return "\n".join(lines)
    
    def can_pick(self, obj: str) -> Tuple[bool, str]:
        """Check if the robot can pick up the specified object.
        
        Args:
            obj: Name of the object to pick up
            
        Returns:
            Tuple of (can_pick, reason) where can_pick is True if action is valid,
            and reason explains why or why not
        """
        if self.robot.holding:
            return False, "Already holding something"
        if obj not in self.objects:
            return False, f"Unknown object: {obj}"
        if self.objects[obj].location != self.robot.location:
            return False, "Object not at robot location"
        return True, "Can pick"
