#!/usr/bin/env python3
"""
Skill Library - High-level behavior decomposition
==================================================

"고수준 행동을 저수준 액션으로"

Maps natural language instructions to executable action sequences

Architecture:
    LLM text → Skill Parser → Skill → Low-level actions

Skills:
- move_to_target(target)
- interact_with_object(object)
- craft_item(item)
- navigate_to_location(location)
- use_tool(tool, target)
"""

import re
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ============================================================================
# Skill Types
# ============================================================================

class SkillType(Enum):
    """Basic skill types"""
    MOVE = "move"                # Movement/navigation
    INTERACT = "interact"        # Interact with object
    CRAFT = "craft"              # Craft/build item
    USE_TOOL = "use_tool"        # Use tool on target
    WAIT = "wait"                # Wait for duration
    OBSERVE = "observe"          # Observe environment
    CUSTOM = "custom"            # Custom skill


@dataclass
class SkillExecution:
    """Result of skill execution"""
    skill_name: str
    success: bool
    duration: float
    actions_taken: int
    error_message: Optional[str] = None


# ============================================================================
# Base Skill
# ============================================================================

class BaseSkill:
    """
    Base class for all skills

    A skill is a high-level behavior that:
    1. Takes parameters (e.g., target object)
    2. Executes a sequence of low-level actions
    3. Returns success/failure
    """

    def __init__(self, name: str, skill_type: SkillType):
        self.name = name
        self.skill_type = skill_type

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Execute skill

        Args:
            agent: Embodied agent (has env, policy, etc.)
            params: Skill parameters

        Returns:
            Execution result
        """
        raise NotImplementedError

    def estimate_duration(self, params: Dict) -> float:
        """Estimate how long this skill will take"""
        return 1.0  # Default 1 second


# ============================================================================
# Movement Skills
# ============================================================================

class MoveToTargetSkill(BaseSkill):
    """Move to target location or object"""

    def __init__(self):
        super().__init__("move_to_target", SkillType.MOVE)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Move to target

        Params:
            target: Target object name or coordinates
            precision: How close to get (default 1.0)
        """
        start_time = time.time()
        target = params.get('target')
        precision = params.get('precision', 1.0)

        print(f"[Skill:{self.name}] Moving to {target}...")

        # Use agent's low-level policy to move
        actions_taken = 0

        # Simple movement loop (can be replaced with proper path planning)
        for _ in range(10):  # Max 10 steps
            # Get current observation
            obs = agent.env.get_observation()

            # TODO: Proper path planning
            # For now, just execute forward movement
            from env_adapter import Action
            action = Action(keys=['w'])  # Move forward

            # Execute via agent's controller
            agent.env.step(action)
            actions_taken += 1

            time.sleep(0.1)

        duration = time.time() - start_time

        return SkillExecution(
            skill_name=self.name,
            success=True,  # TODO: Check if actually reached target
            duration=duration,
            actions_taken=actions_taken
        )


class NavigateToLocationSkill(BaseSkill):
    """Navigate to GPS/map coordinates"""

    def __init__(self):
        super().__init__("navigate_to_location", SkillType.MOVE)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Navigate to location

        Params:
            x, y, z: Coordinates
            or
            location_name: Named location
        """
        start_time = time.time()

        x = params.get('x')
        y = params.get('y')
        z = params.get('z')
        location_name = params.get('location_name')

        print(f"[Skill:{self.name}] Navigating to {location_name or (x,y,z)}...")

        # TODO: Implement proper navigation with path planning
        actions_taken = 0
        duration = time.time() - start_time

        return SkillExecution(
            skill_name=self.name,
            success=True,
            duration=duration,
            actions_taken=actions_taken
        )


# ============================================================================
# Interaction Skills
# ============================================================================

class InteractWithObjectSkill(BaseSkill):
    """Interact with object (open, use, pickup, etc.)"""

    def __init__(self):
        super().__init__("interact_with_object", SkillType.INTERACT)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Interact with object

        Params:
            object: Object name
            action: 'use', 'open', 'pickup', 'talk', etc.
        """
        start_time = time.time()

        obj = params.get('object')
        action_type = params.get('action', 'use')

        print(f"[Skill:{self.name}] {action_type} {obj}...")

        # Execute interaction
        from env_adapter import Action

        # Common interaction key
        action = Action(keys=['e'])  # 'E' for interact in many games

        agent.env.step(action)
        time.sleep(0.2)

        actions_taken = 1
        duration = time.time() - start_time

        return SkillExecution(
            skill_name=self.name,
            success=True,
            duration=duration,
            actions_taken=actions_taken
        )


class CraftItemSkill(BaseSkill):
    """Craft/build item"""

    def __init__(self):
        super().__init__("craft_item", SkillType.CRAFT)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Craft item

        Params:
            item: Item to craft
            quantity: How many (default 1)
        """
        start_time = time.time()

        item = params.get('item')
        quantity = params.get('quantity', 1)

        print(f"[Skill:{self.name}] Crafting {quantity}x {item}...")

        # Open crafting menu, select item, confirm
        # TODO: Implement proper crafting logic
        actions_taken = 3

        duration = time.time() - start_time

        return SkillExecution(
            skill_name=self.name,
            success=True,
            duration=duration,
            actions_taken=actions_taken
        )


# ============================================================================
# Utility Skills
# ============================================================================

class WaitSkill(BaseSkill):
    """Wait for duration"""

    def __init__(self):
        super().__init__("wait", SkillType.WAIT)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Wait

        Params:
            duration: How long to wait (seconds)
        """
        start_time = time.time()
        duration = params.get('duration', 1.0)

        print(f"[Skill:{self.name}] Waiting {duration}s...")
        time.sleep(duration)

        return SkillExecution(
            skill_name=self.name,
            success=True,
            duration=duration,
            actions_taken=0
        )


class ObserveSkill(BaseSkill):
    """Observe environment carefully"""

    def __init__(self):
        super().__init__("observe", SkillType.OBSERVE)

    def execute(self, agent, params: Dict) -> SkillExecution:
        """
        Observe environment

        Params:
            focus: What to focus on (optional)
            duration: How long to observe
        """
        start_time = time.time()

        focus = params.get('focus')
        duration = params.get('duration', 1.0)

        print(f"[Skill:{self.name}] Observing {focus or 'environment'}...")

        # Capture observation
        obs = agent.env.get_observation()

        # Store in memory
        if hasattr(agent, 'memory'):
            agent.memory.store_observation(obs, focus)

        time.sleep(duration)

        return SkillExecution(
            skill_name=self.name,
            success=True,
            duration=duration,
            actions_taken=1
        )


# ============================================================================
# Skill Library
# ============================================================================

class SkillLibrary:
    """
    Library of available skills

    Handles:
    - Skill registration
    - Natural language → Skill parsing
    - Skill execution
    """

    def __init__(self):
        # Built-in skills
        self.skills: Dict[str, BaseSkill] = {}

        # Register default skills
        self._register_default_skills()

        print(f"[SkillLibrary] Loaded {len(self.skills)} skills")

    def _register_default_skills(self):
        """Register default skills"""
        skills = [
            MoveToTargetSkill(),
            NavigateToLocationSkill(),
            InteractWithObjectSkill(),
            CraftItemSkill(),
            WaitSkill(),
            ObserveSkill(),
        ]

        for skill in skills:
            self.register_skill(skill)

    def register_skill(self, skill: BaseSkill):
        """Register new skill"""
        self.skills[skill.name] = skill
        print(f"[SkillLibrary] Registered: {skill.name}")

    def parse_instruction(self, instruction: str) -> Optional[tuple]:
        """
        Parse natural language instruction into skill + params

        Examples:
            "move to the workbench" → (move_to_target, {target: "workbench"})
            "craft an axe" → (craft_item, {item: "axe"})
            "wait for 2 seconds" → (wait, {duration: 2.0})

        Returns:
            (skill_name, params) or None if not parseable
        """
        instruction = instruction.lower().strip()

        # Movement patterns
        if re.search(r'(move|go|walk|run) (to|towards)', instruction):
            # Extract target
            match = re.search(r'(to|towards)\s+(?:the\s+)?(\w+)', instruction)
            if match:
                target = match.group(2)
                return ('move_to_target', {'target': target})

        # Navigation patterns
        if re.search(r'navigate to|go to coordinates', instruction):
            # Try to extract coordinates
            match = re.search(r'(\d+),\s*(\d+)', instruction)
            if match:
                return ('navigate_to_location', {
                    'x': float(match.group(1)),
                    'y': float(match.group(2))
                })

        # Interaction patterns
        if re.search(r'(use|open|interact|talk|pickup)', instruction):
            match = re.search(r'(use|open|interact|talk|pickup)\s+(?:the\s+)?(\w+)', instruction)
            if match:
                action = match.group(1)
                obj = match.group(2)
                return ('interact_with_object', {'object': obj, 'action': action})

        # Crafting patterns
        if re.search(r'craft|make|build|create', instruction):
            match = re.search(r'(craft|make|build|create)\s+(?:an?\s+)?(\w+)', instruction)
            if match:
                item = match.group(2)
                return ('craft_item', {'item': item})

        # Wait patterns
        if re.search(r'wait', instruction):
            match = re.search(r'wait\s+(?:for\s+)?([0-9.]+)', instruction)
            duration = float(match.group(1)) if match else 1.0
            return ('wait', {'duration': duration})

        # Observe patterns
        if re.search(r'look|observe|watch|examine', instruction):
            match = re.search(r'(look|observe|watch|examine)\s+(?:at\s+)?(?:the\s+)?(\w+)', instruction)
            focus = match.group(2) if match else None
            return ('observe', {'focus': focus})

        # Not recognized
        return None

    def execute_skill(self, agent, skill_name: str, params: Dict) -> SkillExecution:
        """Execute skill by name"""
        if skill_name not in self.skills:
            return SkillExecution(
                skill_name=skill_name,
                success=False,
                duration=0.0,
                actions_taken=0,
                error_message=f"Unknown skill: {skill_name}"
            )

        skill = self.skills[skill_name]
        return skill.execute(agent, params)

    def execute_instruction(self, agent, instruction: str) -> SkillExecution:
        """Parse and execute natural language instruction"""
        parsed = self.parse_instruction(instruction)

        if parsed is None:
            return SkillExecution(
                skill_name="unknown",
                success=False,
                duration=0.0,
                actions_taken=0,
                error_message=f"Could not parse: {instruction}"
            )

        skill_name, params = parsed
        print(f"[SkillLibrary] '{instruction}' → {skill_name}({params})")

        return self.execute_skill(agent, skill_name, params)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate skill library"""
    print("\n" + "="*70)
    print("SKILL LIBRARY - Demo")
    print("="*70)

    library = SkillLibrary()

    # Test parsing
    instructions = [
        "move to the workbench",
        "craft an axe",
        "use the door",
        "wait for 2 seconds",
        "observe the enemy",
    ]

    print("\n[Demo] Testing instruction parsing:")
    for inst in instructions:
        parsed = library.parse_instruction(inst)
        print(f"  '{inst}'")
        print(f"    → {parsed}")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    demo()
