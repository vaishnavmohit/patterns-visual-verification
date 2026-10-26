"""Convert Bongard-LOGO action programs into human-readable drawing stories.

An "action program" is a list of normalized strings like
``line_zigzag_0.539-0.500`` or ``arc_normal_0.625_0.373-0.500``. This module
parses each string, denormalizes its numeric fields, and produces a numbered
step-by-step description that an LLM can reason over without seeing images.
"""

from __future__ import annotations

from typing import Any


BASE_ACTIONS = ["<PAD>", "start", "line", "arc", "and", "stop"]
BASE_LINE_TYPES = ["normal", "zigzag", "circle", "triangle", "square"]


def denormalize_turn_angle(normalized: float) -> tuple[str, float]:
    """Map normalized [0,1] turn to ('L'|'R', degrees in [0,180])."""
    normalized = max(0.0, min(1.0, normalized))
    if normalized >= 0.5:
        return "L", normalized * 360 - 180
    return "R", 180 - normalized * 360


def denormalize_arc_angle(normalized: float) -> float:
    """Map normalized [0,1] arc to degrees in [-360, 360]."""
    normalized = max(0.0, min(1.0, normalized))
    return normalized * 720 - 360


def parse_action_string_to_command(action_string: str) -> dict[str, Any]:
    """Parse one action string into a structured command dict."""

    if action_string in ("start", "stop", "and", "<PAD>"):
        return {"action": action_string}

    try:
        parts = action_string.split("_")
        action_name = parts[0]

        # If the slot where 'type' should be is actually numeric, default to 'normal'.
        if len(parts) < 2 or parts[1].replace(".", "", 1).isdigit():
            command: dict[str, Any] = {"action": action_name, "type": "normal"}
            parts.insert(1, "normal")
        else:
            command = {"action": action_name, "type": parts[1]}

        movement_str, norm_turn_str = parts[-1].split("-")
        turn_dir, turn_angle = denormalize_turn_angle(float(norm_turn_str))
        command["turn_direction"] = turn_dir
        command["turn_angle"] = turn_angle

        if action_name == "line":
            command["length"] = float(movement_str)
        elif action_name == "arc":
            command["radius"] = float(parts[2])
            command["arc_angle"] = denormalize_arc_angle(float(movement_str))

        return command
    except (ValueError, IndexError) as e:
        return {"action": "error", "details": f"Failed to parse {action_string!r}: {e}"}


def parse_action_strings(action_strings: list[str]) -> list[dict[str, Any]]:
    return [parse_action_string_to_command(s) for s in action_strings]


def generate_drawing_story(commands: list[dict[str, Any]], figure_index: int = 0) -> str:
    """Render parsed commands as a numbered drawing recipe."""
    if not commands:
        return "This figure is empty."

    parts = [f"To draw figure {figure_index}, follow these steps:"]
    step = 1
    dir_map = {"L": "left", "R": "right", "l": "left", "r": "right"}

    for cmd in commands:
        action = cmd.get("action")
        if action in ("start", "<PAD>", "stop"):
            continue
        if action == "and":
            parts.append("\nThen, start a new, separate shape:")
            step = 1
            continue
        if action == "error":
            parts.append(f"Skipping a malformed command: {cmd.get('details', 'unknown error')}")
            continue

        line = f"Step {step}: "
        step += 1

        action_type = cmd.get("type", "normal")
        if action == "line":
            line += f"draw a {action_type} line of {cmd.get('length', 0):.3f} units."
        elif action == "arc":
            line += (
                f"draw a {action_type} arc with a radius of {cmd.get('radius', 0):.3f} "
                f"and sweeping {cmd.get('arc_angle', 0):.1f} degrees."
            )

        turn_dir = dir_map.get(cmd.get("turn_direction", "L"), "left")
        turn_angle = cmd.get("turn_angle", 0)
        if turn_angle > 0.1:
            line += f" After that, turn {turn_dir} by {turn_angle:.1f} degrees."
        else:
            line += " Then, continue straight without turning."

        parts.append(line)

    parts.append("\nThe figure is now complete.\n\n")
    return "\n".join(parts)


def _stroke_phrase(cmd: dict[str, Any]) -> str:
    action_type = cmd.get("type", "normal")
    if cmd.get("action") == "line":
        return f"draw a {action_type} line of {cmd.get('length', 0):.3f} units."
    return (f"draw a {action_type} arc with a radius of {cmd.get('radius', 0):.3f} "
            f"and sweeping {cmd.get('arc_angle', 0):.1f} degrees.")


def generate_drawing_story_rendered(shapes: list[list[dict[str, Any]]], figure_index: int = 0) -> str:
    """Drawing recipe that follows the renderer's geometry.

    The Bongard-LOGO renderer (NVlabs/Bongard-LOGO, bongard/bongard_painter.py,
    BongardShapePainter.draw) turns the turtle by an action's angle BEFORE it
    draws that action, and draws every shape of a figure separately from its
    own start point. generate_drawing_story attaches each turn after the stroke
    it precedes and, given a flattened program, runs separate shapes together.
    This version states each turn before its stroke and separates shapes. The
    first turn of a shape is relative to an unspecified starting direction.
    """
    dir_map = {"L": "left", "R": "right", "l": "left", "r": "right"}
    parts = [f"To draw figure {figure_index}, follow these steps:"]
    for s_idx, shape in enumerate(shapes):
        if s_idx:
            parts.append("\nThen, start a new, separate shape:")
        for step, cmd in enumerate(shape, start=1):
            if cmd.get("action") == "error":
                parts.append(f"Skipping a malformed command: {cmd.get('details', 'unknown error')}")
                continue
            turn = cmd.get("turn_angle", 0)
            if turn > 0.1:
                lead = f"turn {dir_map.get(cmd.get('turn_direction', 'L'), 'left')} by {turn:.1f} degrees, then"
            else:
                lead = "without turning,"
            parts.append(f"Step {step}: {lead} {_stroke_phrase(cmd)}")
    parts.append("\nThe figure is now complete.\n\n")
    return "\n".join(parts)


_NESTED: dict[tuple[str, ...], list[list[str]]] | None = None


def nested_program(action_list: list[str]) -> list[list[str]]:
    """Shape-nested form of a flattened program, from data/Bongard_Logo_human/nested_programs.json."""
    global _NESTED
    if _NESTED is None:
        import json
        import os
        from pathlib import Path
        path = os.environ.get("HMR_NESTED_PROGRAMS") or str(
            Path(__file__).resolve().parents[3] / "data" / "Bongard_Logo_human" / "nested_programs.json")
        with open(path) as f:
            _NESTED = {tuple(x for shape in fig for x in shape): fig for fig in json.load(f)}
    key = tuple(action_list)
    if key not in _NESTED:
        raise KeyError(f"no nested program for a {len(action_list)}-stroke figure; rebuild nested_programs.json")
    return _NESTED[key]


def process_all_image_specs_rendered(all_image_specs: list[list[str]]) -> list[str]:
    """Rendered-geometry drawing stories (see generate_drawing_story_rendered)."""
    stories = []
    for i, action_list in enumerate(all_image_specs, start=1):
        shapes = [parse_action_strings(shape) for shape in nested_program(action_list)]
        stories.append(generate_drawing_story_rendered(shapes, i))
    return stories


def process_all_image_specs(all_image_specs: list[list[str]]) -> list[str]:
    """Convert each image's action list into a drawing-story string."""
    stories = []
    for i, action_list in enumerate(all_image_specs, start=1):
        if not isinstance(action_list, list):
            stories.append("Invalid image description: not a list.")
            continue
        try:
            commands = parse_action_strings(action_list)
            stories.append(generate_drawing_story(commands, i))
        except Exception as e:
            stories.append(f"An unexpected error occurred while processing an image description: {e}")
    return stories
