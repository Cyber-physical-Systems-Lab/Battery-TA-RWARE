import json
import re
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Any, Dict, List, Tuple
from urllib import error, request

import gymnasium as gym
import numpy as np

from tarware.heuristic import heuristic_episode
from tarware.warehouse import AgentType


parser = ArgumentParser(
    description="Run plain-language LLM control on Battery-TA-RWARE",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--env_id", default="tarware-tiny-1agvs-1pickers-partialobs-chg-v1", type=str, help="Gym environment id")
parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes")
parser.add_argument("--seed", default=0, type=int, help="Base seed")
parser.add_argument("--model", default="llama3.2:3b", type=str, help="Ollama model name")
parser.add_argument("--ollama_url", default="http://localhost:11434/api/generate", type=str, help="Ollama generate endpoint")
parser.add_argument("--request_timeout_s", default=45, type=int, help="HTTP timeout")
parser.add_argument("--max_steps_per_episode", default=400, type=int, help="Safety cap for episode length")
parser.add_argument("--decision_interval", default=1, type=int, help="Query LLM every N steps")
parser.add_argument(
    "--force_llm_replan_steps",
    default=20,
    type=int,
    help="Force an LLM replan every N steps even if no event requires replanning",
)
parser.add_argument("--progress_every", default=10, type=int, help="Print progress every N steps")
parser.add_argument(
    "--max_candidate_ids",
    default=0,
    type=int,
    help="Max candidate action ids per agent in prompt (0 means all valid actions)",
)
parser.add_argument(
    "--log_text_chars",
    default=5000,
    type=int,
    help="Max chars for PROMPT/LLM_OUTPUT logs (0 = no truncation)",
)
parser.add_argument("--render", action="store_true")
parser.add_argument("--skip_heuristic", action="store_true", help="Only run LLM policy")
parser.add_argument(
    "--warehouse_event_logs",
    action="store_true",
    help="Enable warehouse-level [COLLAB]/[DELIVERY] prints from the environment",
)


def log_block(tag: str, text: str) -> None:
    print(f"\n[{tag}]\n{text}\n")

def maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return text[:max_chars] + f"\n...[TRUNCATED {remaining} chars]"


def info_statistics(infos: List[Dict[str, Any]], global_episode_return: float, episode_returns: np.ndarray) -> Dict[str, Any]:
    total_deliveries = 0
    total_clashes = 0
    total_stuck = 0
    total_agvs_distance = 0
    total_pickers_distance = 0
    for info in infos:
        total_deliveries += info.get("shelf_deliveries", 0)
        total_clashes += info.get("clashes", 0)
        total_stuck += info.get("stucks", 0)
        total_agvs_distance += info.get("agvs_distance_travelled", 0)
        total_pickers_distance += info.get("pickers_distance_travelled", 0)
        info["total_deliveries"] = total_deliveries
        info["total_clashes"] = total_clashes
        info["total_stuck"] = total_stuck
        info["total_agvs_distance"] = total_agvs_distance
        info["total_pickers_distance"] = total_pickers_distance

    last_info = infos[-1] if infos else {}
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = float(global_episode_return)
    last_info["episode_returns"] = episode_returns
    if infos and len(infos) > 0:
        last_info["overall_pick_rate"] = last_info["total_deliveries"] * 3600 / (5 * len(infos))
    else:
        last_info["overall_pick_rate"] = 0.0
    return last_info


def safe_valid_action_masks(env) -> np.ndarray:
    try:
        masks = np.asarray(env.compute_valid_action_masks(), dtype=np.float64)
        if masks.shape != (env.num_agents, env.action_size):
            raise ValueError(f"mask shape={masks.shape} expected={(env.num_agents, env.action_size)}")
        return masks
    except Exception:
        return np.ones((env.num_agents, env.action_size), dtype=np.float64)


def classify_action(env, action_id: int) -> str:
    if action_id == 0:
        return "NOOP"
    if 1 <= action_id <= len(env.goals):
        return "GOAL"
    charging_start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    if charging_start <= action_id <= len(env.action_id_to_coords_map):
        return "CHARGING"
    return "SHELF"


def nearest_id_by_path(env, agent, candidate_ids: List[int]) -> int:
    best_id = 0
    best_len = float("inf")
    for action_id in candidate_ids:
        if action_id not in env.action_id_to_coords_map:
            continue
        target = env.action_id_to_coords_map[action_id]
        path = env.find_path((agent.y, agent.x), target, agent, care_for_agents=False)
        if path and len(path) < best_len:
            best_len = len(path)
            best_id = int(action_id)
    return best_id


def get_requested_action_ids(env) -> List[int]:
    inv = {v: k for k, v in env.action_id_to_coords_map.items()}
    ids: List[int] = []
    for shelf in env.request_queue:
        aid = inv.get((int(shelf.y), int(shelf.x)))
        if aid is not None:
            ids.append(int(aid))
    return list(dict.fromkeys(ids))


def charging_action_ids(env) -> List[int]:
    start = len(env.action_id_to_coords_map) - len(env.charging_stations) + 1
    return list(range(start, len(env.action_id_to_coords_map) + 1))


def goal_action_ids(env) -> List[int]:
    return list(range(1, len(env.goals) + 1))


def empty_shelf_action_ids(env) -> List[int]:
    empty_mask = env.get_empty_shelf_information()
    return list(np.array(env.shelf_action_ids)[empty_mask > 0])


def fallback_actions(env, valid_masks: np.ndarray) -> List[int]:
    actions = [0] * env.num_agents
    requested_ids = get_requested_action_ids(env)
    goal_ids = goal_action_ids(env)
    charging_ids = charging_action_ids(env)

    for i, agent in enumerate(env.agents):
        if agent.battery < 25:
            actions[i] = nearest_id_by_path(env, agent, charging_ids)
            continue

        if agent.type == AgentType.AGV:
            if agent.carrying_shelf is not None:
                requested_shelf_ids = {int(s.id) for s in env.request_queue}
                if int(agent.carrying_shelf.id) in requested_shelf_ids:
                    actions[i] = nearest_id_by_path(env, agent, goal_ids)
                else:
                    empties = empty_shelf_action_ids(env)
                    actions[i] = nearest_id_by_path(env, agent, empties)
            else:
                actions[i] = nearest_id_by_path(env, agent, requested_ids)
        else:
            agv_targets = [int(a.target) for a in env.agents[: env.num_agvs] if int(a.target) > 0]
            candidates = agv_targets if agv_targets else requested_ids
            actions[i] = nearest_id_by_path(env, agent, candidates)

    for i in range(env.num_agents):
        if actions[i] < 0 or actions[i] >= env.action_size or valid_masks[i, actions[i]] <= 0:
            actions[i] = 0
    return actions


def candidate_ids_for_agent(env, idx: int, valid_masks: np.ndarray, max_candidate_ids: int) -> List[int]:
    valid_ids = sorted(int(i) for i in np.where(valid_masks[idx] > 0)[0].tolist())
    if not valid_ids:
        valid_ids = [0]
    if max_candidate_ids and max_candidate_ids > 0:
        return valid_ids[:max_candidate_ids]
    return valid_ids


def battery_need_label(battery: float) -> str:
    if battery < 25.0:
        return "critical"
    if battery < 50.0:
        return "need_charging_soon"
    return "not_needed"


def describe_action_id(env, action_id: int) -> str:
    if int(action_id) == 0:
        return "0:NOOP"
    action_type = classify_action(env, int(action_id))
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return f"{int(action_id)}:{action_type}"
    return f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])})"


def describe_action_id_for_agent(env, agent, action_id: int) -> str:
    if int(action_id) == 0:
        return "0:NOOP (distance_steps=0)"
    action_type = classify_action(env, int(action_id))
    coords = env.action_id_to_coords_map.get(int(action_id))
    if coords is None:
        return f"{int(action_id)}:{action_type}"
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    distance = len(path) if path else 0
    return f"{int(action_id)}:{action_type}@({int(coords[1])},{int(coords[0])}) (distance_steps={distance})"


def get_requested_shelves(env) -> List[str]:
    requested = []
    inv = {v: k for k, v in env.action_id_to_coords_map.items()}
    for shelf in env.request_queue:
        aid = inv.get((int(shelf.y), int(shelf.x)))
        if aid is None:
            requested.append(
                f"Requested shelf {int(shelf.id)} is at position ({int(shelf.x)},{int(shelf.y)}), but no action id was found."
            )
        else:
            requested.append(
                f"Requested shelf {int(shelf.id)} is at position ({int(shelf.x)},{int(shelf.y)}). "
                f"To go there, choose action id {int(aid)}."
            )
    return requested


def nearest_charging_station_for_agent(env, agent) -> Tuple[int, int]:
    charging_ids = charging_action_ids(env)
    if not charging_ids:
        return 0, 0
    nearest_id = nearest_id_by_path(env, agent, charging_ids)
    coords = env.action_id_to_coords_map.get(int(nearest_id))
    if nearest_id <= 0 or coords is None:
        return 0, 0
    path = env.find_path((agent.y, agent.x), coords, agent, care_for_agents=False)
    return int(nearest_id), len(path) if path else 0


def charging_station_occupancy(env) -> List[str]:
    lines: List[str] = []
    charging_ids = charging_action_ids(env)
    for cid in charging_ids:
        coords = env.action_id_to_coords_map.get(int(cid))
        if coords is None:
            continue
        x, y = int(coords[1]), int(coords[0])
        occupants: List[str] = []
        for i, agent in enumerate(env.agents):
            if (int(agent.x), int(agent.y)) == (x, y):
                occupants.append(
                    f"agent_{i}({agent.type.name},battery={float(agent.battery):.1f},need={battery_need_label(float(agent.battery))})"
                )
        lines.append(f"charging_action={cid} at ({x},{y}) occupants=[{', '.join(occupants) if occupants else 'none'}]")
    return lines


def charging_now_agents(env) -> List[str]:
    station_cells = {(int(s.x), int(s.y)) for s in env.charging_stations}
    out: List[str] = []
    for i, agent in enumerate(env.agents):
        if (int(agent.x), int(agent.y)) in station_cells:
            out.append(f"agent_{i}({agent.type.name},battery={float(agent.battery):.1f})")
    return out


def assign_picker_support_pairs(env) -> Dict[int, int]:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    support_agvs: List[int] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or not at_target or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        if (int(agent.x), int(agent.y)) in picker_positions:
            continue
        support_agvs.append(i)

    picker_indices = list(range(env.num_agvs, env.num_agents))
    if not picker_indices or not support_agvs:
        return {}

    remaining_pickers = set(picker_indices)
    pairs: Dict[int, int] = {}
    for agv_idx in support_agvs:
        agv = env.agents[agv_idx]
        target = env.action_id_to_coords_map.get(int(agv.target))
        if target is None:
            continue
        best_picker = None
        best_len = float("inf")
        for p_idx in remaining_pickers:
            picker = env.agents[p_idx]
            path = env.find_path((picker.y, picker.x), target, picker, care_for_agents=False)
            if path and len(path) < best_len:
                best_len = len(path)
                best_picker = p_idx
        if best_picker is not None:
            pairs[agv_idx] = int(best_picker)
            remaining_pickers.remove(best_picker)
    return pairs


def _ranked_picker_distances_to_target(env, target_coords: Tuple[int, int]) -> List[Tuple[int, int]]:
    ranked: List[Tuple[int, int]] = []
    for p_idx in range(env.num_agvs, env.num_agents):
        picker = env.agents[p_idx]
        path = env.find_path((picker.y, picker.x), target_coords, picker, care_for_agents=False)
        if path:
            ranked.append((int(p_idx), int(len(path))))
    ranked.sort(key=lambda x: x[1])
    return ranked


def waiting_agv_support_lines(env, max_picker_suggestions: int = 3) -> List[str]:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    lines: List[str] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or not at_target or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        picker_here = (int(agent.x), int(agent.y)) in picker_positions
        if picker_here:
            continue
        ranked = _ranked_picker_distances_to_target(env, coords)
        if ranked:
            recommended_picker, recommended_steps = ranked[0]
            top = ranked[: max(1, max_picker_suggestions)]
            top_text = ", ".join([f"agent_{p}(steps={d})" for p, d in top])
            lines.append(
                f"AGV agent_{i} is waiting for Picker support at shelf action id {target_id} "
                f"position ({int(coords[1])},{int(coords[0])}). "
                f"Recommended picker is agent_{recommended_picker} (shortest path {recommended_steps} steps). "
                f"Closest pickers: {top_text}."
            )
            continue
        lines.append(
            f"AGV agent_{i} is waiting for Picker support at shelf action id {target_id} "
            f"position ({int(coords[1])},{int(coords[0])})."
        )
    return lines


def picker_support_priority_line(env, picker_idx: int) -> str:
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    parts: List[str] = []
    for i, agent in enumerate(env.agents[: env.num_agvs]):
        target_id = int(agent.target)
        coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
        at_target = bool(coords is not None and (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0])))
        if not bool(agent.busy) or not at_target or target_id <= 0:
            continue
        if classify_action(env, target_id) != "SHELF":
            continue
        if (int(agent.x), int(agent.y)) in picker_positions:
            continue
        ranked = _ranked_picker_distances_to_target(env, coords)
        order = [p for p, _ in ranked]
        if picker_idx in order:
            rank = order.index(picker_idx) + 1
            distance = dict(ranked)[picker_idx]
            parts.append(
                f"For waiting AGV agent_{i} at action {target_id}, your picker rank is {rank}/{len(order)} (distance {distance} steps)."
            )
    if not parts:
        return "You are not currently ranked for any waiting AGV support."
    return " ".join(parts)


def build_agent_prompt(env, agent_idx: int, valid_masks: np.ndarray, step_count: int, max_candidate_ids: int) -> str:
    requested = get_requested_shelves(env)
    picker_pairs = assign_picker_support_pairs(env)
    picker_to_agv = {p: a for a, p in picker_pairs.items()}
    charge_now = charging_now_agents(env)
    charge_occupancy = charging_station_occupancy(env)
    agent = env.agents[agent_idx]
    candidate_ids = candidate_ids_for_agent(env, agent_idx, valid_masks, max_candidate_ids)
    candidate_text = [describe_action_id_for_agent(env, agent, c) for c in candidate_ids]

    target_id = int(agent.target)
    target_coords = env.action_id_to_coords_map.get(target_id) if target_id > 0 else None
    target_text = "none" if target_coords is None else f"{target_id}:{classify_action(env, target_id)}@({int(target_coords[1])},{int(target_coords[0])})"
    is_moving = bool(agent.busy and target_id > 0)
    nearest_charge_id, nearest_charge_steps = nearest_charging_station_for_agent(env, agent)
    carrying = bool(agent.carrying_shelf is not None)
    carrying_text = f"carrying shelf_id={int(agent.carrying_shelf.id)}" if carrying else "not carrying a shelf"
    battery_need = battery_need_label(float(agent.battery))

    meet_text = "no one"
    if agent.type == AgentType.AGV:
        assigned_picker = picker_pairs.get(agent_idx)
        if assigned_picker is not None:
            meet_text = f"picker agent_{assigned_picker}"
    else:
        assigned_agv = picker_to_agv.get(agent_idx)
        if assigned_agv is not None:
            meet_text = f"agv agent_{assigned_agv}"

    all_agents_overview: List[str] = []
    for i, a in enumerate(env.agents):
        t_id = int(a.target)
        t_coords = env.action_id_to_coords_map.get(t_id) if t_id > 0 else None
        t_str = "none" if t_coords is None else f"{t_id}@({int(t_coords[1])},{int(t_coords[0])})/{classify_action(env, t_id)}"
        all_agents_overview.append(
            f"agent_{i} type={a.type.name} pos=({int(a.x)},{int(a.y)}) target={t_str} "
            f"busy={bool(a.busy)} battery={float(a.battery):.1f} need={battery_need_label(float(a.battery))} "
            f"carrying={int(a.carrying_shelf.id) if a.carrying_shelf else None}"
        )

    pair_lines = [f"AGV agent_{a} -> Picker agent_{p}" for a, p in picker_pairs.items()]
    if not pair_lines:
        pair_lines = ["none"]
    waiting_support = waiting_agv_support_lines(env)
    if not waiting_support:
        waiting_support = ["none"]
    picker_priority_note = (
        picker_support_priority_line(env, agent_idx)
        if agent.type == AgentType.PICKER
        else "Not a picker agent."
    )
    goal_id_span = f"1..{len(env.goals)}" if len(env.goals) > 0 else "none"

    return (
        "You are planning ONE warehouse robot action for this time step.\n"
        "Core context:\n"
        "- Multi-agent warehouse with AGV and Picker collaboration.\n"
        "- Objective: maximize pick/delivery throughput while keeping batteries safe.\n"
        "- Avoid any out-of-charge scenario.\n"
        "- Use charging strategically so throughput does not collapse.\n"
        "- AGV-Picker coordinate at shelf interactions (load/unload at shelf location).\n"
        "\n"
        "Typical flow:\n"
        "- AGV without load moves to requested shelf.\n"
        "- Picker and AGV meet at that shelf for loading.\n"
        "- AGV carries shelf to GOAL.\n"
        "- AGV returns shelf to storage shelf location with Picker support for unload.\n"
        "- Repeat with next request.\n"
        "\n"
        "Battery model:\n"
        "- Move step consumption = 1.\n"
        "- Load/Unload consumption = 2.\n"
        "- Charging need level: critical if <25, need_charging_soon if <50, not_needed if >=50.\n"
        "\n"
        "Charging situation (global):\n"
        "- Agents currently on charging cells: " + (", ".join(charge_now) if charge_now else "none") + "\n"
        "- Charging station occupancy:\n"
        "  - " + "\n  - ".join(charge_occupancy if charge_occupancy else ["none"]) + "\n"
        "\n"
        "Current global state:\n"
        "- Requested shelves:\n"
        "  - " + ("\n  - ".join(requested) if requested else "none") + "\n"
        "- AGV/Picker support pairs to reduce waiting:\n"
        "  - " + "\n  - ".join(pair_lines) + "\n"
        "- AGVs currently waiting for picker support:\n"
        "  - " + "\n  - ".join(waiting_support) + "\n"
        f"- Picker support priority note for you: {picker_priority_note}\n"
        "- All agents battery/position/target snapshot:\n"
        "  - " + "\n  - ".join(all_agents_overview) + "\n"
        "\n"
        f"You are agent_{agent_idx} ({agent.type.name}).\n"
        f"Your current position is ({int(agent.x)},{int(agent.y)}).\n"
        f"Your movement status: {'moving' if is_moving else 'not moving'}.\n"
        f"Your current target is {target_text}.\n"
        f"You need to meet: {meet_text}.\n"
        f"Your battery level is {float(agent.battery):.1f} ({battery_need}).\n"
        f"You are {carrying_text}.\n"
        f"Nearest charging station is action_id={nearest_charge_id} (estimated_steps={nearest_charge_steps}).\n"
        "\n"
        "Decision policy:\n"
        "- Choose exactly ONE action_id for this agent.\n"
        "- Choose from the valid action ids listed below.\n"
        "- If you are an AGV and you are carrying a requested shelf, move it to a GOAL action id.\n"
        f"- GOAL action ids are {goal_id_span}. Any GOAL position is acceptable for delivery.\n"
        "- If you are a Picker and any AGV is waiting for picker support, prioritize moving to that AGV target shelf action id.\n"
        "- If battery is critical, strongly prioritize charging.\n"
        "- If battery needs charging soon, decide between charging vs productive task based on risk and current coordination demand.\n"
        "- If battery is not_needed and there is productive work, avoid charging.\n"
        "- Keep AGV/Picker coordination tight to reduce AGV waiting.\n"
        "\n"
        "Valid action ids for this agent (from environment action mask):\n"
        "- " + "\n- ".join(candidate_text) + "\n"
        "\n"
        "Respond in plain text only:\n"
        "Reasoning: <short>\n"
        "Action: <action_id>\n"
    )


def build_language_prompt(env, valid_masks: np.ndarray, step_count: int, max_candidate_ids: int) -> str:
    # Retained for backward compatibility with existing logs/tools.
    # New planning loop uses per-agent prompts via build_agent_prompt.
    requested = get_requested_shelves(env)

    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    agent_lines = []
    for idx, agent in enumerate(env.agents):
        cands = candidate_ids_for_agent(env, idx, valid_masks, max_candidate_ids)
        cand_desc = []
        for aid in cands:
            if aid == 0:
                cand_desc.append("0:NOOP")
            else:
                ty = classify_action(env, int(aid))
                yx = env.action_id_to_coords_map.get(int(aid))
                if yx is None:
                    cand_desc.append(f"{int(aid)}:{ty}")
                else:
                    cand_desc.append(f"{int(aid)}:{ty}@({int(yx[1])},{int(yx[0])})")

        current_target = int(agent.target)
        target_type = classify_action(env, current_target) if current_target > 0 else "NOOP"
        target_coords = env.action_id_to_coords_map.get(current_target) if current_target > 0 else None
        at_target = bool(target_coords is not None and (int(agent.x), int(agent.y)) == (int(target_coords[1]), int(target_coords[0])))
        picker_at_same_cell = bool((int(agent.x), int(agent.y)) in picker_positions)
        needs_picker_for_load = bool(
            agent.type == AgentType.AGV
            and at_target
            and bool(agent.busy)
            and agent.carrying_shelf is None
            and target_type == "SHELF"
        )
        needs_picker_for_unload = bool(
            agent.type == AgentType.AGV
            and at_target
            and bool(agent.busy)
            and agent.carrying_shelf is not None
            and target_type == "SHELF"
        )
        waiting_for_picker_support = bool(
            agent.type == AgentType.AGV
            and (needs_picker_for_load or needs_picker_for_unload)
            and not picker_at_same_cell
        )

        agent_lines.append(
            f"agent_{idx} type={agent.type.name} pos=({int(agent.x)},{int(agent.y)}) battery={float(agent.battery):.1f} "
            f"busy={bool(agent.busy)} carrying_shelf_id={int(agent.carrying_shelf.id) if agent.carrying_shelf else None} "
            f"current_target={current_target} target_type={target_type} at_target={at_target} "
            f"picker_at_same_cell={picker_at_same_cell} "
            f"needs_picker_for_load={needs_picker_for_load} "
            f"needs_picker_for_unload={needs_picker_for_unload} "
            f"waiting_for_picker_support={waiting_for_picker_support} "
            f"candidates=[{', '.join(cand_desc)}]"
        )

    charging_ids = charging_action_ids(env)
    charging_span = f"{charging_ids[0]}..{charging_ids[-1]}" if charging_ids else "none"

    return (
        "You are dispatching robots in a warehouse.\n"
        "Goal: maximize shelf pickups/deliveries while keeping batteries safe and coordinating AGV with Picker.\n"
        "\n"
        "WAREHOUSE ABSTRACTION (must follow):\n"
        "- A request is for a SHELF (pod), not an item unit.\n"
        "- A request is fulfilled when the requested shelf is carried by an AGV to a GOAL location.\n"
        "- AGV-Picker co-location is required at shelf interactions only:\n"
        "  (a) pickup/load at shelf location, and (b) return/unload at shelf location.\n"
        "- Picker does NOT need to travel with AGV to GOAL.\n"
        "- After GOAL delivery, AGV still carries the same shelf and should return it to an empty shelf slot.\n"
        "\n"
        "BATTERY POLICY (important):\n"
        "- critical battery: < 10 => choose CHARGING.\n"
        "- medium battery: 10..29 => charging is optional, prefer task progress unless clearly risky.\n"
        "- healthy battery: >= 30 => DO NOT choose CHARGING for AGV/Picker.\n"
        "- With battery around 50+, charging is a bad decision in this task.\n"
        "\n"
        "CRITICAL RULE:\n"
        "- For each agent_i, choose ONLY one action_id from that same agent_i candidates=[...].\n"
        "- Do NOT invent new IDs. Do NOT use IDs from another agent's candidate list.\n"
        "- If an action_id is not listed in that agent's candidates, you must NOT choose it.\n"
        "- Requested shelves list is global context, but final choice must still come from agent_i candidates.\n"
        "- Avoid arbitrary retargeting when busy=True, but DO reassign when it reduces AGV waiting_for_picker_support or resolves idling.\n"
        "\n"
        "Action ID legend:\n"
        "- 0 = NOOP\n"
        f"- 1..{len(env.goals)} = GOAL locations\n"
        f"- {charging_span} = CHARGING stations\n"
        "- Other listed IDs are shelf locations.\n"
        "\n"
        "ACTION SELECTION ORDER (strict):\n"
        "1) If carrying_shelf_id is not None: prioritize GOAL action_ids (1..N), not requested shelf ids.\n"
        "2) Else if battery is critical (<10): choose CHARGING.\n"
        "3) Else choose a requested shelf/support shelf from candidates.\n"
        "4) Use NOOP only if no productive candidate is appropriate.\n"
        "5) Minimize waiting: schedule AGV and Picker movements so AGV waiting_for_picker_support stays low.\n"
        "6) CO-LOCATION RULE: for shelf interaction, AGV and Picker must go to the SAME shelf action_id in the same time window.\n"
        "   If AGV targets shelf X for load/unload support, Picker should also target shelf X unless battery is critical.\n"
        "7) Dynamic reassignment for throughput: if one agent is busy/unavailable, assign the nearest AVAILABLE counterpart\n"
        "   (nearest by visible coordinates among valid candidates) to avoid waiting and increase pickup rate.\n"
        "8) In 1-AGV/1-Picker mode, align picker action with AGV support shelf whenever possible.\n"
        "\n"
        "VARIABLE-DRIVEN SUPPORT RULE (strict):\n"
        "- Read AGV flags directly from agent lines: picker_at_same_cell, needs_picker_for_load, needs_picker_for_unload, waiting_for_picker_support.\n"
        "- If an AGV has waiting_for_picker_support=true, assign the Picker to that AGV current_target shelf action_id immediately.\n"
        "- If an AGV has needs_picker_for_load=true and picker_at_same_cell=false, Picker must move to that AGV shelf (same action_id).\n"
        "- If an AGV has needs_picker_for_unload=true and picker_at_same_cell=false, Picker must move to that AGV shelf (same action_id).\n"
        "- Do NOT send Picker to a different requested shelf while any AGV waiting_for_picker_support=true (unless picker battery is critical).\n"
        "- In conflicts, AGV support has higher priority than starting a new shelf request.\n"
        "- Simple mapping clue: if agent_x (AGV) has waiting_for_picker_support=true, then Picker action := agent_x.current_target.\n"
        "- Simple mapping clue: if agent_x needs_picker_for_load=true and picker_at_same_cell=false, then Picker action := agent_x.current_target.\n"
        "- Simple mapping clue: if agent_x needs_picker_for_unload=true and picker_at_same_cell=false, then Picker action := agent_x.current_target.\n"
        "Important scenarios:\n"
        "- If battery is critical (<10), send that agent to a CHARGING action_id.\n"
        "- If battery is healthy (>=30), do not send that agent to charging.\n"
        "- If AGV carries requested shelf, send to GOAL.\n"
        "- If AGV carries non-requested shelf, send to empty shelf return location.\n"
        "- If AGV goes to a requested shelf, Picker should choose the SAME shelf action_id unless picker battery is critical.\n"
        "- If AGV is returning a carried shelf to storage, Picker should support at that return shelf action_id.\n"
        "- Avoid sending picker to charging when AGV needs immediate shelf support and picker battery is not critical.\n"
        "- Deadlock breaker: if AGV is at a requested shelf with carrying_shelf_id=None, picker must move to that same shelf action_id.\n"
        "- If waiting_for_picker_support=true for AGV, assign nearest available picker to AGV support shelf immediately.\n"
        "- Picker NOOP is discouraged; only use NOOP when battery is critical, or when picker is already at the AGV support shelf.\n"
        "- Do not claim AGV is carrying shelf unless carrying_shelf_id is not None in the provided state.\n"
        "\n"
        "Respond in plain text ONLY in this format:\n"
        "Reasoning: <short explanation>\n"
        "Actions:\n"
        "agent_0 -> <action_id>\n"
        "agent_1 -> <action_id>\n"
        "... one line per agent\n"
        "Example:\n"
        "Reasoning: AGV goes to requested shelf; picker supports same shelf.\n"
        "Actions:\n"
        "agent_0 -> 12\n"
        "agent_1 -> 12\n"
        "Bad example (avoid): battery=51 and choosing charging.\n"
        "\n"
        "Do not output JSON.\n"
        "Use only listed candidate action ids for each agent.\n"
        f"\nStep: {step_count}\n"
        f"Requested shelves:\n- " + ("\n- ".join(requested) if requested else "none") + "\n"
        f"\nAgents:\n- " + "\n- ".join(agent_lines)
    )


def query_ollama_text(model: str, ollama_url: str, prompt: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 700},
    }
    req = request.Request(
        ollama_url,
        data=str.encode(json.dumps(payload)),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")).strip()


def parse_actions_from_text(text: str, num_agents: int) -> List[int]:
    actions = [0] * num_agents
    found = 0

    # Primary pattern: agent_0 -> 12
    for m in re.finditer(r"agent[_\s-]*(\d+)\s*[-:=]>?\s*(-?\d+)", text, flags=re.IGNORECASE):
        idx = int(m.group(1))
        act = int(m.group(2))
        if 0 <= idx < num_agents:
            actions[idx] = act
            found += 1

    # Secondary fallback: line starts with index
    if found == 0:
        for m in re.finditer(r"(?:^|\n)\s*(\d+)\s*[:=-]\s*(-?\d+)", text):
            idx = int(m.group(1))
            act = int(m.group(2))
            if 0 <= idx < num_agents:
                actions[idx] = act
                found += 1

    if found == 0:
        raise ValueError("No parseable actions found in LLM text output")

    return actions


def parse_single_action_from_text(text: str) -> int:
    # Prefer explicit final action lines. This avoids accidentally reading
    # "action_id=..." values that may appear in reasoning sentences.
    line_patterns = [
        r"^\s*action\s*[:=-]\s*(-?\d+)\b",
        r"^\s*action\s*[:=-]\s*(-?\d+)\s*:",
        r"^\s*agent[_\s-]*\d+\s*[-:=]>\s*(-?\d+)\b",
    ]
    for pattern in line_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))

    patterns = [
        r"action\s*[:=-]?\s*(-?\d+)",
        r"agent[_\s-]*\d+\s*[-:=]>\s*(-?\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Last-resort fallback: first integer anywhere.
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group(0))
    raise ValueError("No parseable single action found in LLM text output")


def validate_actions(env, actions: List[int], valid_masks: np.ndarray) -> List[int]:
    out = [0] * env.num_agents
    for i in range(env.num_agents):
        a = int(actions[i]) if i < len(actions) else 0
        if a < 0 or a >= env.action_size:
            a = 0
        if valid_masks[i, a] <= 0:
            a = 0
        out[i] = a
    return out


def _at_target(env, agent) -> bool:
    if int(agent.target) <= 0:
        return False
    coords = env.action_id_to_coords_map.get(int(agent.target))
    if coords is None:
        return False
    return (int(agent.x), int(agent.y)) == (int(coords[1]), int(coords[0]))


def replan_signals(env) -> tuple[bool, List[str]]:
    """
    Query LLM only when there is a meaningful planning event:
    - some agent is free (not busy),
    - some busy agent reached target (transition point),
    - AGV is waiting for picker support at target shelf.
    """
    reasons: List[str] = []
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}

    free_agents = [i for i, a in enumerate(env.agents) if not bool(a.busy)]
    if free_agents:
        reasons.append(f"free_agents={free_agents}")

    reached_targets = [i for i, a in enumerate(env.agents) if bool(a.busy) and _at_target(env, a)]
    if reached_targets:
        reasons.append(f"reached_targets={reached_targets}")

    waiting_support: List[int] = []
    for i, a in enumerate(env.agents[: env.num_agvs]):
        if not bool(a.busy) or not _at_target(env, a):
            continue
        if int(a.target) <= 0:
            continue
        target_type = classify_action(env, int(a.target))
        if target_type != "SHELF":
            continue
        picker_here = (int(a.x), int(a.y)) in picker_positions
        if not picker_here:
            waiting_support.append(i)
    if waiting_support:
        reasons.append(f"agv_waiting_picker={waiting_support}")

    return (len(reasons) > 0), reasons


def explain_actions(env, actions: List[int]) -> str:
    lines: List[str] = []
    for i, a in enumerate(actions):
        agent = env.agents[i]
        if int(a) == 0:
            lines.append(f"agent_{i} {agent.type.name} action=0 NOOP busy={bool(agent.busy)}")
            continue
        yx = env.action_id_to_coords_map.get(int(a))
        if yx is None:
            lines.append(f"agent_{i} {agent.type.name} action={int(a)} UNKNOWN busy={bool(agent.busy)}")
            continue
        note = "will_apply_new_target" if not agent.busy else "currently_busy_keeps_current_path"
        lines.append(
            f"agent_{i} {agent.type.name} action={int(a)} {classify_action(env, int(a))}@({int(yx[1])},{int(yx[0])}) "
            f"busy={bool(agent.busy)} {note}"
        )
    return "\n".join(lines)

def explain_candidates(env, valid_masks: np.ndarray, max_candidate_ids: int) -> str:
    lines: List[str] = []
    for idx, agent in enumerate(env.agents):
        cands = candidate_ids_for_agent(env, idx, valid_masks, max_candidate_ids)
        cand_desc = []
        for aid in cands:
            if aid == 0:
                cand_desc.append("0:NOOP")
            else:
                ty = classify_action(env, int(aid))
                yx = env.action_id_to_coords_map.get(int(aid))
                if yx is None:
                    cand_desc.append(f"{int(aid)}:{ty}")
                else:
                    cand_desc.append(f"{int(aid)}:{ty}@({int(yx[1])},{int(yx[0])})")
        lines.append(f"agent_{idx} type={agent.type.name} candidates=[{', '.join(cand_desc)}]")
    return "\n".join(lines)


def snapshot_agents(env) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    picker_positions = {(int(a.x), int(a.y)) for a in env.agents if a.type == AgentType.PICKER}
    for i, a in enumerate(env.agents):
        out.append(
            {
                "idx": i,
                "type": a.type.name,
                "x": int(a.x),
                "y": int(a.y),
                "busy": bool(a.busy),
                "target": int(a.target),
                "carrying": int(a.carrying_shelf.id) if a.carrying_shelf is not None else None,
                "battery": float(a.battery),
                "picker_at_same_cell": bool((int(a.x), int(a.y)) in picker_positions) if a.type == AgentType.AGV else None,
            }
        )
    return out


def explain_state_changes(before: List[Dict[str, Any]], env) -> str:
    lines: List[str] = []
    for b in before:
        i = int(b["idx"])
        a = env.agents[i]
        new_carry = int(a.carrying_shelf.id) if a.carrying_shelf is not None else None
        changes: List[str] = []
        if b["carrying"] != new_carry:
            if b["carrying"] is None and new_carry is not None:
                changes.append(f"load_event carrying None->{new_carry}")
            elif b["carrying"] is not None and new_carry is None:
                changes.append(f"unload_event carrying {b['carrying']}->None")
            else:
                changes.append(f"carrying {b['carrying']}->{new_carry}")
        if b["target"] != int(a.target):
            changes.append(f"target {b['target']}->{int(a.target)}")
        if b["busy"] != bool(a.busy):
            changes.append(f"busy {b['busy']}->{bool(a.busy)}")
        if (b["x"], b["y"]) != (int(a.x), int(a.y)):
            changes.append(f"pos ({b['x']},{b['y']})->({int(a.x)},{int(a.y)})")
        if abs(float(b["battery"]) - float(a.battery)) > 1e-9:
            changes.append(f"battery {float(b['battery']):.1f}->{float(a.battery):.1f}")
        if a.type == AgentType.AGV:
            picker_positions = {(int(p.x), int(p.y)) for p in env.agents if p.type == AgentType.PICKER}
            picker_here = bool((int(a.x), int(a.y)) in picker_positions)
            if b["picker_at_same_cell"] != picker_here:
                changes.append(f"picker_at_same_cell {b['picker_at_same_cell']}->{picker_here}")
        if changes:
            lines.append(f"agent_{i} type={a.type.name}: " + "; ".join(changes))
    return "\n".join(lines) if lines else "no_state_change"


def llm_episode(env, args, seed: int):
    _ = env.reset(seed=seed)
    done = False
    step_count = 0
    llm_calls = 0
    llm_failures = 0
    delivery_happened = False
    steps_since_replan = 0

    infos: List[Dict[str, Any]] = []
    episode_returns = np.zeros(env.num_agents)
    global_episode_return = 0.0
    last_actions = [0] * env.num_agents

    log_block("START", f"seed={seed}, env={args.env_id}, agents={env.num_agents}")

    while not done and step_count < args.max_steps_per_episode:
        valid_masks = safe_valid_action_masks(env)
        log_block("STEP", f"step={step_count}")

        should_replan = False
        reasons: List[str] = []
        if step_count % max(1, args.decision_interval) == 0:
            should_replan, reasons = replan_signals(env)
            if not should_replan and steps_since_replan >= max(1, args.force_llm_replan_steps):
                should_replan = True
                reasons = [f"forced_replan_after_{steps_since_replan}_steps"]

        if should_replan:
            log_block("REPLAN_TRIGGER", f"step={step_count}; " + ", ".join(reasons))
            log_block("CANDIDATES", explain_candidates(env, valid_masks, args.max_candidate_ids))
            fallback = fallback_actions(env, valid_masks)
            actions = [0] * env.num_agents
            for idx in range(env.num_agents):
                agent_prompt = build_agent_prompt(env, idx, valid_masks, step_count, args.max_candidate_ids)
                log_block(f"PROMPT_agent_{idx}", maybe_truncate(agent_prompt, args.log_text_chars))
                candidates = candidate_ids_for_agent(env, idx, valid_masks, args.max_candidate_ids)
                try:
                    llm_text = query_ollama_text(args.model, args.ollama_url, agent_prompt, args.request_timeout_s)
                    llm_calls += 1
                    log_block(f"LLM_OUTPUT_agent_{idx}", maybe_truncate(llm_text, args.log_text_chars))
                    action_id = parse_single_action_from_text(llm_text)
                    if action_id < 0 or action_id >= env.action_size or valid_masks[idx, action_id] <= 0:
                        raise ValueError(f"action_id={action_id} invalid_by_mask")
                    actions[idx] = int(action_id)
                except (error.URLError, TimeoutError, ValueError) as exc:
                    llm_failures += 1
                    actions[idx] = int(fallback[idx])
                    log_block(f"LLM_PARSE_ERROR_agent_{idx}", repr(exc))
                    log_block(f"FALLBACK_agent_{idx}", f"action={actions[idx]}")
            log_block("VALIDATED_ACTIONS", str(actions))
            log_block("ACTION_EFFECT", explain_actions(env, actions))
            steps_since_replan = 0
        else:
            actions = validate_actions(env, last_actions, valid_masks)
            steps_since_replan += 1
            log_block("REPLAN_SKIP", f"step={step_count} no_event; reuse_actions; steps_since_replan={steps_since_replan}")
            log_block("REUSED_ACTIONS", str(actions))
            log_block("ACTION_EFFECT", explain_actions(env, actions))

        last_actions = list(actions)

        if args.render:
            env.render(mode="human")

        prev_agents = snapshot_agents(env)
        _, reward, terminated, truncated, info = env.step(actions)
        log_block("STATE_CHANGE", explain_state_changes(prev_agents, env))
        infos.append(info)
        episode_returns += np.asarray(reward, dtype=np.float64)
        global_episode_return += float(np.sum(reward))

        step_deliveries = int(info.get("shelf_deliveries", 0))
        if step_deliveries > 0:
            delivery_happened = True
            log_block("DELIVERY", f"step={step_count} shelf_deliveries={step_deliveries}")

        step_count += 1
        done = all(terminated) or all(truncated)

        if args.progress_every > 0 and step_count % args.progress_every == 0:
            log_block(
                "PROGRESS",
                f"step={step_count} total_deliveries={sum(i.get('shelf_deliveries', 0) for i in infos)} "
                f"llm_calls={llm_calls} llm_failures={llm_failures}",
            )

    if infos:
        infos[-1]["llm_calls"] = llm_calls
        infos[-1]["llm_failures"] = llm_failures
        infos[-1]["delivery_happened"] = delivery_happened

    return infos, global_episode_return, episode_returns, delivery_happened


def print_episode_result(prefix: str, idx: int, elapsed: float, metrics: Dict[str, Any]):
    print(
        f"\n{prefix} Episode {idx}: "
        f"[Overall Pick Rate={metrics.get('overall_pick_rate', 0.0):.2f}] "
        f"[Global return={metrics.get('global_episode_return', 0.0):.2f}] "
        f"[Total shelf deliveries={metrics.get('total_deliveries', 0):.2f}] "
        f"[Total clashes={metrics.get('total_clashes', 0):.2f}] "
        f"[Total stuck={metrics.get('total_stuck', 0):.2f}] "
        f"[Total AGVs distance={metrics.get('total_agvs_distance', 0):.2f}] "
        f"[Total Pickers distance={metrics.get('total_pickers_distance', 0):.2f}] "
        f"[Episode length={metrics.get('episode_length', 0)}] "
        f"[FPS={metrics.get('episode_length', 0) / elapsed if elapsed > 0 else 0.0:.2f}]\n"
    )


def summarize_policy_results(name: str, metric_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not metric_rows:
        return {"overall_pick_rate": 0.0, "global_episode_return": 0.0, "total_deliveries": 0.0}
    avg_pick = float(np.mean([m.get("overall_pick_rate", 0.0) for m in metric_rows]))
    avg_ret = float(np.mean([m.get("global_episode_return", 0.0) for m in metric_rows]))
    avg_del = float(np.mean([m.get("total_deliveries", 0.0) for m in metric_rows]))
    print(
        f"\n{name} AVG: [Overall Pick Rate={avg_pick:.2f}] "
        f"[Global return={avg_ret:.2f}] [Total deliveries={avg_del:.2f}]\n"
    )
    return {
        "overall_pick_rate": avg_pick,
        "global_episode_return": avg_ret,
        "total_deliveries": avg_del,
    }


if __name__ == "__main__":
    args = parser.parse_args()

    env = gym.make(args.env_id)
    env.unwrapped.verbose_events = bool(args.warehouse_event_logs)

    llm_rows: List[Dict[str, Any]] = []
    llm_delivery_any = False

    for ep in range(args.num_episodes):
        start = time.time()
        infos, global_ret, ep_rets, delivery_happened = llm_episode(env.unwrapped, args, seed=args.seed + ep)
        llm_delivery_any = llm_delivery_any or delivery_happened
        metrics = info_statistics(infos, global_ret, ep_rets)
        print_episode_result("LLM", ep, time.time() - start, metrics)
        llm_rows.append(metrics)

    llm_avg = summarize_policy_results("LLM", llm_rows)
    log_block("RESULT", f"delivery_happened_any_episode={llm_delivery_any}")

    if not args.skip_heuristic:
        heuristic_rows: List[Dict[str, Any]] = []
        for ep in range(args.num_episodes):
            start = time.time()
            infos, global_ret, ep_rets = heuristic_episode(env.unwrapped, render=args.render, seed=args.seed + ep, save_gif=False)
            metrics = info_statistics(infos, global_ret, ep_rets)
            print_episode_result("HEURISTIC", ep, time.time() - start, metrics)
            heuristic_rows.append(metrics)
        heuristic_avg = summarize_policy_results("HEURISTIC", heuristic_rows)

        print("\nCOMPARISON:\n")
        print(f"Pick Rate delta (LLM-HEURISTIC): {llm_avg['overall_pick_rate'] - heuristic_avg['overall_pick_rate']:.2f}")
        print(f"Return delta (LLM-HEURISTIC): {llm_avg['global_episode_return'] - heuristic_avg['global_episode_return']:.2f}")
        print(f"Deliveries delta (LLM-HEURISTIC): {llm_avg['total_deliveries'] - heuristic_avg['total_deliveries']:.2f}")
