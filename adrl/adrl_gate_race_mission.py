#!/usr/bin/env python3
"""
ADRL gate-race mission (single drone).

Loads a race map, finds gate objects, sorts them in numeric order,
and flies the drone through each gate center.

Run (with ADRL already running):
  uv run python adrl_gate_race_mission.py --level Soccer_Field_Easy
"""

import argparse
import math
import re
import time
from typing import List, Tuple

import airsimdroneracinglab as airsim


def natural_gate_key(name: str):
    nums = re.findall(r"\d+", name)
    first = int(nums[0]) if nums else 10**9
    return (first, name)


def is_valid_pose(pose) -> bool:
    p = pose.position
    vals = [p.x_val, p.y_val, p.z_val]
    return all(not (math.isnan(v) or math.isinf(v)) for v in vals)


def list_gates(client, pattern: str) -> List[str]:
    # Common ADRL/AirSim gate naming usually starts with Gate
    names = client.simListSceneObjects(pattern)
    if not names and pattern != "Gate.*":
        names = client.simListSceneObjects("Gate.*")
    return sorted(names, key=natural_gate_key)


def gate_centers(client, gate_names: List[str], z_offset: float) -> List[Tuple[str, float, float, float]]:
    out = []
    for g in gate_names:
        pose = client.simGetObjectPose(g)
        if not is_valid_pose(pose):
            continue
        x = float(pose.position.x_val)
        y = float(pose.position.y_val)
        z = float(pose.position.z_val + z_offset)
        out.append((g, x, y, z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--level", default="Soccer_Field_Easy", help="ADRL race map name")
    ap.add_argument("--load-level", action="store_true", default=True)
    ap.add_argument("--no-load-level", action="store_true", help="Do not call simLoadLevel")
    ap.add_argument("--start-race", action="store_true", help="Call simStartRace(tier)")
    ap.add_argument("--race-tier", type=int, default=1)
    ap.add_argument("--gate-pattern", default="Gate.*", help="Regex for gate object names")
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--settle", type=float, default=3.0, help="Seconds after level load")
    ap.add_argument("--pause", type=float, default=0.10, help="Pause between gates")
    ap.add_argument("--z-offset", type=float, default=0.0, help="Adjust gate center altitude in NED z")
    ap.add_argument("--vehicle", default="drone_1")
    args = ap.parse_args()

    do_load_level = args.load_level and not args.no_load_level

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()

    if do_load_level:
        print(f"Loading level: {args.level}")
        client.simLoadLevel(args.level)
        time.sleep(args.settle)
        client.confirmConnection()

    # ADRL Python client compatibility fix:
    # some builds reference self.race_tier in simGetObjectPose() before defining it.
    if not hasattr(client, "race_tier"):
        client.race_tier = args.race_tier

    if args.start_race:
        print(f"Starting race tier {args.race_tier}")
        client.simStartRace(args.race_tier)
        time.sleep(0.5)

    vehicle = args.vehicle
    if hasattr(client, "listVehicles"):
        try:
            vs = client.listVehicles()
            if vs and vehicle not in vs:
                print(f"Configured vehicle '{vehicle}' not found in {vs}; using '{vs[0]}'")
                vehicle = vs[0]
        except Exception:
            pass

    gates = list_gates(client, args.gate_pattern)
    if not gates:
        raise RuntimeError(
            f"No gate objects found with pattern '{args.gate_pattern}'. "
            "Try --gate-pattern 'Gate.*' and confirm a race map is loaded."
        )

    targets = gate_centers(client, gates, args.z_offset)
    if not targets:
        raise RuntimeError("Gate objects found, but no valid gate poses were readable.")

    print(f"Using vehicle={vehicle}")
    print(f"Found {len(targets)} gate targets")

    client.enableApiControl(vehicle_name=vehicle)
    client.arm(vehicle_name=vehicle)

    client.takeoffAsync(vehicle_name=vehicle).join()
    time.sleep(0.2)

    for i, (g, x, y, z) in enumerate(targets, start=1):
        print(f"[{i:02d}/{len(targets):02d}] {g} -> x={x:.2f}, y={y:.2f}, z={z:.2f}")
        client.moveToPositionAsync(
            x,
            y,
            z,
            args.speed,
            timeout_sec=args.timeout,
            vehicle_name=vehicle,
        ).join()
        time.sleep(args.pause)

    client.hoverAsync(vehicle_name=vehicle).join()
    client.landAsync(timeout_sec=30, vehicle_name=vehicle).join()
    client.disarm(vehicle_name=vehicle)
    print("Gate mission complete")


if __name__ == "__main__":
    main()
