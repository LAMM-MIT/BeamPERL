#!/usr/bin/env python3
"""
Beam prompt calculation utilities.
Handles the calculation of prompt components for beam analysis question generation.
"""

import json
from typing import Dict, Any

def prompt_text_from_load_and_params(sample: Dict[str, Any]) -> str:
    """Generate prompt text using only load_position and parameters columns.
    
    Args:
        sample: Dictionary containing the sample data with load_position and parameters columns
        
    Returns:
        str: The prompt text combining load position and parameters information
    """
    load_positions = sample.get("load_positions", "")
    parameters_str = sample.get("parameters", "")
    points = sample.get("points", "")
    reactions = sample.get("reactions", "")
    
    # Parse parameters if it's a JSON string
    if isinstance(parameters_str, str):
        try:
            parameters = json.loads(parameters_str)
        except (json.JSONDecodeError, TypeError):
            parameters = {}
    else:
        parameters = parameters_str
    
    # Parse points if it's a JSON string
    if isinstance(points, str):
        try:
            points_data = json.loads(points)
        except (json.JSONDecodeError, TypeError):
            points_data = {"points": []}
    else:
        points_data = points
    
    # Parse load_positions if it's a JSON string
    if isinstance(load_positions, str):
        try:
            load_positions_data = json.loads(load_positions)
        except (json.JSONDecodeError, TypeError):
            load_positions_data = []
    else:
        load_positions_data = load_positions
    
    # Extract individual parameters
    L = str(parameters.get("L", ""))
    E = str(parameters.get("E", ""))
    I = str(parameters.get("I", ""))
    
    # Extract loads from P object
    P_loads = parameters.get("P", {})
    M_moments = parameters.get("M", {})
    Q_distributed = parameters.get("Q", {})
    R_supports = parameters.get("R", {})

    prompt_parts = []

    prompt_parts.append(f"The beam has a length of {L}.")
    prompt_parts.append(f"The beam has a Young's modulus of {E}.")
    prompt_parts.append(f"The beam has a moment of inertia of {I}.")
    
    # Handle point loads from P object
    for load_key, load_data in P_loads.items():
        if isinstance(load_data, dict):
            location = load_data.get("location", "")
            magnitude = load_data.get("magnitude", "")
            if magnitude and magnitude != "0":
                prompt_parts.append(f"There is an applied point load of {magnitude} at x={location}.")
                if "-" in magnitude:
                    prompt_parts.append(f"A negative load means the load is applied downward.")
    
    # Handle moments from M object
    for moment_key, moment_data in M_moments.items():
        if isinstance(moment_data, dict):
            location = moment_data.get("location", "")
            magnitude = moment_data.get("magnitude", "")
            if magnitude and magnitude != "0":
                prompt_parts.append(f"There is an applied moment of {magnitude} at x={location}.")
    
    # Handle distributed loads from Q object
    for q_key, q_data in Q_distributed.items():
        if isinstance(q_data, dict):
            start_location = q_data.get("start_location", "")
            end_location = q_data.get("end_location", "")
            magnitude = q_data.get("magnitude", "")
            if magnitude and magnitude != "0":
                if start_location and end_location:
                    prompt_parts.append(f"There is a distributed load of {magnitude} from x={start_location} to x={end_location}.")
                else:
                    prompt_parts.append(f"There is a distributed load of {magnitude}.")
    
    # Extract support information from R object
    supports = []
    for support_key, support_data in R_supports.items():
        if isinstance(support_data, dict):
            location = support_data.get("location", "")
            support_type = support_data.get("type", "")
            if location and support_type:
                supports.append((location, support_type))
    
    # Add support information to prompt
    if len(supports) >= 2:
        support1_coord, support1_type = supports[0]
        support2_coord, support2_type = supports[1]
        prompt_parts.append(f"The beam has a {support1_type} support at x={support1_coord} and a {support2_type} support at x={support2_coord}.")
    elif len(supports) == 1:
        support_coord, support_type = supports[0]
        prompt_parts.append(f"The beam has a {support_type} support at x={support_coord}.")
    else:
        # Fallback to default if no supports found
        prompt_parts.append(f"The beam has a pin support at x=0 and a roller support at x={L}.")
    
    return "\n".join(prompt_parts)