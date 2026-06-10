#!/usr/bin/env python3
"""
Generate the expanded BeamRL evaluation dataset v2 (123 samples, six categories).

Categories
----------
  id              30 samples  | single point load, supports at ends
  ood_loads       30 samples  | N in {2,3,4} point loads, supports at ends
  ood_supports    18 samples  | varying support positions, N in {1,2,3} point loads
  ood_dist_left   5 samples   | distributed load anchored to the pin
  ood_dist_middle 5 samples   | distributed load fully interior
  ood_dist_right  5 samples   | distributed load anchored to the roller
  ood_length      15 samples  | beam length in {7l, 11l, 13l}, supports at ends
  ood_moment      15 samples  | pure applied couple M = c*P*L, no point load
  TOTAL           123 samples

The original 24-sample evaluation set (tphage/BeamRL-EvalData) is loaded
bit-identically and counted within the id / ood_loads / ood_supports buckets —
30/30/18 = 4/8/12 (original) + 26/22/6 (freshly generated).

All beam configurations are solved symbolically with symbeam. Questions for
freshly-generated samples are produced with the same 7B LLM used in the
original dataset notebook (RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8),
yielding 4 natural-language question variants per sample (consistent with the
original 24 samples; the eval harness consumes query[0]).

The combined dataset is uploaded as tphage/BeamRL-EvalData-v2.

Usage (after huggingface-cli login)
-----------------------------------
    cd /path/to/beamperl/DataGen
    python generate_eval_v2.py

    # Skip LLM (template questions only — for testing without GPU). Implies
    # --no-upload so a template-only dataset never lands on the Hub by accident.
    python generate_eval_v2.py --no-llm

    # Skip upload, save locally
    python generate_eval_v2.py --no-upload --out eval_v2_local.json
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Iterable

import sympy
from sympy.abc import L, E, I, P, M, q, x

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── symbeam path setup ────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYMBEAM_PATH = os.path.join(SCRIPT_DIR, "symbeam_v2")
if os.path.exists(SYMBEAM_PATH):
    sys.path.insert(0, SYMBEAM_PATH)
else:
    raise FileNotFoundError(f"symbeam_v2 not found at {SYMBEAM_PATH}")

from symbeam import beam as SymBeam

# ── LLM constants ─────────────────────────────────────────────────────────────

LLM_MODEL_NAME = "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8"
N_QUESTION_VARIANTS = 4

PROMPT_Q_SYSTEM = """You are a question generation assistant. You will be given information about the setup of a statically loaded beam. Your task is to generate a question that asks the reader to calculate the reaction forces at the supports.

Generate a single, self-contained question that includes all the provided details from the setup below. All details are correct. The question should be short and concise. Use limited time reasoning.

The question needs to state the length of the beam, the location and type of the supports of the beam, and the location, magnitude, direction and type of the loads applied to the beam.

Following the think tag concluding the reasoning section, return only the question and no other dialogue referencing the prompt or the setup."""

# ── Design constants (locked) ─────────────────────────────────────────────────

# Held-out position grid: positions x_i / L in this set are explicitly outside
# the training grid {0.05k | k = 0..20}. Used for ID, multi-load, varying-supports
# and length-variation categories.
HELDOUT_POSITIONS: tuple[float, ...] = (
    0.07, 0.13, 0.21, 0.27, 0.33, 0.41, 0.53, 0.61, 0.73, 0.83,
)

# Point-load magnitudes (multiples of P; sign indicates direction — negative = down).
P_MAGNITUDES: tuple[int, ...] = (-7, -13, -19)

# Beam-length multipliers for the OOD-length-variation category.
LENGTH_VARIANTS: tuple[int, ...] = (7, 11, 13)

# Applied-moment coefficients (M = c * P * L) and fractional locations along the beam.
MOMENT_C_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5)
MOMENT_LOCATIONS_FRAC: tuple[float, ...] = (0.25, 0.5, 0.75)  # of beam length

# Beam length used by everything except OOD-length-variation.
DEFAULT_LENGTH_MULT = 9

# Per-category targets for *freshly generated* samples
# (the original 24 are loaded separately; sums below + 24 = 123)
TARGET_ID_NEW = 26              # 30 ID total = 4 (original) + 26 (new)
TARGET_MULTI_LOAD_NEW = 22      # 30 multi   = 8 (original) + 22 (new)
TARGET_VARYING_SUPPORTS_NEW = 6  # 18 supports = 12 (orig)  + 6 (new)
TARGET_DIST_PER_SUBCAT = 5      # 5 left + 5 middle + 5 right = 15
TARGET_LENGTH_PER_VARIANT = 5   # 5 × 3 lengths = 15
TARGET_MOMENT = 15              # 5 c-values × 3 locations  = 15

# ── symbolic helpers ──────────────────────────────────────────────────────────

def coeff_of_P(sympy_expr_str: str) -> float:
    expr = sympy.sympify(sympy_expr_str)
    return float(expr.coeff(P))


def format_answer_str(coeff: float) -> str:
    """Six-decimal signed coefficient string. ``-0.222222`` → ``"-0.222222P"``."""
    return f"{coeff:.6f}P"


def verify_equilibrium(
    reactions_list: list[dict],
    total_applied_load_coeff: float,
    tol: float = 1e-4,
) -> bool:
    """ΣF_y = 0: sum of force reactions must offset the total applied vertical load.

    ``total_applied_load_coeff`` is the signed sum of applied vertical loads
    expressed as a coefficient of P (downward → negative). For a pure-couple
    configuration the value should be 0.0.
    """
    total_reaction = sum(
        coeff_of_P(r["value"]) for r in reactions_list if r.get("type") == "Force"
    )
    return abs(total_reaction + total_applied_load_coeff) < tol


# ── beam solvers ──────────────────────────────────────────────────────────────

def _build_beam(L_mult: int) -> "SymBeam":
    b = SymBeam(L_mult * L)
    b.set_young(0, L_mult * L, E)
    b.set_inertia(0, L_mult * L, I)
    return b


def _sort_reactions_by_position(force_rxns: list[dict]) -> list[tuple[float, float]]:
    return sorted(
        ((float(sympy.sympify(r["point"]).subs(L, 1)), coeff_of_P(r["value"]))
         for r in force_rxns),
        key=lambda t: t[0],
    )


def solve_point_load_beam(
    L_mult: int,
    supports: list[tuple[str, str]],   # [(loc_expr, support_type), ...]
    load_positions_frac: list[float],  # fractions of L_mult*L
    load_magnitudes: list[int],        # multiples of P (e.g. -7, -13, -19)
) -> dict | None:
    """Solve a beam with point loads only. Returns a dict with ``answer``,
    ``reactions_raw``, and ``total_load_coeff`` (sum of magnitudes)."""
    try:
        b = _build_beam(L_mult)
        for loc_expr, support_type in supports:
            b.add_support(sympy.sympify(loc_expr), support_type)
        for frac, mag in zip(load_positions_frac, load_magnitudes):
            x_loc = sympy.sympify(f"{frac * L_mult}*L")
            b.add_point_load(x_loc, sympy.sympify(f"{mag}*P"))
        result = b.solve_v3(subs={L: 1}, output=False)
    except Exception as e:
        logger.warning(
            f"solve_point_load_beam failed (L_mult={L_mult}, supports={supports}, "
            f"loads={list(zip(load_positions_frac, load_magnitudes))}): {e}"
        )
        return None

    force_rxns = [r for r in result["reactions"]["reactions"] if r.get("type") == "Force"]
    if len(force_rxns) != len(supports):
        logger.warning(
            f"Expected {len(supports)} force reactions, got {len(force_rxns)}"
        )
        return None

    total_load_coeff = float(sum(load_magnitudes))
    if not verify_equilibrium(force_rxns, total_load_coeff):
        logger.warning(
            f"Equilibrium FAILED for point-load config "
            f"(L_mult={L_mult}, mags={load_magnitudes})"
        )
        return None

    sorted_rxns = _sort_reactions_by_position(force_rxns)
    return {
        "answer": [format_answer_str(c) for _, c in sorted_rxns],
        "reactions_raw": [r["value"] for r in force_rxns],
        "total_load_coeff": total_load_coeff,
    }


def solve_dist_load_beam(k: int, n_start: float, n_end: float) -> dict | None:
    """Simply supported beam (pin@0, roller@9L) with q = -k·P/L on [n_start·L, n_end·L]."""
    try:
        b = _build_beam(DEFAULT_LENGTH_MULT)
        b.add_support(0, "pin")
        b.add_support(DEFAULT_LENGTH_MULT * L, "roller")
        b.add_distributed_load(
            sympy.sympify(f"{n_start}*L"),
            sympy.sympify(f"{n_end}*L"),
            f"-{k}*P/L",
        )
        result = b.solve_v3(subs={L: 1}, output=False)
    except Exception as e:
        logger.warning(f"solve_dist_load_beam failed (k={k}, [{n_start},{n_end}]): {e}")
        return None

    force_rxns = [r for r in result["reactions"]["reactions"] if r.get("type") == "Force"]
    if len(force_rxns) != 2:
        logger.warning(f"Expected 2 force reactions, got {len(force_rxns)} for k={k}")
        return None

    total_load_coeff = -k * (n_end - n_start)
    if not verify_equilibrium(force_rxns, total_load_coeff):
        logger.warning(f"Equilibrium FAILED for k={k}, [{n_start},{n_end}]")
        return None

    sorted_rxns = _sort_reactions_by_position(force_rxns)
    return {
        "answer": [format_answer_str(c) for _, c in sorted_rxns],
        "reactions_raw": [r["value"] for r in force_rxns],
        "total_load_coeff": total_load_coeff,
    }


def solve_moment_beam(c: int, x_m_frac: float) -> dict | None:
    """Simply supported beam with a pure couple M = c·P·L applied at x_m_frac·9L."""
    try:
        b = _build_beam(DEFAULT_LENGTH_MULT)
        b.add_support(0, "pin")
        b.add_support(DEFAULT_LENGTH_MULT * L, "roller")
        x_m = sympy.sympify(f"{x_m_frac * DEFAULT_LENGTH_MULT}*L")
        b.add_point_moment(x_m, sympy.sympify(f"{c}*P*L"))
        result = b.solve_v3(subs={L: 1}, output=False)
    except Exception as e:
        logger.warning(f"solve_moment_beam failed (c={c}, x_m_frac={x_m_frac}): {e}")
        return None

    force_rxns = [r for r in result["reactions"]["reactions"] if r.get("type") == "Force"]
    if len(force_rxns) != 2:
        logger.warning(f"Expected 2 force reactions, got {len(force_rxns)} for moment c={c}")
        return None

    # Pure couple: total applied vertical load is 0.
    if not verify_equilibrium(force_rxns, 0.0):
        logger.warning(f"Equilibrium FAILED for moment c={c}, x_m_frac={x_m_frac}")
        return None

    sorted_rxns = _sort_reactions_by_position(force_rxns)
    return {
        "answer": [format_answer_str(c_val) for _, c_val in sorted_rxns],
        "reactions_raw": [r["value"] for r in force_rxns],
        "total_load_coeff": 0.0,
    }


# ── description builders ──────────────────────────────────────────────────────

def _fmt_loc(frac: float, L_mult: int) -> str:
    """Format a location expression: ``frac * L_mult * L``."""
    val = frac * L_mult
    # Round to a clean number of decimals; sympy is happy with either form.
    if abs(val - round(val)) < 1e-9:
        return f"{int(round(val))}*L"
    return f"{val:.4f}*L"


def build_point_load_description(
    L_mult: int,
    supports: list[tuple[str, str]],
    load_positions_frac: list[float],
    load_magnitudes: list[int],
) -> str:
    lines = [
        f"The beam has a length of {L_mult}*L.",
        "The beam has a Young's modulus of E.",
        "The beam has a moment of inertia of I.",
    ]
    for loc_expr, support_type in supports:
        lines.append(f"The beam has a {support_type} support at x={loc_expr}.")
    for frac, mag in zip(load_positions_frac, load_magnitudes):
        loc_str = _fmt_loc(frac, L_mult)
        sign_word = "downward" if mag < 0 else "upward"
        lines.append(
            f"There is a point load of {mag}*P applied at x={loc_str}, acting {sign_word}."
        )
    return "\n".join(lines)


def build_dist_load_description(k: int, n_start: float, n_end: float) -> str:
    start_str = "0" if n_start == 0 else (
        f"{int(n_start)}*L" if n_start == int(n_start) else f"{n_start}*L"
    )
    end_str = f"{int(n_end)}*L" if n_end == int(n_end) else f"{n_end}*L"
    return "\n".join([
        f"The beam has a length of {DEFAULT_LENGTH_MULT}*L.",
        "The beam has a Young's modulus of E.",
        "The beam has a moment of inertia of I.",
        f"There is a uniform distributed load of -{k}*P/L per unit length"
        f" from x={start_str} to x={end_str}.",
        "A negative load means the load is applied downward.",
        f"The beam has a pin support at x=0 and a roller support at x={DEFAULT_LENGTH_MULT}*L.",
    ])


def build_moment_description(c: int, x_m_frac: float) -> str:
    loc_str = _fmt_loc(x_m_frac, DEFAULT_LENGTH_MULT)
    return "\n".join([
        f"The beam has a length of {DEFAULT_LENGTH_MULT}*L.",
        "The beam has a Young's modulus of E.",
        "The beam has a moment of inertia of I.",
        f"There is an applied moment (couple) of {c}*P*L at x={loc_str}.",
        "A positive moment is counter-clockwise.",
        f"The beam has a pin support at x=0 and a roller support at x={DEFAULT_LENGTH_MULT}*L.",
    ])


def build_template_question(description: str) -> str:
    return description + (
        "\nCalculate the vertical support reaction forces at each support."
        " Express your answers as multiples of P."
    )


# ── LLM question generation ───────────────────────────────────────────────────

def load_llm():
    from vllm import LLM, SamplingParams
    logger.info(f"Loading question-generation LLM: {LLM_MODEL_NAME}")
    llm = LLM(model=LLM_MODEL_NAME, max_model_len=3072, dtype="half")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=5120,
        n=N_QUESTION_VARIANTS,
    )
    logger.info("LLM loaded.")
    return llm, sampling_params


def extract_question_from_response(response_text: str) -> str:
    close_tag = re.compile(r"</\s*think\s*>", flags=re.IGNORECASE)
    matches = list(close_tag.finditer(response_text))
    if matches:
        candidate = response_text[matches[-1].end():].strip()
        if candidate:
            return candidate
    return response_text.strip()


def generate_questions_with_llm(samples: list[dict], llm, sampling_params) -> list[dict]:
    """Replace each sample's placeholder ``query`` with N_QUESTION_VARIANTS LLM completions."""
    descriptions = [s["_description"] for s in samples]
    prompts = [
        PROMPT_Q_SYSTEM + "\n\n" + desc + "\n<think>\n" for desc in descriptions
    ]

    logger.info(f"Generating questions for {len(prompts)} samples via LLM ...")
    outputs = llm.generate(prompts, sampling_params)

    for sample, output in zip(samples, outputs):
        questions = []
        for completion in output.outputs:
            q_text = extract_question_from_response(completion.text)
            if q_text:
                questions.append(q_text)
        if not questions:
            logger.warning(
                f"LLM returned no usable questions for {sample['configuration_id']}; "
                "keeping template fallback."
            )
        else:
            sample["query"] = questions

    return samples


# ── shared sample-record helper ───────────────────────────────────────────────

def _empty_kinematics() -> dict:
    """Lists and JSON-string fields that the original 24-sample schema has but
    which we don't compute for eval samples (only reactions matter for scoring)."""
    return {
        "x_coordinates":       [],
        "shear_force":         [],
        "bending_moment":      [],
        "slope":               [],
        "deflection":          [],
        "shear_force_info":    "{}",
        "bending_moment_info": "{}",
        "slope_info":          "{}",
        "deflection_info":     "{}",
        "points":              "{}",
        "segments":            "{}",
        "internal_loads":      "{}",
        "deflections":         "{}",
    }


def _build_record(
    *,
    configuration_id: str,
    category: str,
    L_mult: int,
    load_positions: list[float],
    load_values: list[str],
    support_positions: list[float],
    parameters: dict,
    template_q: str,
    description: str,
    answer: list[str],
    reactions_payload: dict,
    extra: dict | None = None,
) -> dict:
    record = {
        "configuration_id":  configuration_id,
        "category":          category,
        "version":           "v2",
        "load_position":     None,  # legacy column kept for schema compatibility
        "load_positions":    load_positions,
        "load_values":       load_values,
        "support_positions": support_positions,
        "parameters":        json.dumps(parameters),
        "query":             [template_q],
        "answer":             answer,
        "reactions":         json.dumps(reactions_payload),
        # Private — stripped before upload
        "_description":      description,
        **_empty_kinematics(),
    }
    if extra:
        record.update(extra)
    return record


# ── per-category generators ───────────────────────────────────────────────────

def _id_config_iter() -> Iterable[tuple[float, int]]:
    """Yield (position_fraction, magnitude) in a deterministic order."""
    for pos in HELDOUT_POSITIONS:
        for mag in P_MAGNITUDES:
            yield pos, mag


def generate_id_samples(target: int) -> list[dict]:
    """Single point load, supports at the ends. L = 9*L."""
    samples = []
    for pos, mag in _id_config_iter():
        if len(samples) >= target:
            break
        solved = solve_point_load_beam(
            L_mult=DEFAULT_LENGTH_MULT,
            supports=[("0", "pin"), (f"{DEFAULT_LENGTH_MULT}*L", "roller")],
            load_positions_frac=[pos],
            load_magnitudes=[mag],
        )
        if solved is None:
            continue

        loc_str = _fmt_loc(pos, DEFAULT_LENGTH_MULT)
        parameters = {
            "L": f"{DEFAULT_LENGTH_MULT}*L", "E": "E", "I": "I",
            "P": {"P1": {"location": loc_str, "value": f"{mag}*P"}},
            "M": {}, "Q": {},
            "R": {
                "r1": {"location": "0",                              "type": "pin"},
                "r2": {"location": f"{DEFAULT_LENGTH_MULT}*L",       "type": "roller"},
            },
        }
        desc = build_point_load_description(
            DEFAULT_LENGTH_MULT,
            [("0", "pin"), (f"{DEFAULT_LENGTH_MULT}*L", "roller")],
            [pos], [mag],
        )
        samples.append(_build_record(
            configuration_id=f"v2_id_p{int(round(pos*100))}_m{abs(mag)}",
            category="id",
            L_mult=DEFAULT_LENGTH_MULT,
            load_positions=[pos],
            load_values=[f"{mag}*P"],
            support_positions=[0.0, 1.0],
            parameters=parameters,
            template_q=build_template_question(desc),
            description=desc,
            answer=solved["answer"],
            reactions_payload={
                "header": "Exterior Reactions",
                "reactions": [
                    {"point": "0",                              "type": "Force", "value": solved["reactions_raw"][0]},
                    {"point": f"{DEFAULT_LENGTH_MULT}*L",       "type": "Force", "value": solved["reactions_raw"][1]},
                ],
            },
        ))
    return samples


def _multi_load_config_iter() -> Iterable[tuple[int, tuple[float, ...], int]]:
    """Yield (N, positions_tuple, magnitude). Each sample uses a single magnitude
    for all N loads (equal-magnitude multi-load configurations)."""
    # Pre-built position tuples drawn from HELDOUT_POSITIONS, varying N.
    pos = HELDOUT_POSITIONS
    n2_pairs = [
        (pos[0], pos[5]), (pos[1], pos[6]), (pos[2], pos[7]),
        (pos[3], pos[8]), (pos[4], pos[9]), (pos[0], pos[4]),
        (pos[2], pos[6]), (pos[1], pos[7]),
    ]
    n3_triples = [
        (pos[0], pos[3], pos[7]), (pos[1], pos[4], pos[8]),
        (pos[2], pos[5], pos[9]), (pos[0], pos[4], pos[9]),
        (pos[1], pos[5], pos[7]), (pos[2], pos[4], pos[8]),
        (pos[0], pos[5], pos[8]), (pos[1], pos[3], pos[9]),
    ]
    n4_quads = [
        (pos[0], pos[2], pos[5], pos[8]),
        (pos[1], pos[3], pos[6], pos[9]),
        (pos[0], pos[3], pos[5], pos[7]),
        (pos[2], pos[4], pos[6], pos[8]),
        (pos[1], pos[4], pos[5], pos[9]),
        (pos[0], pos[2], pos[6], pos[9]),
        (pos[1], pos[3], pos[7], pos[8]),
        (pos[0], pos[4], pos[7], pos[9]),
    ]
    # Cycle through magnitudes so each N gets a mix.
    mags = P_MAGNITUDES
    for i, tup in enumerate(n2_pairs):
        yield 2, tup, mags[i % len(mags)]
    for i, tup in enumerate(n3_triples):
        yield 3, tup, mags[i % len(mags)]
    for i, tup in enumerate(n4_quads):
        yield 4, tup, mags[i % len(mags)]


def generate_multi_load_samples(target: int) -> list[dict]:
    samples = []
    for n, positions, mag in _multi_load_config_iter():
        if len(samples) >= target:
            break
        load_positions_frac = list(positions)
        load_magnitudes = [mag] * n
        solved = solve_point_load_beam(
            L_mult=DEFAULT_LENGTH_MULT,
            supports=[("0", "pin"), (f"{DEFAULT_LENGTH_MULT}*L", "roller")],
            load_positions_frac=load_positions_frac,
            load_magnitudes=load_magnitudes,
        )
        if solved is None:
            continue

        p_dict = {
            f"P{i+1}": {"location": _fmt_loc(pos, DEFAULT_LENGTH_MULT),
                        "value": f"{mag}*P"}
            for i, pos in enumerate(load_positions_frac)
        }
        parameters = {
            "L": f"{DEFAULT_LENGTH_MULT}*L", "E": "E", "I": "I",
            "P": p_dict, "M": {}, "Q": {},
            "R": {
                "r1": {"location": "0",                        "type": "pin"},
                "r2": {"location": f"{DEFAULT_LENGTH_MULT}*L", "type": "roller"},
            },
        }
        desc = build_point_load_description(
            DEFAULT_LENGTH_MULT,
            [("0", "pin"), (f"{DEFAULT_LENGTH_MULT}*L", "roller")],
            load_positions_frac, load_magnitudes,
        )
        pos_id = "_".join(f"{int(round(p*100))}" for p in load_positions_frac)
        samples.append(_build_record(
            configuration_id=f"v2_ood_loads_N{n}_p{pos_id}_m{abs(mag)}",
            category="ood_loads",
            L_mult=DEFAULT_LENGTH_MULT,
            load_positions=list(load_positions_frac),
            load_values=[f"{mag}*P"] * n,
            support_positions=[0.0, 1.0],
            parameters=parameters,
            template_q=build_template_question(desc),
            description=desc,
            answer=solved["answer"],
            reactions_payload={
                "header": "Exterior Reactions",
                "reactions": [
                    {"point": "0",                        "type": "Force", "value": solved["reactions_raw"][0]},
                    {"point": f"{DEFAULT_LENGTH_MULT}*L", "type": "Force", "value": solved["reactions_raw"][1]},
                ],
            },
        ))
    return samples


def _varying_support_configs() -> list[tuple[str, str, str, str]]:
    """Return list of (pin_loc, roller_loc, pin_loc_frac, roller_loc_frac) tuples
    for the three OOD-supports configurations."""
    return [
        # (pin_expr, roller_expr, pin_frac_of_9L_as_str, roller_frac_of_9L_as_str)
        ("0.1*L",                                     f"{DEFAULT_LENGTH_MULT}*L", "0.1_over_9",  "1.0"),
        ("0",                                          "0.9*L",                   "0.0",         "0.1_over_9_inv"),
        ("0.1*L",                                      "0.9*L",                   "0.1_over_9",  "0.1_over_9_inv"),
    ]


def _varying_supports_load_specs() -> list[tuple[int, tuple[float, ...], int]]:
    """Six (N, positions, magnitude) load specs, yielded per support configuration.
    With TARGET_VARYING_SUPPORTS_NEW = 6 only the first configuration (pin at
    0.1L, roller at 9L) is consumed; the remaining configurations are a pool for
    larger targets. For the first configuration all load positions (x = 0.13..0.83
    of the 9L span) lie inside [pin_loc, roller_loc]."""
    pos = HELDOUT_POSITIONS
    return [
        (1, (pos[1],),                              -13),
        (1, (pos[5],),                              -7),
        (2, (pos[2], pos[6]),                       -13),
        (2, (pos[3], pos[8]),                       -19),
        (3, (pos[1], pos[4], pos[7]),               -13),
        (3, (pos[2], pos[5], pos[8]),               -7),
    ]


def generate_varying_support_samples(target: int) -> list[dict]:
    samples = []
    for pin_expr, roller_expr, pin_tag, roller_tag in _varying_support_configs():
        for n, positions, mag in _varying_supports_load_specs():
            if len(samples) >= target:
                break
            load_positions_frac = list(positions)
            load_magnitudes = [mag] * n
            solved = solve_point_load_beam(
                L_mult=DEFAULT_LENGTH_MULT,
                supports=[(pin_expr, "pin"), (roller_expr, "roller")],
                load_positions_frac=load_positions_frac,
                load_magnitudes=load_magnitudes,
            )
            if solved is None:
                continue

            p_dict = {
                f"P{i+1}": {"location": _fmt_loc(p, DEFAULT_LENGTH_MULT),
                            "value": f"{mag}*P"}
                for i, p in enumerate(load_positions_frac)
            }
            parameters = {
                "L": f"{DEFAULT_LENGTH_MULT}*L", "E": "E", "I": "I",
                "P": p_dict, "M": {}, "Q": {},
                "R": {
                    "r1": {"location": pin_expr,    "type": "pin"},
                    "r2": {"location": roller_expr, "type": "roller"},
                },
            }
            desc = build_point_load_description(
                DEFAULT_LENGTH_MULT,
                [(pin_expr, "pin"), (roller_expr, "roller")],
                load_positions_frac, load_magnitudes,
            )
            pos_id = "_".join(f"{int(round(p*100))}" for p in load_positions_frac)
            samples.append(_build_record(
                configuration_id=f"v2_ood_supports_pin{pin_tag}_rol{roller_tag}_N{n}_p{pos_id}_m{abs(mag)}",
                category="ood_supports",
                L_mult=DEFAULT_LENGTH_MULT,
                load_positions=list(load_positions_frac),
                load_values=[f"{mag}*P"] * n,
                support_positions=[
                    float(sympy.sympify(pin_expr).subs(L, 1)) / DEFAULT_LENGTH_MULT,
                    float(sympy.sympify(roller_expr).subs(L, 1)) / DEFAULT_LENGTH_MULT,
                ],
                parameters=parameters,
                template_q=build_template_question(desc),
                description=desc,
                answer=solved["answer"],
                reactions_payload={
                    "header": "Exterior Reactions",
                    "reactions": [
                        {"point": pin_expr,    "type": "Force", "value": solved["reactions_raw"][0]},
                        {"point": roller_expr, "type": "Force", "value": solved["reactions_raw"][1]},
                    ],
                },
            ))
        if len(samples) >= target:
            break
    return samples


def generate_distributed_load_samples() -> list[dict]:
    """5 left + 5 middle + 5 right. k ∈ {1,2,3}; (n_start, n_end) chosen per sub-cat."""
    sub_specs = {
        "ood_dist_left": [
            # n_start=0, varying n_end
            (1, 0, 3), (2, 0, 4), (3, 0, 5), (1, 0, 6), (2, 0, 7),
        ],
        "ood_dist_middle": [
            (1, 1, 5), (2, 2, 7), (3, 1, 7), (1, 3, 8), (2, 2, 8),
        ],
        "ood_dist_right": [
            # n_end=9 (the beam length multiplier), varying n_start
            (1, 2, 9), (2, 3, 9), (3, 4, 9), (1, 5, 9), (2, 6, 9),
        ],
    }
    samples = []
    for category, configs in sub_specs.items():
        for k, n_start, n_end in configs:
            solved = solve_dist_load_beam(k, n_start, n_end)
            if solved is None:
                continue

            parameters = {
                "L": f"{DEFAULT_LENGTH_MULT}*L", "E": "E", "I": "I",
                "P": {}, "M": {},
                "Q": {
                    "q1": {
                        "start_location": ("0" if n_start == 0
                                           else f"{n_start}*L"),
                        "end_location":   f"{n_end}*L",
                        "magnitude":      f"-{k}*P/L",
                    }
                },
                "R": {
                    "r1": {"location": "0",                        "type": "pin"},
                    "r2": {"location": f"{DEFAULT_LENGTH_MULT}*L", "type": "roller"},
                },
            }
            desc = build_dist_load_description(k, n_start, n_end)
            samples.append(_build_record(
                configuration_id=f"v2_{category}_k{k}_{int(n_start*10)}to{int(n_end*10)}",
                category=category,
                L_mult=DEFAULT_LENGTH_MULT,
                load_positions=[],
                load_values=[],
                support_positions=[0.0, 1.0],
                parameters=parameters,
                template_q=build_template_question(desc),
                description=desc,
                answer=solved["answer"],
                reactions_payload={
                    "header": "Exterior Reactions",
                    "reactions": [
                        {"point": "0",                        "type": "Force", "value": solved["reactions_raw"][0]},
                        {"point": f"{DEFAULT_LENGTH_MULT}*L", "type": "Force", "value": solved["reactions_raw"][1]},
                    ],
                },
                extra={
                    "dist_load_k":      k,
                    "dist_load_n_start": n_start,
                    "dist_load_n_end":  n_end,
                },
            ))
    return samples


def generate_length_variation_samples() -> list[dict]:
    """5 samples per L ∈ {7l, 11l, 13l}. Each set mixes N=1 and N=2 with the three
    P magnitudes; supports at ends."""
    pos = HELDOUT_POSITIONS
    per_length_specs = [
        (1, (pos[2],),               -13),
        (1, (pos[5],),               -7),
        (2, (pos[1], pos[6]),        -13),
        (2, (pos[3], pos[8]),        -19),
        (1, (pos[7],),               -19),
    ]
    samples = []
    for L_mult in LENGTH_VARIANTS:
        for n, positions, mag in per_length_specs:
            load_positions_frac = list(positions)
            load_magnitudes = [mag] * n
            solved = solve_point_load_beam(
                L_mult=L_mult,
                supports=[("0", "pin"), (f"{L_mult}*L", "roller")],
                load_positions_frac=load_positions_frac,
                load_magnitudes=load_magnitudes,
            )
            if solved is None:
                continue

            p_dict = {
                f"P{i+1}": {"location": _fmt_loc(p, L_mult),
                            "value": f"{mag}*P"}
                for i, p in enumerate(load_positions_frac)
            }
            parameters = {
                "L": f"{L_mult}*L", "E": "E", "I": "I",
                "P": p_dict, "M": {}, "Q": {},
                "R": {
                    "r1": {"location": "0",          "type": "pin"},
                    "r2": {"location": f"{L_mult}*L","type": "roller"},
                },
            }
            desc = build_point_load_description(
                L_mult, [("0", "pin"), (f"{L_mult}*L", "roller")],
                load_positions_frac, load_magnitudes,
            )
            pos_id = "_".join(f"{int(round(p*100))}" for p in load_positions_frac)
            samples.append(_build_record(
                configuration_id=f"v2_ood_length_L{L_mult}_N{n}_p{pos_id}_m{abs(mag)}",
                category="ood_length",
                L_mult=L_mult,
                load_positions=list(load_positions_frac),
                load_values=[f"{mag}*P"] * n,
                support_positions=[0.0, 1.0],
                parameters=parameters,
                template_q=build_template_question(desc),
                description=desc,
                answer=solved["answer"],
                reactions_payload={
                    "header": "Exterior Reactions",
                    "reactions": [
                        {"point": "0",          "type": "Force", "value": solved["reactions_raw"][0]},
                        {"point": f"{L_mult}*L","type": "Force", "value": solved["reactions_raw"][1]},
                    ],
                },
                extra={"length_multiplier": L_mult},
            ))
    return samples


def generate_moment_samples() -> list[dict]:
    """5 c-values × 3 fractional locations = 15 samples. Pure couple, no point load."""
    samples = []
    for c in MOMENT_C_VALUES:
        for x_m_frac in MOMENT_LOCATIONS_FRAC:
            solved = solve_moment_beam(c, x_m_frac)
            if solved is None:
                continue

            loc_str = _fmt_loc(x_m_frac, DEFAULT_LENGTH_MULT)
            parameters = {
                "L": f"{DEFAULT_LENGTH_MULT}*L", "E": "E", "I": "I",
                "P": {}, "Q": {},
                "M": {"M1": {"location": loc_str, "value": f"{c}*P*L"}},
                "R": {
                    "r1": {"location": "0",                        "type": "pin"},
                    "r2": {"location": f"{DEFAULT_LENGTH_MULT}*L", "type": "roller"},
                },
            }
            desc = build_moment_description(c, x_m_frac)
            samples.append(_build_record(
                configuration_id=f"v2_ood_moment_c{c}_loc{int(round(x_m_frac*100))}",
                category="ood_moment",
                L_mult=DEFAULT_LENGTH_MULT,
                load_positions=[],
                load_values=[],
                support_positions=[0.0, 1.0],
                parameters=parameters,
                template_q=build_template_question(desc),
                description=desc,
                answer=solved["answer"],
                reactions_payload={
                    "header": "Exterior Reactions",
                    "reactions": [
                        {"point": "0",                        "type": "Force", "value": solved["reactions_raw"][0]},
                        {"point": f"{DEFAULT_LENGTH_MULT}*L", "type": "Force", "value": solved["reactions_raw"][1]},
                    ],
                },
                extra={"moment_c": c, "moment_x_frac": x_m_frac},
            ))
    return samples


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate BeamRL eval dataset v2 (123 samples, 6 categories)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM question generation. Implies --no-upload "
                             "unless --force-upload-template is also set.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading to HuggingFace Hub")
    parser.add_argument("--force-upload-template", action="store_true",
                        help="Allow upload even when LLM is disabled (queries are templates only).")
    parser.add_argument("--out", type=str, default=None,
                        help="Also save dataset locally to this JSON file")
    parser.add_argument("--repo", type=str, default="tphage/BeamRL-EvalData-v2",
                        help="HuggingFace Hub repo name (default: tphage/BeamRL-EvalData-v2)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make the HuggingFace repo private (default: True)")
    args = parser.parse_args()

    # --no-llm implies --no-upload (safer default; prevents accidental upload of
    # a template-questions-only dataset).
    if args.no_llm and not args.force_upload_template:
        args.no_upload = True

    from datasets import Dataset, load_dataset

    # ── 1. Load original 24 samples verbatim ──────────────────────────────────
    logger.info("Loading original 24-sample eval dataset (tphage/BeamRL-EvalData) ...")
    existing_ds = None
    for split in ("train", "test"):
        try:
            existing_ds = load_dataset("tphage/BeamRL-EvalData", split=split)
            break
        except Exception:
            existing_ds = None
            continue
    if existing_ds is None:
        logger.error("Could not load tphage/BeamRL-EvalData (any split). "
                     "Run: huggingface-cli login")
        sys.exit(1)

    existing_samples = [dict(row) for row in existing_ds]
    # Unify configuration_id type with the new samples (which use semantic
    # string IDs like "v2_id_p7_m7"). The original 24 store it as int64;
    # leaving the mix triggers pyarrow type inference errors in Dataset.from_list.
    for s in existing_samples:
        if "configuration_id" in s and not isinstance(s["configuration_id"], str):
            s["configuration_id"] = f"v0_orig_{int(s['configuration_id']):02d}"
    logger.info(f"Loaded {len(existing_samples)} original samples.")

    # Original 24 are mapped onto the new category vocabulary by row index
    # (same mapping used in v0 of this generator). They are preserved bit-identically
    # in their other fields.
    CATEGORY_MAP = {
        **{i: "id"           for i in range(0,  4)},
        **{i: "ood_loads"    for i in range(4,  12)},
        **{i: "ood_supports" for i in range(12, 24)},
    }
    for idx, s in enumerate(existing_samples):
        s["category"] = CATEGORY_MAP.get(idx, "id")
        s["version"]  = "v2"
        s.setdefault("load_position", None)

    # ── 2. Freshly generate samples for each category (no GPU yet) ────────────
    logger.info(f"Generating {TARGET_ID_NEW} ID samples ...")
    new_id              = generate_id_samples(TARGET_ID_NEW)
    logger.info(f"Generating {TARGET_MULTI_LOAD_NEW} OOD-multi-load samples ...")
    new_multi           = generate_multi_load_samples(TARGET_MULTI_LOAD_NEW)
    logger.info(f"Generating {TARGET_VARYING_SUPPORTS_NEW} OOD-varying-supports samples ...")
    new_supports        = generate_varying_support_samples(TARGET_VARYING_SUPPORTS_NEW)
    logger.info("Generating OOD-distributed-load samples (5 left + 5 middle + 5 right) ...")
    new_dist            = generate_distributed_load_samples()
    logger.info("Generating OOD-length-variation samples (5 per length, 3 lengths) ...")
    new_length          = generate_length_variation_samples()
    logger.info("Generating OOD-applied-moment samples (5 c-values × 3 locations) ...")
    new_moment          = generate_moment_samples()

    new_samples = new_id + new_multi + new_supports + new_dist + new_length + new_moment
    logger.info(
        f"Freshly generated: {len(new_id)}+{len(new_multi)}+{len(new_supports)}"
        f"+{len(new_dist)}+{len(new_length)}+{len(new_moment)} = {len(new_samples)}"
    )

    # ── 3. LLM question generation for new samples only ──────────────────────
    if args.no_llm:
        logger.info("--no-llm set: keeping template questions for fresh samples. "
                    "Re-run without --no-llm to generate natural-language variants.")
    else:
        llm, sampling_params = load_llm()
        new_samples = generate_questions_with_llm(new_samples, llm, sampling_params)
        logger.info("LLM question generation complete.")

    # Strip private _description field
    for s in new_samples:
        s.pop("_description", None)

    # ── 4. Combine ───────────────────────────────────────────────────────────
    all_samples = existing_samples + new_samples
    logger.info(
        f"\nTotal: {len(existing_samples)} (original) + {len(new_samples)} (new) "
        f"= {len(all_samples)} samples."
    )

    # Align keys across all rows
    all_keys = set().union(*(s.keys() for s in all_samples))
    for s in all_samples:
        for k in all_keys:
            s.setdefault(k, None)

    dataset_v2 = Dataset.from_list(all_samples)

    # ── 5. Save locally (optional) ───────────────────────────────────────────
    if args.out:
        dataset_v2.to_json(args.out)
        logger.info(f"Saved locally: {args.out}")

    # ── 6. Upload ────────────────────────────────────────────────────────────
    if not args.no_upload:
        logger.info(f"Uploading to {args.repo} ...")
        dataset_v2.push_to_hub(args.repo, private=args.private)
        logger.info(f"Done: https://huggingface.co/datasets/{args.repo}")
    else:
        logger.info("Skipping upload.")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    from collections import Counter
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    cats = Counter(s["category"] for s in all_samples)
    for cat, count in sorted(cats.items()):
        print(f"  {cat:<22}: {count:>3} samples")
    print(f"  {'TOTAL':<22}: {len(all_samples):>3} samples")
    print("=" * 60)

    # Spot-check first new sample per category
    by_cat: dict[str, dict] = {}
    for s in new_samples:
        by_cat.setdefault(s["category"], s)
    for cat, s in sorted(by_cat.items()):
        q_preview = (s["query"][0][:120] if s["query"] else "(empty)")
        print(f"\nSpot-check ({cat}, {s['configuration_id']}):")
        print(f"  Answer:   {s['answer']}")
        print(f"  Question: {q_preview}")


if __name__ == "__main__":
    main()
