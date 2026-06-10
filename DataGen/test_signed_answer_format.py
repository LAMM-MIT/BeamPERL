#!/usr/bin/env python3
"""
Standalone test for signed-coefficient answer handling.

Why this test exists
--------------------
The OOD-applied-moment category in eval v2 produces reactions with opposite signs
(e.g. R_pin = +0.222222P and R_roller = -0.222222P for a centered pure couple).
Both the generator's `format_answer_str()` and the trainer's `accuracy_reward()`
must round-trip those signs correctly. If they don't, every moment-category
sample will be marked wrong regardless of the model's actual output, polluting
the eval set with false negatives. This test catches that class of bug *before*
GPU time is spent on data generation or evaluation.

Run from anywhere:
    python test_signed_answer_format.py

Exits 0 on success, non-zero on failure (suitable for CI / pre-flight checks).
"""

import os
import sys

# Path setup: this script lives at beamperl/DataGen/test_signed_answer_format.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                # DataGen
BEAMRL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "BeamRL")) # BeamRL
sys.path.insert(0, SCRIPT_DIR)         # so we can import generate_eval_v2
sys.path.insert(0, BEAMRL_DIR)         # so we can import beamrl.rewards

# Generator side: signed string formatting
from generate_eval_v2 import format_answer_str

# Trainer side: reward function used at training and eval time
from beamrl.rewards import accuracy_reward


def _check(condition: bool, msg: str) -> None:
    """Tiny inline assert with a clear failure message."""
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"OK  : {msg}")


def test_format_emits_signed_string() -> None:
    """Sanity-check the generator's string format for negative coefficients."""
    _check(format_answer_str(0.2222222) == "0.222222P",
           "format_answer_str(+0.2222222) == '0.222222P'")
    _check(format_answer_str(-0.2222222) == "-0.222222P",
           "format_answer_str(-0.2222222) == '-0.222222P'")
    _check(format_answer_str(-3.3333333) == "-3.333333P",
           "format_answer_str(-3.3333333) == '-3.333333P'")


def _fake_completion(boxed_answers: list[str]) -> list[dict]:
    """Build the (single-completion) structure that accuracy_reward expects.

    The reward function reads completion[0]["content"]. The text after the final
    </think> tag is parsed; only coefficients inside \\boxed{...} are extracted
    when type="pred" (matching the format reward's requirement that the answer
    appear inside \\boxed{}). ``boxed_answers`` is therefore a list of strings
    that go inside each \\boxed{}, in the order the model would emit them.
    """
    inner = "\n".join(f"R reaction: \\boxed{{{a}}}" for a in boxed_answers)
    text = f"<think>working...</think>\n\nFinal answer:\n{inner}\n"
    return [{"content": text}]


def test_accuracy_reward_matches_signed_multiset() -> None:
    """The reward must give 1.0 when the model's signed coefficients match the
    ground truth as a multiset, and 0.0 otherwise."""
    # Ground truth: one positive, one negative reaction (typical pure-couple case).
    gt = ["0.222222P", "-0.222222P"]

    # Case A: model emits matching values inside two \boxed{}, same order as GT
    rewards = accuracy_reward(
        completions=[_fake_completion(["0.222222P", "-0.222222P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "matching signed reactions (same order) → 1.0")

    # Case B: order swapped — multiset matching should still succeed
    rewards = accuracy_reward(
        completions=[_fake_completion(["-0.222222P", "0.222222P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "matching signed reactions (swapped order) → 1.0")

    # Case C: BOTH signs flipped → multiset becomes {-, +} which equals {+, -}.
    # Multisets are sign-aware here, so this still matches.
    rewards = accuracy_reward(
        completions=[_fake_completion(["-0.222222P", "0.222222P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "sign-swap on both reactions (same multiset) → 1.0")

    # Case D: BOTH reactions have the same sign → wrong multiset → fail
    rewards = accuracy_reward(
        completions=[_fake_completion(["0.222222P", "0.222222P"])],
        solution=[gt],
    )
    _check(rewards == [0.0],
           "wrong multiset (both positive) → 0.0")

    # Case E: only one of two reactions has wrong sign → must fail
    rewards = accuracy_reward(
        completions=[_fake_completion(["0.222222P", "0.111111P"])],
        solution=[gt],
    )
    _check(rewards == [0.0],
           "one of two reactions wrong → 0.0")

    # Case F: tolerance check — 4 decimal places vs. 6 → delta > 1e-4 → fails
    rewards = accuracy_reward(
        completions=[_fake_completion(["0.22P", "-0.22P"])],
        solution=[gt],
    )
    _check(rewards == [0.0],
           "rounded prediction outside 1e-4 tolerance → 0.0")

    # Case G: tolerance check — within 1e-4 → succeeds
    rewards = accuracy_reward(
        completions=[_fake_completion(["0.222200P", "-0.222200P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "rounded prediction inside 1e-4 tolerance → 1.0")

    # Case H: \frac with P inside the numerator (\frac{2P}{9}) is parsed correctly.
    # This is the form the parser supports for fractional answers; documenting it
    # here so future changes don't break it.
    rewards = accuracy_reward(
        completions=[_fake_completion(["\\frac{2P}{9}", "\\frac{-2P}{9}"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "\\frac{cP}{d} form parsed correctly → 1.0")

    # Case I: \frac{c}{d}P form (P OUTSIDE the fraction) — fixed by adding
    # pattern (A3) in extract_coeffs_times_symbol. Before the fix this would
    # silently misparse to [1.0, 1.0] (bare-P fallback) and yield a false
    # negative on every moment-category sample. Regression-test the fix.
    rewards = accuracy_reward(
        completions=[_fake_completion(["\\frac{2}{9}P", "\\frac{-2}{9}P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "\\frac{c}{d}P form (P outside fraction) parsed correctly → 1.0")

    # Case J: \frac with \cdot before the symbol — same fix should cover this.
    rewards = accuracy_reward(
        completions=[_fake_completion(["\\frac{2}{9}\\cdot P", "\\frac{-2}{9}\\cdot P"])],
        solution=[gt],
    )
    _check(rewards == [1.0],
           "\\frac{c}{d}\\cdot P form parsed correctly → 1.0")


if __name__ == "__main__":
    print("=" * 60)
    print("Signed-answer format test for BeamPERL eval v2 (moment category)")
    print("=" * 60)
    test_format_emits_signed_string()
    test_accuracy_reward_matches_signed_multiset()
    print("=" * 60)
    print("ALL CHECKS PASSED")
