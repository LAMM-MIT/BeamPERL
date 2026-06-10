"""Reward functions for GRPO training."""

import re
from typing import Optional, List, Tuple

def format_reward(completions, **kwargs):
    #"""Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, by checking if there is a single </think> tag in the completion."""

    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            #count += 1.0

            # New: add 1 if there is a non-empty \boxed{...} after the final </think>
            last_think = text.rfind("\n</think>\n")
            if last_think != -1:
                after = text[last_think + len("\n</think>\n"):]
                boxes = re.findall(r"\\boxed\s*{\s*([^}]*)\s*}", after, flags=re.DOTALL)
                if any(b.strip() for b in boxes):
                    count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    format_rewards = [count_tags(c) for c in contents]
    print("--------------------------------Format rewards--------------------------------")
    print(format_rewards)
    print("--------------------------------Format rewards--------------------------------")
    return format_rewards

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:

    def single_accuracy_reward(generation_text: str, ground_truth_terms: List[str], symbol_regex: str = _symbol_pat, tol: float = 1e-4) -> int:
        """
        Compute accuracy reward by comparing the parsed prediction extracted from
        the model output to the ground truth terms like ["0.1P", "1.9P"].

        Returns 1 if they match as multisets within tolerance, else 0.
        """
        after = _text_after_think(generation_text)
        if not after:
            # after = generation_text or ""
            after = ""

        print("Text after think:")
        after_for_print = after.replace("<|fim_pad|>", "")
        print(after_for_print)
        del after_for_print
        print("\n")

        pred = extract_coeffs_times_symbol(after, symbol_regex=symbol_regex, type="pred")
        gold = parse_ground_truth(ground_truth_terms, symbol_regex=symbol_regex)

        if isinstance(pred, list) and isinstance(gold, list):
            # Create a copy of pred that we can modify (remove matched elements)
            pred_remaining = list(pred)
            ok = True
            for g in gold:
                # print(f"Looking for {g} (type: {type(g)})")
                # Find a matching prediction and remove it to handle duplicates correctly
                matched = False
                for i, p in enumerate(pred_remaining):
                    if abs(p - g) <= tol:
                        # Found a match, remove it from remaining predictions
                        pred_remaining.pop(i)
                        matched = True
                        # print(f"Found {g}")
                        break
                if not matched:
                    # print(f"Not found {g}")
                    ok = False
                    break
        else:
            ok = False

        print("Gold:")
        print(gold)
        print("Pred:")
        print(pred)
        print("Passed:")
        print(ok)

        if ok:
            return pred, float(1)
        else:
            return pred, float(0)

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        pred, reward = single_accuracy_reward(content, sol)
        rewards.append(reward)
    
    print("--------------------------------Accuracy rewards--------------------------------")
    print(rewards)
    print("--------------------------------Accuracy rewards--------------------------------")
    return rewards

#-------------------------------- Reward utils --------------------------------#

_fraction_or_decimal = r"[+-]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
_symbol_pat = r"P"

def _text_after_think(text: str) -> str:
    """
    Return only the portion of the response after the final closing </think> tag.
    If no closing tag is present, return the original text unchanged.
    Case-insensitive match.
    """
    close_tag = re.compile(r"</\s*think\s*>", flags=re.IGNORECASE)
    matches = list(close_tag.finditer(text))
    if not matches:
        # return text
        return ""
    last = matches[-1]
    return text[last.end():]

def parse_ground_truth(gt_terms: List[str], symbol_regex: str = _symbol_pat) -> List[float]:
    """
    Convert a list of strings like ["0.1P", "1.9P"] to floats [0.1, 1.9].
    Uses the existing extractor to be consistent with parsing rules.
    """
    joined = " ; ".join(gt_terms)
    return extract_coeffs_times_symbol(joined, symbol_regex=symbol_regex, type="gold")

def _to_float(x: str) -> float:
    x = x.strip()
    if "/" in x:
        num, den = x.split("/", 1)
        return float(num) / float(den)
    return float(x)

def extract_coeffs_times_symbol(text: str, symbol_regex: str = _symbol_pat, type: str = "gold") -> List[float]:
    """
    Extract coefficients c from patterns like 'c P', 'c*P', 'c\\cdot P', '\\frac{c}{d}P', and '\\frac{cP}{d}'.
    - Looks inside \\boxed{...} first (and only falls back to full text for type == "gold").
    - Captures standalone zeros like '0' or '0.0' (no symbol attached).
    - Treats bare symbol occurrences ('P') as implicit coefficient 1.0.
    - Preserves left-to-right order, including zeros and implicit ones.
    - Accepts both \\frac and \\dfrac/\\tfrac (normalized).
    Depends on:
      - _fraction_or_decimal: regex that matches numeric or LaTeX fractional literals (no symbols).
      - _to_float: converter from a numeric/fraction string to float.
      - _symbol_pat: default symbol regex (e.g., r"P").
    """
    # 1) Pull everything inside \boxed{...} with proper brace matching
    def extract_boxed_content(text):
        results = []
        pattern = r"\\boxed\s*{"
        for match in re.finditer(pattern, text):
            start = match.end()
            brace_count = 1
            pos = start
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                content = text[start:pos-1]
                results.append(content)
        return results
    
    boxed_contents = extract_boxed_content(text)
    if type == "gold":
        hay = " ; ".join(boxed_contents) if boxed_contents else text
    else:
        hay = " ; ".join(boxed_contents) if boxed_contents else ""

    # Normalize \dfrac,\tfrac -> \frac so existing fraction logic works uniformly
    hay = re.sub(r"\\[dt]?frac", r"\\frac", hay)

    items: List[Tuple[int, int, float]] = []

    # (A) Handle \frac{ <num> [* or \cdot]? P }{ <den> }  e.g., \frac{3P}{2}
    # Also handle \frac{ P }{ <den> }  e.g., \frac{P}{6} (implicit coefficient 1)
    frac_sym_pat = rf"""
        \\frac
        \s*{{\s*({_fraction_or_decimal})\s*(?:\\cdot|\*)?\s*({symbol_regex})\s*}}
        \s*{{\s*({_fraction_or_decimal})\s*}}
    """
    for m in re.finditer(frac_sym_pat, hay, flags=re.VERBOSE):
        num_str, _sym, den_str = m.groups()
        try:
            val = _to_float(num_str) / _to_float(den_str)
            items.append((m.start(), m.end(), val))
        except Exception:
            pass
    
    # (A2) Handle \frac{ P }{ <den> }  e.g., \frac{P}{6} (implicit coefficient 1)
    frac_sym_only_pat = rf"""
        \\frac
        \s*{{\s*({symbol_regex})\s*}}
        \s*{{\s*({_fraction_or_decimal})\s*}}
    """
    for m in re.finditer(frac_sym_only_pat, hay, flags=re.VERBOSE):
        _sym, den_str = m.groups()
        try:
            val = 1.0 / _to_float(den_str)  # implicit coefficient 1
            items.append((m.start(), m.end(), val))
        except Exception:
            pass

    # (A3) Handle \frac{ <num> }{ <den> }<sym>  e.g., \frac{2}{9}P  (symbol OUTSIDE fraction)
    # This form is common in LaTeX output: the model writes the coefficient as a
    # standalone fraction and then multiplies by the symbol. Without this pattern
    # the bare-symbol fallback (D) below would assign implicit coefficient 1.0
    # to the P, silently misparsing every such answer. Critical for the
    # OOD-applied-moment category whose reactions are c/9 P with small c.
    frac_then_sym_pat = rf"""
        \\frac
        \s*{{\s*({_fraction_or_decimal})\s*}}
        \s*{{\s*({_fraction_or_decimal})\s*}}
        \s*(?:\\cdot|\*)?\s*({symbol_regex})\b
    """
    for m in re.finditer(frac_then_sym_pat, hay, flags=re.VERBOSE):
        num_str, den_str, _sym = m.groups()
        try:
            val = _to_float(num_str) / _to_float(den_str)
            items.append((m.start(), m.end(), val))
        except Exception:
            pass

    # (B) (num) [* or \cdot]? P, or (num)P  e.g., \frac{3}{2}P, 3P, 3*P
    # But avoid overlaps with fraction patterns (A, A2, A3)
    def overlaps_any(span, spans):
        s, e = span
        for s2, e2 in spans:
            if not (e <= s2 or e2 <= s):
                return True
        return False
    
    covered_spans = [(s, e) for (s, e, _) in items]
    pat = rf"({_fraction_or_decimal})\s*(?:\\cdot|\*)?\s*({symbol_regex})\b"
    for m in re.finditer(pat, hay):
        span = (m.start(), m.end())
        if not overlaps_any(span, covered_spans):
            c_str, _ = m.groups()
            try:
                val = _to_float(c_str)
                items.append((m.start(), m.end(), val))
                covered_spans.append(span)
            except Exception:
                pass

    # (C) Standalone zeros like 0 or 0.0 (not part of a larger number)
    zero_pat = r"(?<![\d.])0(?:\.0+)?(?![\d.])"
    for m in re.finditer(zero_pat, hay):
        try:
            val = _to_float(m.group(0))
            items.append((m.start(), m.end(), val))
        except Exception:
            pass

    # (D) Bare symbol => implicit coefficient 1.0, but avoid overlaps with prior matches
    # covered_spans is already updated from pattern B
    bare_sym_pat = rf"\b({symbol_regex})\b"
    for m in re.finditer(bare_sym_pat, hay):
        span = (m.start(), m.end())
        if not overlaps_any(span, covered_spans):
            items.append((m.start(), m.end(), 1.0))
            covered_spans.append(span)

    # Preserve left-to-right order
    items.sort(key=lambda t: t[0])
    coeffs = [val for _, _, val in items]
    return coeffs