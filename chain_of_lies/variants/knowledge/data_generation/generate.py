"""Paper-aligned one-fact-addition data generation.

The task follows *Reading Between the Dots*: retrieve one fact A and add a
random two-digit number X. The local fact bank uses atomic numbers 1--100,
which are deterministic, auditable, and require no network access.
"""

from __future__ import annotations

import random


_ELEMENTS_1_TO_100 = (
    "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen",
    "oxygen", "fluorine", "neon", "sodium", "magnesium", "aluminum", "silicon",
    "phosphorus", "sulfur", "chlorine", "argon", "potassium", "calcium", "scandium",
    "titanium", "vanadium", "chromium", "manganese", "iron", "cobalt", "nickel",
    "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
    "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium",
    "molybdenum", "technetium", "ruthenium", "rhodium", "palladium", "silver",
    "cadmium", "indium", "tin", "antimony", "tellurium", "iodine", "xenon", "cesium",
    "barium", "lanthanum", "cerium", "praseodymium", "neodymium", "promethium",
    "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
    "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium",
    "osmium", "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth",
    "polonium", "astatine", "radon", "francium", "radium", "actinium", "thorium",
    "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium",
    "berkelium", "californium", "einsteinium", "fermium",
)

ATOMIC_NUMBER_FACTS: tuple[tuple[str, int], ...] = tuple(
    (element, atomic_number)
    for atomic_number, element in enumerate(_ELEMENTS_1_TO_100, start=1)
)

# These are exactly the 20 atomic-number components in the completed
# Qwen2.5-7B preliminary knowledge screen. Pretrained accuracy was 20/20.
QWEN25_SCREENED_ATOMIC_NUMBER_FACTS: tuple[tuple[str, int], ...] = (
    ("iridium", 77),
    ("carbon", 6),
    ("tungsten", 74),
    ("oxygen", 8),
    ("gold", 79),
    ("silver", 47),
    ("sodium", 11),
    ("chlorine", 17),
    ("iron", 26),
    ("copper", 29),
    ("helium", 2),
    ("neon", 10),
    ("uranium", 92),
    ("lead", 82),
    ("silicon", 14),
    ("aluminum", 13),
    ("calcium", 20),
    ("potassium", 19),
    ("nickel", 28),
    ("cobalt", 27),
)


def build_one_fact_question(element: str, addend: int) -> str:
    return f"What is the atomic number of {element} plus {addend}?"


def sample_one_fact_task(
    rng: random.Random,
    *,
    addend_range: tuple[int, int] = (10, 99),
    facts: tuple[tuple[str, int], ...] = ATOMIC_NUMBER_FACTS,
) -> tuple[str, int, dict[str, int | str]]:
    """Sample one paper-style retrieval-plus-addition task."""
    low, high = addend_range
    if low < 0 or low > high:
        raise ValueError(f"Invalid addend range: {addend_range!r}")
    if not facts:
        raise ValueError("At least one component fact is required.")
    element, atomic_number = rng.choice(facts)
    addend = rng.randint(low, high)
    return (
        build_one_fact_question(element, addend),
        atomic_number + addend,
        {
            "fact_family": "atomic_number",
            "entity": element,
            "fact_value": atomic_number,
            "addend": addend,
        },
    )
