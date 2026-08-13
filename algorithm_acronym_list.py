from functools import lru_cache

import mealpy


@lru_cache(maxsize=1)
def _available_optimizer_names():
    optimizers = mealpy.get_all_optimizers(verbose=False)
    return {
        optimizer_name.casefold(): optimizer_name
        for optimizer_name in optimizers.keys()
    }


def resolve_optimizer_name(name):
    raw_name = str(name).strip()

    candidates = [
        raw_name,
        f"Original{raw_name}",
        f"Base{raw_name}",
        f"Dev{raw_name}",
    ]

    available_names = _available_optimizer_names()

    for candidate in candidates:
        resolved_name = available_names.get(candidate.casefold())

        if resolved_name is not None:
            return resolved_name

    raise ValueError(
        "Unknown MEALPY optimizer "
        f"'{name}'. Tried: {', '.join(candidates)}"
    )
