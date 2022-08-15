from typing import Any, Dict, Iterable, List


def lower_dict(
    d: Dict[Any, Any], keys: bool = True, values: bool = True
) -> Dict[str, str]:
    """Returns a dictionary with lowercase strings.

    Doesn't work on nested dicts.
    """
    return {
        k.lower()
        if isinstance(k, str) and keys
        else k: v.lower()
        if isinstance(v, str) and values
        else v
        for k, v in d.items()
    }


def lower_dicts(
    dicts: Iterable[Dict[Any, Any]], keys: bool = True, values: bool = True
) -> List[Dict[str, str]]:
    return [lower_dict(d, keys=keys, values=values) for d in dicts]
