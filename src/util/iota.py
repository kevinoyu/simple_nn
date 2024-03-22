_IOTA: int = 0


def iota() -> str:
    global _IOTA
    _IOTA += 1
    return str(_IOTA)
