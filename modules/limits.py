import inspect

# 全局限制与预算
MAX_TOOL_STEPS = 8
HARD_CHAR_LIMIT = 12000
SAFE_MAX_N = 200
HISTORY_CHAR_BUDGET = 160000

def clamp_text(s: str, limit: int = HARD_CHAR_LIMIT) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= limit:
        return s
    return f"[TRUNCATED to {limit} chars]\n" + s[:limit]

def call_registry(name: str, args: dict, registry):
    fn = registry[name]
    sig = inspect.signature(fn)
    filtered = {}
    for k, v in (args or {}).items():
        if k not in sig.parameters:
            continue
        if k in {"n", "topk"}:
            try:
                v = max(1, min(int(v), SAFE_MAX_N))
            except Exception:
                v = SAFE_MAX_N
        filtered[k] = v
    if "n" in sig.parameters and "n" not in filtered:
        filtered["n"] = SAFE_MAX_N
    out = fn(**filtered)
    return clamp_text(out, HARD_CHAR_LIMIT)

def prune_history(messages, budget_chars: int = HISTORY_CHAR_BUDGET):
    def total_len(msgs):
        return sum(len(str(m.get("content", ""))) for m in msgs)
    if total_len(messages) <= budget_chars:
        return messages
    kept = []
    if messages:
        kept.append(messages[0])  # system
    if len(messages) > 1:
        kept.append(messages[1])  # first user
    tail_rev = list(reversed(messages[2:]))
    for m in tail_rev:
        if m.get("role") == "tool" and len(m.get("content", "")) > 4000:
            continue
        kept.append(m)
        if total_len(kept) > budget_chars:
            kept.pop(); break
    while total_len(kept) > budget_chars and len(kept) > 3:
        idx = None
        for i in range(2, len(kept)):
            if kept[i].get("role") in {"tool", "assistant"} and len(kept[i].get("content", "")) > 2000:
                idx = i; break
        if idx is None: break
        kept.pop(idx)
    return kept