import re
import json
from typing import Dict, Any, List, Optional

KB: Dict[str, Dict[str, Any]] = {
    "ts_code": {"name": "股票代码", "desc": "证券代码，通常为6位数字+交易所后缀（SH/SZ）。", "type": "string", "unit": None, "synonyms": ["代码", "标的", "券代码", "股票", "证券代码", "股票代号"], "examples": ["600000.SH", "000001.SZ"]},
    "trade_date": {"name": "交易日期", "desc": "YYYYMMDD 交易日。", "type": "string(YYYYMMDD)", "unit": None, "synonyms": ["日期", "交易日", "时间", "自然日"], "examples": ["20250101", "20241231"]},
    "open": {"name": "开盘价", "desc": "当日第一笔成交价格。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["开盘", "开盘价"], "examples": [10.23]},
    "high": {"name": "最高价", "desc": "当日最高成交价。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["最高", "最高价"], "examples": [10.80]},
    "low": {"name": "最低价", "desc": "当日最低成交价。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["最低", "最低价"], "examples": [9.95]},
    "close": {"name": "收盘价", "desc": "当日最后一笔或收盘集合竞价价格。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["收盘", "价格", "收盘价", "股价", "价位"], "examples": [10.50]},
    "pre_close": {"name": "前收盘价", "desc": "上一交易日的收盘价。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["昨收", "昨收盘", "前收", "前一日收盘"], "examples": [10.20]},
    "change": {"name": "涨跌额", "desc": "close - pre_close。", "type": "number", "unit": "元/股（依数据源）", "synonyms": ["涨跌额", "涨跌", "价格变化", "价差"], "examples": [0.30, -0.10]},
    "pct_chg": {"name": "涨跌幅(%)", "desc": "100 * (close/pre_close - 1)。", "type": "number", "unit": "百分比（%）", "synonyms": ["涨跌幅", "收益率", "涨幅", "回报率", "收益", "%", "百分比", "盈利"], "examples": [3.03, -1.25]},
    "vol": {"name": "成交量", "desc": "成交数量（股/手）。", "type": "number", "unit": "股/手（依数据源）", "synonyms": ["成交量", "量", "成交数量", "成交手数"], "examples": [1000000]},
    "amount": {"name": "成交额", "desc": "成交量×成交均价。", "type": "number", "unit": "元（或千元/万元）", "synonyms": ["成交额", "金额", "成交金额", "成交总额", "成交价值"], "examples": [1200000]},
    "total_revenue": {"name": "营业收入", "desc": "公司营业收入（口径依数据源）。", "type": "number", "unit": "元（或万元/亿元）", "synonyms": ["营收", "收入", "营业收入"], "examples": [5000000]},
    "net_profit": {"name": "净利润", "desc": "公司净利润（口径依数据源）。", "type": "number", "unit": "元（或万元/亿元）", "synonyms": ["净利润", "利润", "净利"], "examples": [2000000]},
    "eps": {"name": "每股收益 EPS", "desc": "净利润/总股本。", "type": "number", "unit": "元/股", "synonyms": ["EPS", "每股收益", "盈利/股"], "examples": [1.25]},
    "bps": {"name": "每股净资产 BPS", "desc": "股东权益/总股本。", "type": "number", "unit": "元/股", "synonyms": ["BPS", "每股净资产", "净资产/股", "账面价值/股"], "examples": [3.80]},
}

_ALIAS2COL: Dict[str, str] = {}
for _col, _meta in KB.items():
    _aliases = set([_col] + _meta.get("synonyms", []))
    for a in _aliases:
        _ALIAS2COL[a.lower()] = _col

_DEF_WS = re.compile(r"\s+")
TOK = re.compile(r"[a-z0-9_%]+|[\u4e00-\u9fff]")

def _normalize(s: str) -> str:
    return _DEF_WS.sub("", str(s)).lower()

def _tokenize(s: str):
    return TOK.findall(_normalize(s))

def _jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

# 工具函数

def kb_all():
    return json.dumps(KB, ensure_ascii=False)

def kb_desc(cols=None):
    if not cols:
        return json.dumps(KB, ensure_ascii=False)
    out = {c: KB[c] for c in cols if c in KB}
    if not out:
        return json.dumps({"ERROR": "no valid cols"}, ensure_ascii=False)
    return json.dumps(out, ensure_ascii=False)

def kb_aliases():
    return json.dumps(_ALIAS2COL, ensure_ascii=False)

def kb_search(text: str, topk: int = 5):
    q = _normalize(text); qtoks = _tokenize(text)
    scored = []
    for alias, col in _ALIAS2COL.items():
        if alias in q:
            scored.append((1.0, col, f"命中别名:{alias}"))
    for col, meta in KB.items():
        toks = _tokenize(col)
        for syn in meta.get("synonyms", []):
            toks += _tokenize(syn)
        j = _jaccard(qtoks, toks)
        if j > 0:
            scored.append((min(0.99, j), col, "Jaccard"))
    best, reason = {}, {}
    for score, col, why in scored:
        if score > best.get(col, -1):
            best[col] = score
            reason[col] = why
    ranked = sorted(best.items(), key=lambda x: -x[1])[: max(1, int(topk))]
    result = [{"col": col, "score": float(score), "reason": reason[col], "name": KB[col]["name"]} for col, score in ranked]
    return json.dumps(result, ensure_ascii=False)

KB_TOOLS = [
  {"type":"function","function":{"name":"kb_all","description":"返回字段知识库全量定义（JSON）","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"kb_desc","description":"按列名返回知识解释（JSON）","parameters":{"type":"object","properties":{"cols":{"type":"array","items":{"type":"string"}}}}}},
  {"type":"function","function":{"name":"kb_aliases","description":"返回别名->列名映射（JSON）","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"kb_search","description":"根据自然语言关键词匹配最可能的列（JSON）","parameters":{"type":"object","properties":{"text":{"type":"string"},"topk":{"type":"integer","minimum":1,"maximum":50}},"required":["text"]}}},
]

KB_REGISTRY = {
    "kb_all": kb_all,
    "kb_desc": kb_desc,
    "kb_aliases": kb_aliases,
    "kb_search": kb_search,
}