# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from . import shared_state as S
import re

# 统一获取全局 DataFrame（由 data_source 写入 S.df）
def _df() -> pd.DataFrame:
    return S.df if S.df is not None else pd.DataFrame()

# =========================
# 小工具：保证日期比较/最新切片的类型一致
# =========================
def _slice_latest(d: pd.DataFrame, date_col: str):
    """返回(最新交易日切片, latest_value)。用布尔索引避免 .query 与类型不一致。"""
    if date_col not in d.columns:
        return d.iloc[0:0].copy(), None
    latest = d[date_col].max()
    return d[d[date_col] == latest], latest

def _between_dates_mask(s: pd.Series, start: str, end: str):
    """
    让左右两端与 Series 保持一致的类型：
    - 若 s 是数值型：把 start/end 转 int（失败则转成字符串后按字符串比较）
    - 若 s 是字符串/对象：全转成字符串比较
    """
    if pd.api.types.is_numeric_dtype(s):
        try:
            st = int(start)
            ed = int(end)
            return (s >= st) & (s <= ed)
        except Exception:
            s2 = s.astype(str)
            return (s2 >= str(start)) & (s2 <= str(end))
    else:
        s2 = s.astype(str)
        return (s2 >= str(start)) & (s2 <= str(end))

def _retry_numeric_literal_query(expr: str) -> str:
    """
    把 '20250101' 这类纯数字字符串与比较运算的场景改成不带引号，缓解
    'Invalid comparison between dtype=int64 and str'。
    仅处理 ==, >=, <=, >, < 这些最常见运算。
    """
    expr2 = re.sub(r"(==|>=|<=|>|<)\s*'(\d+)'", r"\1 \2", expr)
    expr2 = re.sub(r"(==|>=|<=|>|<)\s*\"(\d+)\"", r"\1 \2", expr2)
    return expr2

# =========================
# 基础工具
# =========================
def df_head(n: int = 20):
    return _df().head(n).to_csv(index=False)

def df_describe():
    return json.dumps(_df().describe(include="all").to_dict(), ensure_ascii=False)

def df_select(cols: List[str]):
    safe = [c for c in cols if c in _df().columns]
    return _df()[safe].to_csv(index=False)

def df_query(expr: str, n: int = 1000):
    # 简单白名单：拒绝可疑表达式，避免注入
    if any(x in expr for x in ["__", "import", "os.", "pd.", "eval", "exec"]):
        return "ERROR: expr contains unsafe tokens."
    try:
        out = _df().query(expr)
        return out.head(n).to_csv(index=False)
    except Exception as e:
        # 针对 int64 vs str 的典型比较错误，自动重写一次表达式后重试
        msg = str(e)
        if "Invalid comparison between dtype=int64 and str" in msg or "TypeError" in msg:
            expr2 = _retry_numeric_literal_query(expr)
            try:
                out = _df().query(expr2)
                return out.head(n).to_csv(index=False)
            except Exception:
                return f"ERROR: {e}"
        return f"ERROR: {e}"

# 对应的 function-calling 描述
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "df_head",
            "description": "返回数据集前N行（CSV）",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 2000}},
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "df_describe",
            "description": "返回数值列统计摘要（JSON）",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "df_select",
            "description": "按列名选择若干列并返回CSV",
            "parameters": {
                "type": "object",
                "properties": {"cols": {"type": "array", "items": {"type": "string"}}},
                "required": ["cols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "df_query",
            "description": "用 pandas.query 过滤并返回前N行CSV（只读）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string"},
                    "n": {"type": "integer", "minimum": 1, "maximum": 5000},
                },
                "required": ["expr"],
            },
        },
    },
]

# 工具实现注册表
REGISTRY: Dict[str, Any] = {
    "df_head": df_head,
    "df_describe": df_describe,
    "df_select": df_select,
    "df_query": df_query,
}

# =========================
# 辅助函数
# =========================
def _safe_cols(cols: List[str]) -> List[str]:
    return [c for c in (cols or []) if c in _df().columns]

def _as_bool_list(val, n: int) -> List[bool]:
    # 允许 ascending 传单个 bool 或列表
    if isinstance(val, list):
        if len(val) != n:
            return [bool(val[0])] * n
        return [bool(x) for x in val]
    return [bool(val)] * n

# =========================
# 信息/概览类
# =========================
def df_columns():
    return json.dumps(list(_df().columns), ensure_ascii=False)

def df_dtypes():
    d = _df()
    return json.dumps({c: str(t) for c, t in zip(d.columns, d.dtypes)}, ensure_ascii=False)

def df_info_lite():
    d = _df()
    return json.dumps(
        {"rows": int(d.shape[0]), "cols": int(d.shape[1]), "mem_bytes": int(d.memory_usage(index=True).sum())}
    )

def df_nulls():
    return json.dumps(_df().isna().sum().to_dict(), ensure_ascii=False)

# =========================
# 取尾/抽样/排序/计数
# =========================
def df_tail(n: int = 20):
    return _df().tail(n).to_csv(index=False)

def df_sample(n: int = 100, random_state: int = 0):
    d = _df()
    n = int(min(n, len(d)))
    return d.sample(n=n, random_state=random_state).to_csv(index=False)

def df_sort(cols: List[str], ascending=True, n: int = 100):
    d = _df()
    cols = _safe_cols(cols)
    if not cols:
        return "ERROR: no valid columns"
    asc = _as_bool_list(ascending, len(cols))
    out = d.sort_values(cols, ascending=asc, kind="mergesort")
    return out.head(n).to_csv(index=False)

def df_value_counts(col: str, n: int = 50, normalize: bool = False, dropna: bool = True):
    d = _df()
    if col not in d.columns:
        return "ERROR: invalid column"
    vc = d[col].value_counts(normalize=normalize, dropna=dropna).head(n).reset_index()
    vc.columns = [col, "count" if not normalize else "ratio"]
    return vc.to_csv(index=False)

def df_unique(col: str, n: int = 100):
    d = _df()
    if col not in d.columns:
        return "ERROR: invalid column"
    uniq = pd.Series(d[col].unique()[:n], name=col).to_frame()
    return uniq.to_csv(index=False)

# =========================
# 日期相关（默认 trade_date 为 YYYYMMDD 字符串）
# =========================
def df_latest_date(date_col: str = "trade_date"):
    d = _df()
    if date_col not in d.columns:
        return "ERROR: invalid date_col"
    latest = d[date_col].max()
    return json.dumps({"latest": None if pd.isna(latest) else str(latest)})

def df_between_dates(start: str, end: str, date_col: str = "trade_date", n: int = 5000):
    d = _df()
    if date_col not in d.columns:
        return "ERROR: invalid date_col"
    mask = _between_dates_mask(d[date_col], start, end)
    return d.loc[mask].head(n).to_csv(index=False)

def df_at_latest_date(date_col: str = "trade_date", cols: Optional[List[str]] = None, n: int = 5000):
    d = _df()
    if date_col not in d.columns:
        return "ERROR: invalid date_col"
    latest = d[date_col].max()
    if pd.isna(latest):
        return "ERROR: no latest date"
    subset = d[d[date_col] == latest]
    if cols:
        cols = _safe_cols(cols)
        if cols:
            subset = subset[cols]
    return subset.head(n).to_csv(index=False)

# =========================
# 分组聚合
# =========================
def df_groupby_agg(
    group_cols: List[str],
    agg_spec: Dict[str, Any],
    sort_by: Optional[List[str]] = None,
    ascending: bool = False,
    n: int = 500,
):
    d = _df()
    group_cols = _safe_cols(group_cols)
    if not group_cols:
        return "ERROR: no valid group cols"

    safe_keys = _safe_cols(list(agg_spec.keys()))
    if not safe_keys:
        return "ERROR: no valid agg columns"

    safe_spec = {k: agg_spec[k] for k in safe_keys}
    out = d.groupby(group_cols, dropna=False).agg(safe_spec)
    # 扁平化多级列
    out.columns = [
        f"{a}_{b}" if isinstance(b, str) else str(a) for a, b in out.columns.to_flat_index()
    ]
    out = out.reset_index()

    if sort_by:
        sort_by = [c for c in sort_by if c in out.columns]
        if sort_by:
            out = out.sort_values(sort_by, ascending=ascending)

    return out.head(n).to_csv(index=False)

# =========================
# 常用因子/特征
# =========================
def df_pct_change(col: str, periods: int = 1, by: str = "ts_code", n: int = 5000):
    d = _df()
    if col not in d.columns or (by and by not in d.columns):
        return "ERROR: invalid column"
    s = d.groupby(by)[col].pct_change(periods)
    out = d.assign(**{f"{col}_pctchg_{periods}": s})
    return out.head(n).to_csv(index=False)

def df_rolling_mean(col: str, window: int = 5, min_periods: Optional[int] = None, by: str = "ts_code", n: int = 5000):
    d = _df()
    if col not in d.columns or (by and by not in d.columns):
        return "ERROR: invalid column"
    if min_periods is None:
        min_periods = window
    s = d.groupby(by)[col].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    out = d.assign(**{f"{col}_ma_{window}": s})
    return out.head(n).to_csv(index=False)

def df_factor_momentum(pct_col: str = "pct_chg", window: int = 5, by: str = "ts_code",
                       date_col: str = "trade_date", topk: int = 20):
    d = _df()
    if pct_col not in d.columns or by not in d.columns or date_col not in d.columns:
        return "ERROR: invalid columns"
    retw = d.groupby(by)[pct_col].rolling(window, min_periods=max(2, window // 2)).mean().reset_index(level=0, drop=True)
    temp = d.assign(ret_w=retw)
    latest_slice, latest = _slice_latest(temp, date_col)
    if latest is None or latest_slice.empty:
        return "ERROR: no latest date"
    out = (
        latest_slice.dropna(subset=["ret_w"])
        .assign(score=lambda x: x["ret_w"])
        .sort_values("score", ascending=False)
        .head(topk)[[by, date_col, "score", "ret_w"]]
    )
    return out.to_csv(index=False)

def df_factor_volatility(pct_col: str = "pct_chg", window: int = 20, by: str = "ts_code",
                         date_col: str = "trade_date", topk: int = 20):
    d = _df()
    if pct_col not in d.columns or by not in d.columns or date_col not in d.columns:
        return "ERROR: invalid columns"
    sigma = (
        d.groupby(by)[pct_col]
        .rolling(window, min_periods=max(3, window // 3)).std()
        .reset_index(level=0, drop=True)
    )
    temp = d.assign(sigma=sigma)
    latest_slice, latest = _slice_latest(temp, date_col)
    if latest is None or latest_slice.empty:
        return "ERROR: no latest date"
    out = (
        latest_slice.dropna(subset=["sigma"])
        .assign(score=lambda x: -x["sigma"])  # 波动越小越好
        .sort_values("score", ascending=False)
        .head(topk)[[by, date_col, "score", "sigma"]]
    )
    return out.to_csv(index=False)

def df_factor_volume(vol_col: str = "vol", window: int = 5, by: str = "ts_code",
                     date_col: str = "trade_date", topk: int = 20):
    d = _df()
    if vol_col not in d.columns or by not in d.columns or date_col not in d.columns:
        return "ERROR: invalid columns"
    volchg = d.groupby(by)[vol_col].pct_change(window)
    temp = d.assign(vol_chg_w=volchg)
    latest_slice, latest = _slice_latest(temp, date_col)
    if latest is None or latest_slice.empty:
        return "ERROR: no latest date"
    out = (
        latest_slice.dropna(subset=["vol_chg_w"])
        .assign(score=lambda x: x["vol_chg_w"])
        .sort_values("score", ascending=False)
        .head(topk)[[by, date_col, "score", "vol_chg_w"]]
    )
    return out.to_csv(index=False)

def df_join_factors(weights_json, date_col: str = "trade_date", by: str = "ts_code", topk: int = 20):
    """
    按权重线性合成已有因子列（例如 {"ret_w":0.6,"vol_chg_w":0.3,"sigma":-0.1}）。
    要求这些列已经通过前面工具生成并存在 df 中。
    """
    d = _df()
    try:
        weights = json.loads(weights_json) if isinstance(weights_json, str) else dict(weights_json)
    except Exception:
        return "ERROR: invalid weights_json"

    miss = [k for k in weights.keys() if k not in d.columns]
    if miss:
        return f"ERROR: missing factor columns: {miss}"

    score = None
    for k, w in weights.items():
        score = (w * d[k]) if score is None else (score + w * d[k])
    tmp = d.assign(score=score)

    latest_slice, latest = _slice_latest(tmp, date_col)
    if latest is None or latest_slice.empty:
        return "ERROR: no latest date"
    out = (
        latest_slice.dropna(subset=["score"])
        .sort_values("score", ascending=False)
        .head(topk)[[by, date_col, "score"] + list(weights.keys())]
    )
    return out.to_csv(index=False)

def df_corr(cols: Optional[List[str]] = None):
    d = _df()
    cols = [c for c in (cols or list(d.select_dtypes(include=[np.number]).columns)) if c in d.columns]
    if len(cols) < 2:
        return "ERROR: need >=2 numeric columns"
    return d[cols].corr().to_csv()

# =========================
# 把扩展工具注册到 TOOLS / REGISTRY
# =========================
MORE_TOOLS = [
    {"type": "function", "function": {"name": "df_columns", "description": "返回所有列名（JSON）",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "df_dtypes", "description": "返回各列数据类型（JSON）",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "df_info_lite", "description": "返回规模/内存等元信息（JSON）",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "df_nulls", "description": "各列缺失值统计（JSON）",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "df_tail", "description": "返回数据集尾部N行（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 2000}}}}},
    {"type": "function", "function": {"name": "df_sample", "description": "随机抽样N行（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 10000},
                                                                    "random_state": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "df_sort", "description": "按列排序并返回前N行（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"cols": {"type": "array", "items": {"type": "string"}},
                                                                    "ascending": {"oneOf": [{"type": "boolean"},
                                                                                            {"type": "array",
                                                                                             "items": {"type": "boolean"}}]},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 10000}},
                                                     "required": ["cols"]}}},
    {"type": "function", "function": {"name": "df_value_counts", "description": "分类列频数（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"col": {"type": "string"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 10000},
                                                                    "normalize": {"type": "boolean"},
                                                                    "dropna": {"type": "boolean"}},
                                                     "required": ["col"]}}},
    {"type": "function", "function": {"name": "df_unique", "description": "列唯一值（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"col": {"type": "string"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 20000}},
                                                     "required": ["col"]}}},
    {"type": "function", "function": {"name": "df_latest_date", "description": "返回最新交易日（JSON）",
                                      "parameters": {"type": "object",
                                                     "properties": {"date_col": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "df_between_dates", "description": "日期范围过滤（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"start": {"type": "string"},
                                                                    "end": {"type": "string"},
                                                                    "date_col": {"type": "string"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 100000}},
                                                     "required": ["start", "end"]}}},
    {"type": "function", "function": {"name": "df_at_latest_date", "description": "最新交易日切片（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"date_col": {"type": "string"},
                                                                    "cols": {"type": "array", "items": {"type": "string"}},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 100000}}}}},
    {"type": "function", "function": {"name": "df_groupby_agg", "description": "分组聚合（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"group_cols": {"type": "array", "items": {"type": "string"}},
                                                                    "agg_spec": {"type": "object"},
                                                                    "sort_by": {"type": "array", "items": {"type": "string"}},
                                                                    "ascending": {"type": "boolean"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 100000}},
                                                     "required": ["group_cols", "agg_spec"]}}},
    {"type": "function", "function": {"name": "df_pct_change", "description": "按组计算百分比变化（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"col": {"type": "string"},
                                                                    "periods": {"type": "integer"},
                                                                    "by": {"type": "string"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 100000}},
                                                     "required": ["col"]}}},
    {"type": "function", "function": {"name": "df_rolling_mean", "description": "按组滚动均值（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"col": {"type": "string"},
                                                                    "window": {"type": "integer"},
                                                                    "min_periods": {"type": "integer"},
                                                                    "by": {"type": "string"},
                                                                    "n": {"type": "integer", "minimum": 1, "maximum": 100000}},
                                                     "required": ["col", "window"]}}},
    {"type": "function", "function": {"name": "df_factor_momentum", "description": "动量因子TopK（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"pct_col": {"type": "string"},
                                                                    "window": {"type": "integer"},
                                                                    "by": {"type": "string"},
                                                                    "date_col": {"type": "string"},
                                                                    "topk": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "df_factor_volatility", "description": "波动率因子TopK（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"pct_col": {"type": "string"},
                                                                    "window": {"type": "integer"},
                                                                    "by": {"type": "string"},
                                                                    "date_col": {"type": "string"},
                                                                    "topk": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "df_factor_volume", "description": "成交量变化因子TopK（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"vol_col": {"type": "string"},
                                                                    "window": {"type": "integer"},
                                                                    "by": {"type": "string"},
                                                                    "date_col": {"type": "string"},
                                                                    "topk": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "df_join_factors", "description": "权重线性合成因子并在最新交易日选TopK（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"weights_json": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                                                                    "date_col": {"type": "string"},
                                                                    "by": {"type": "string"},
                                                                    "topk": {"type": "integer"}},
                                                     "required": ["weights_json"]}}},
    {"type": "function", "function": {"name": "df_corr", "description": "相关系数矩阵（CSV）",
                                      "parameters": {"type": "object",
                                                     "properties": {"cols": {"type": "array", "items": {"type": "string"}}}}}},
]

TOOLS += MORE_TOOLS

REGISTRY.update({
    "df_columns": df_columns,
    "df_dtypes": df_dtypes,
    "df_info_lite": df_info_lite,
    "df_nulls": df_nulls,
    "df_tail": df_tail,
    "df_sample": df_sample,
    "df_sort": df_sort,
    "df_value_counts": df_value_counts,
    "df_unique": df_unique,
    "df_latest_date": df_latest_date,
    "df_between_dates": df_between_dates,
    "df_at_latest_date": df_at_latest_date,
    "df_groupby_agg": df_groupby_agg,
    "df_pct_change": df_pct_change,
    "df_rolling_mean": df_rolling_mean,
    "df_factor_momentum": df_factor_momentum,
    "df_factor_volatility": df_factor_volatility,
    "df_factor_volume": df_factor_volume,
    "df_join_factors": df_join_factors,
    "df_corr": df_corr,
})
