import re
import json

WEB_METHOD = "options"  # 默认使用 web_search_options

# —— 联网语义判定模式 —— #
_patterns = [
    r"是否.?上市|上.?市了吗|IPO|招股说明书|发行价|在哪.?上市",
    r"今天|当前|实时|最新|盘中|盘后|收盘|开盘|报价|股价|涨停|跌停",
    r"新闻|公告|传闻|并购|重组|停牌|复牌|澄清|互动易|董秘",
    r"官网|网站|地址|主营|业务|简介|电话",
    r"市值|PE|PB|估值|募资|分红|回购",
    r"政策|宏观|美联储|CPI|PMI|油价|黄金|汇率|美元|人民币",
]

def need_web_for_question(q: str) -> bool:
    if any(re.search(p, q) for p in _patterns):
        return True
    if re.search(r"(股份|集团|科技|有限|有限公司|公司|银行|证券|微电子|工业|制造)", q) \
       and not re.search(r"\b\d{6}\.(SH|SZ)\b", q):
        return True
    return False

# ================= 新增：量化/df 意图判定 =================
_QUANT_HINTS = [
    r"明天|次日|隔日|T\+1|预测|概率|打分|信号|回测|超额|alpha",
    r"动量|波动|成交量变化|突破|收益风险比|窗口|rolling|因子|rank|topk|筛选|分组",
    r"pct[_ ]?chg|ts_code|trade_date|vol|amount|open|close|high|low|pre_close",
    r"近\d+日|过去\d+日|[>\-+]?\d+%|涨幅|跌幅|排序|排名|胜率",
]
def is_quant_or_df_intent(q: str) -> bool:
    return any(re.search(p, q, re.IGNORECASE) for p in _QUANT_HINTS)

def is_pure_web_intent(q: str) -> bool:
    """需要联网，且不含量化/df 线索 → 纯外网意图"""
    return need_web_for_question(q) and not is_quant_or_df_intent(q)

# =========================================================

def csv_empty_or_error(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if str(text).startswith("ERROR"):
        return True
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    return len(lines) <= 1

def recent_tools_look_scarce(messages, k: int = 4) -> bool:
    cnt, scarce = 0, 0
    for m in reversed(messages):
        if m.get("role") == "tool":
            cnt += 1
            if csv_empty_or_error(m.get("content", "")):
                scarce += 1
            if cnt >= k:
                break
    return scarce >= max(2, k // 2)


# ===== 以下为联网探测与参数构造 =====

STRICT_WEB_PROOF = False
AUTO_FORCE_FLAGS_AFTER_FAILED_PROBE = True

def _json_dump_safe(obj) -> str:
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), ensure_ascii=False, default=str)
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def looks_like_web(text: str, raw_obj, strict: bool = STRICT_WEB_PROOF) -> bool:
    txt = (text or "").lower()
    raw = _json_dump_safe(raw_obj).lower()
    has_domain = re.search(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", txt) is not None
    has_trace  = any(k in raw for k in ["web_search", "trace", "source", "sources", "citation"])
    if has_domain or has_trace:
        return True
    if not strict and "【web_ok】" in txt:
        return True
    return False

def build_provider_kwargs(method: str):
    if method == "options":
        return {"web_search_options": {"enable": True, "enable_trace": True}}
    else:
        return {"extra_body": {"web_search": {"enable": True, "enable_trace": True}}}

def provider_web_probe_dual(client, model: str, question: str, prefer: str = "options"):
    methods = [prefer, "extra" if prefer == "options" else "options"]
    last = (False, False, "")
    for m in methods:
        try:
            kwargs = build_provider_kwargs(m)
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": question + "（若使用了联网检索，请在文末输出【WEB_OK】并写出1-2个来源域名）"
                }],
                tools=[],
                tool_choice="none",
                max_tokens=512,
                **kwargs
            )
            msg = resp.choices[0].message
            used_web = looks_like_web(msg.content or "", resp, STRICT_WEB_PROOF)
            if used_web:
                return True, True, msg.content or ""
            last = (True, False, msg.content or "")
        except Exception as e:
            last = (False, False, f"[provider_probe_error:{m}] {type(e).__name__}: {e}")
    return last

