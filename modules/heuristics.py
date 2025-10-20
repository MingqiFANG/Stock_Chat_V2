import re

WEB_METHOD = "options"  # 默认使用 web_search_options

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
    if re.search(r"(股份|集团|科技|有限|有限公司|公司|银行|证券|微电子|工业|制造)", q) and not re.search(r"\b\d{6}\.(SH|SZ)\b", q):
        return True
    return False

def csv_empty_or_error(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if text.startswith("ERROR"):
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
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