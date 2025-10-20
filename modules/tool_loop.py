from typing import Optional, List, Dict, Any
from .heuristics import need_web_for_question, recent_tools_look_scarce, WEB_METHOD
from .limits import prune_history, call_registry, MAX_TOOL_STEPS, HISTORY_CHAR_BUDGET
import json

def tool_loop(
    client,
    model: str,
    user_question: str,
    provider_web: str = "auto",
    provider_method: str = WEB_METHOD,
    extra_system: Optional[str] = None,
    history_text: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    registry: Optional[Dict[str, Any]] = None,
) -> str:
    assert tools is not None and registry is not None, "tools/registry 不能为空"

    system_msg = (
        "你是专业股票量化员（Quant）。优先基于提供的数据集 df 与工具复现结论。"
        "在开始计算前，应优先调用【kb_search/kb_desc】完成列名与指标语义对齐；"
        "当 df/KB 无法覆盖（如上市状态、实时/最新信息、公告新闻、代码映射等）时，允许使用供应商内置联网搜索。"
        "使用联网时请在答案末尾列出来源域名（不必给长链接），并标注【WEB_OK】；未联网时禁止伪造来源。\n"
        "【重要约束】任何需要返回列表/表格的工具，必须显式传入 n<=200；若未传或超过限制，系统将自动裁剪。\n"
        "若用户使用别名/口语（如“收益率/涨跌幅/%/盈利/成交额/量/营收/净利”等），必须通过知识库映射到明确字段名。\n\n"
        "遇到诸如“明天最可能盈利超过3%”等前瞻问题时，必须在给定数据上构建可解释的打分/概率估计，并据此筛选候选股票。\n\n"
        "【使用规范】1) 必须通过 df_* 工具读取所需列与切片；2) 不得以“无法预测”为由拒绝，应说明是历史统计估计；"
        "3) 若数据不足，需说明缺失项并给出最小字段清单；4) 输出先讲因子与窗口，再给Top表，最后提示风险。\n\n"
        "【重要】所有结论都应可由工具返回的 CSV/JSON 结果复现；展示关键中间结果。"
    )
    if extra_system:
        system_msg += f"\n[补充系统提示]\n{extra_system}"

    if history_text:
        user_question = f"[对话历史]\n{history_text}\n\n[当前问题]\n{user_question}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_question},
    ]

    step = 0
    while True:
        messages = prune_history(messages, HISTORY_CHAR_BUDGET)

        enable_web = False
        if provider_web == "on":
            enable_web = True
        elif provider_web == "auto":
            enable_web = need_web_for_question(user_question) or recent_tools_look_scarce(messages)
            if not enable_web and step >= MAX_TOOL_STEPS - 1:
                enable_web = True

        provider_kwargs = {}
        if enable_web:
            if provider_method == "options":
                provider_kwargs["web_search_options"] = {"enable": True, "enable_trace": True}
            else:
                provider_kwargs["extra_body"] = {"web_search": {"enable": True, "enable_trace": True}}

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **provider_kwargs,
        )
        msg = resp.choices[0].message

        if getattr(msg, "tool_calls", None):
            step += 1
            if step > MAX_TOOL_STEPS:
                messages.append({"role": "assistant", "content": "工具调用步数已达上限，请基于已有结果直接给出结论与风险提示。"})
                continue
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}") if hasattr(tc.function, 'arguments') else {}
                result = call_registry(name, args, registry)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        return msg.content