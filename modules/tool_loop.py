from typing import Optional, List, Dict, Any
import json
from .heuristics import (
    need_web_for_question,
    recent_tools_look_scarce,
    is_quant_or_df_intent,
    is_pure_web_intent,
    WEB_METHOD,
    provider_web_probe_dual,
    build_provider_kwargs,
    AUTO_FORCE_FLAGS_AFTER_FAILED_PROBE,
)
from .limits import prune_history, call_registry, MAX_TOOL_STEPS, HISTORY_CHAR_BUDGET


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
        "【重要约束】任何需要返回列表/表格的工具，必须显式传入 n<=200。\n"
        "若问题涉及‘预测/明天/打分/动量/因子/筛选/TopK 等’，必须调用 df_* 工具链构造可复现实证；"
        "若用户使用别名/口语（如“收益率/涨跌幅/%/盈利/成交额/量/营收/净利”等），"
        "必须通过知识库映射到明确字段名（如 pct_chg/amount/vol/total_revenue/net_profit）。\n\n"
        "遇到诸如“明天最可能盈利超过3%”等前瞻问题时，必须在给定数据上构建可解释的打分/概率估计"
        "（如近3-5日动量、波动率、成交量变化、突破幅度、收益风险比等），并据此筛选候选股票。\n\n"
        "【使用规范】\n"
        "1) 必须通过 df_* 工具读取所需列与切片；\n"
        "2) 不得以“无法预测”为由拒绝，应明确说明这是基于历史统计的估计/概率而非确定结论；\n"
        "3) 若数据不足，需说明缺失项并给出最小所需字段清单或替代方案,必须输出答案，如果某个tool报错则使用其他tool完成用户的指令；\n"
        "4) 输出格式：先简述所用因子与窗口；随后给出Top候选表（示例列：ts_code、近5日涨幅、"
        "成交量变化%、信号分数/估计概率、简要理由）；最后附风险提示（基于历史数据的近似估计）。\n\n"
        "【重要】所有基于给定数据的非联网结论都应可由工具返回的 CSV/JSON 结果复现；尽量展示关键中间结果【中间结果必须保证数据的准确性和真实性】。"
        "【重要】当 df/KB 无法覆盖（如上市状态、实时/最新信息、公告新闻、代码映射等）时，允许使用供应商内置联网搜索。"
        "使用联网时请在答案末尾列出来源域名，并标注【WEB_OK】；未联网时禁止伪造来源。\n"
        "【重要】禁止“我主要专注于股票量化分析，对于实时汇率查询这类需要联网获取最新数据的问题，目前我无法直接为您提供准确”、“我无法在现有数据中找到”等回答，“当前数据集无法提供”"
        "【重要】任何无法基于df/KB(数据)回答的问题，都使用联网搜索"
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

    # ==== Ⅰ) 判定意图 ====
    quant_intent = is_quant_or_df_intent(user_question)
    pure_web     = is_pure_web_intent(user_question)

    # ==== Ⅱ) 联网探测（仅在纯外网时可短路返回）====
    force_flags_after_fail = False
    enable_web_initial = (provider_web == "on") or (provider_web == "auto" and need_web_for_question(user_question))
    if enable_web_initial and pure_web:
        ok, used_web, web_text = provider_web_probe_dual(client, model, user_question, provider_method)
        if ok and used_web:
            return web_text  # ✅ 纯外网→直接返回
        if AUTO_FORCE_FLAGS_AFTER_FAILED_PROBE:
            force_flags_after_fail = True
    elif enable_web_initial and not pure_web:
        # 非纯外网（混合问题）：可先探测，把结果加进上下文，但不短路
        try:
            ok, used_web, web_text = provider_web_probe_dual(client, model, user_question, provider_method)
            if ok and used_web and web_text:
                messages.append({"role": "assistant", "content": "【联网检索参考】\n" + web_text})
        except Exception:
            pass

    # ==== Ⅲ) 主循环（量化优先）====
    while True:
        messages = prune_history(messages, HISTORY_CHAR_BUDGET)

        provider_kwargs: Dict[str, Any] = {}
        if provider_web == "on":
            # 用户强制开启联网；若量化意图强烈，仍可让模型走工具（不阻止）
            provider_kwargs = build_provider_kwargs(provider_method)
        elif provider_web == "auto":
            if quant_intent:
                # ⚠ 量化/df 意图：优先不注入联网参数，避免模型走“纯文本答复”
                provider_kwargs = {}
            else:
                # 非量化：按稀缺/步数判定是否注入
                if need_web_for_question(user_question) or recent_tools_look_scarce(messages) or step >= MAX_TOOL_STEPS - 1 or force_flags_after_fail:
                    provider_kwargs = build_provider_kwargs(provider_method)

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