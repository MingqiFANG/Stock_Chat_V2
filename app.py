# -*- coding: utf-8 -*-
import os
import streamlit as st

# 依赖：pip install -r requirements.txt
try:
    from openai import OpenAI  # DeepSeek (OpenAI 兼容)
except Exception:
    OpenAI = None

try:
    from google import genai   # Gemini 官方 SDK
except Exception:
    genai = None

from modules.data_source import sidebar_data_section
from modules.tooling import TOOLS, REGISTRY, tool_names
from modules.tool_loop import tool_loop
from modules.kb import KB

# ------------------------------
# 页面与样式
# ------------------------------
st.set_page_config(page_title="Stock Chat 2.0", page_icon="💬", layout="wide")
st.sidebar.title("设置")
font_size = st.sidebar.slider("字号(px)", 12, 18, 14, 1)
st.markdown(
    f"""
    <style>
    html, body, [data-testid='stAppViewContainer'] * {{
        font-size: {font_size}px !important;
        line-height: 1.55 !important;
    }}
    [data-testid='stChatMessage'] p {{ margin: 0.25rem 0 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Chat 2.0")

# ------------------------------
# Secrets / 环境变量
# ------------------------------
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
GEMINI_API_KEY   = st.secrets.get("GEMINI_API_KEY",   os.environ.get("GEMINI_API_KEY",   ""))

# 引擎切换
engine = st.sidebar.radio("引擎", ["DeepSeek 工具链", "Gemini 直答"], index=0, horizontal=True)

# ------------------------------
# 数据源：上传或读取后端默认 CSV
# ------------------------------
from modules.shared_state import df as GLOBAL_DF  # 仅用于类型提示（真正的 df 内部会被更新）
sidebar_data_section()  # 把 df 写入 shared_state.df

# ------------------------------
# 对话状态
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {role: user|assistant, content: str}

# 额外系统提示
system_prompt = st.sidebar.text_area("系统提示（可选）", height=80, placeholder="你是一个中文助手...")

# DeepSeek 选项
if engine == "DeepSeek 工具链":
    ds_model = st.sidebar.selectbox("DeepSeek 模型", ["deepseek-chat"], index=0)
    web_strategy = st.sidebar.selectbox("联网策略", ["auto", "on", "off"], index=0, help="auto：按问题/结果自动决定是否联网")
    web_method   = st.sidebar.selectbox("联网实现", ["options", "extra"], index=0)

# Gemini 选项
if engine == "Gemini 直答":
    gm_model = st.sidebar.selectbox("Gemini 模型", ["gemini-2.5-flash"], index=0)

# 展示历史
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# 输入并响应
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 组装近期对话供 DeepSeek 背景使用
    recent = st.session_state.messages[-6:]
    history_text = "\n".join([f"[{x['role']}] {x['content']}" for x in recent if x["role"] in {"user", "assistant"}][:-1])

    with st.chat_message("assistant"):
        box = st.empty()
        if engine == "DeepSeek 工具链":
            if OpenAI is None:
                answer = "请求失败：缺少 openai 库，请先 pip install openai"
            elif not DEEPSEEK_API_KEY:
                answer = "请求失败：未检测到 DEEPSEEK_API_KEY（本地设为环境变量或在 Streamlit Secrets 配置）"
            else:
                try:
                    ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
                    answer = tool_loop(
                        client=ds_client,
                        model=ds_model,
                        user_question=prompt,
                        provider_web=web_strategy,
                        provider_method=web_method,
                        extra_system=system_prompt or None,
                        history_text=history_text or None,
                        tools=TOOLS,
                        registry=REGISTRY,
                    )
                except Exception as e:
                    answer = f"请求失败：{e}"
            box.markdown(answer)
        else:
            # Gemini 直答（流式）
            if genai is None:
                answer = "请求失败：缺少 google-genai 库，请先 pip install google-genai"
                box.markdown(answer)
            elif not GEMINI_API_KEY:
                answer = "请求失败：未检测到 GEMINI_API_KEY（本地设为环境变量或在 Streamlit Secrets 配置）"
                box.markdown(answer)
            else:
                try:
                    gclient = genai.Client(api_key=GEMINI_API_KEY)
                    contents = []
                    if system_prompt:
                        contents.append({"role":"user", "parts":[{"text": f"[系统提示]\n{system_prompt}"}]})
                    for mm in st.session_state.messages:
                        contents.append({
                            "role": ("user" if mm["role"]=="user" else "model"),
                            "parts": [{"text": mm["content"]}]
                        })
                    acc = ""
                    for chunk in gclient.models.generate_content_stream(model=gm_model, contents=contents):
                        piece = getattr(chunk, "text", "") or ""
                        if piece:
                            acc += piece
                            box.markdown(acc)
                    answer = acc
                except Exception as e:
                    answer = f"请求失败：{e}"
                    box.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# 辅助面板
with st.expander("字段知识库（KB）预览"):
    st.json(KB)
with st.expander("已注册的工具（function calling）"):
    st.write(tool_names())