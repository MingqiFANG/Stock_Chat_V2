# modules/data_source.py
from pathlib import Path
import os
import pandas as pd
import streamlit as st
from . import shared_state as S

# 计算项目根目录：modules/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 支持用环境变量覆盖（比如在云端改路径），否则用项目内 dataset/daily_quarter.csv
DEFAULT_CSV_PATH = Path(os.environ.get(
    "DEFAULT_CSV_PATH",
    PROJECT_ROOT / "dataset" / "daily_quarter.csv"
))

@st.cache_data(show_spinner=False)
def _load_backend_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="gbk")

def sidebar_data_section():
    with st.sidebar.expander("数据集来源（CSV）", expanded=True):
        source = st.radio(
            "选择数据来源",
            ["自动（优先使用上传文件）", "仅上传文件", "仅使用后端默认"],
            index=0,
        )

        # 展示绝对路径，避免歧义；也允许手动修改
        default_path_str = st.text_input(
            "后端默认 CSV 路径",
            value=str(DEFAULT_CSV_PATH),
            help="当未上传文件或选择“仅使用后端默认”时，读取此路径",
            disabled=(source == "仅上传文件"),
        )

        uploaded = st.file_uploader(
            "上传 CSV（第一行是表头）",
            type=["csv"],
            disabled=(source == "仅使用后端默认"),
        )

        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame()

        df_loaded = None

        # 1) 优先使用上传
        if source in ("自动（优先使用上传文件）", "仅上传文件") and uploaded is not None:
            try:
                df_loaded = pd.read_csv(uploaded, low_memory=False)
            except Exception as e:
                st.error(f"读取上传 CSV 失败：{e}")
                df_loaded = None

        # 2) 否则使用后端默认
        if df_loaded is None and source in ("自动（优先使用上传文件）", "仅使用后端默认"):
            try:
                df_loaded = _load_backend_csv(Path(default_path_str))
            except FileNotFoundError:
                st.warning(f"未找到默认 CSV：{default_path_str}")
                df_loaded = pd.DataFrame()
            except Exception as e:
                st.error(f"读取默认 CSV 失败：{e}")
                df_loaded = pd.DataFrame()

        if df_loaded is not None:
            st.session_state.df = df_loaded.fillna(0)

        df = st.session_state.df
        st.caption(f"当前数据维度：{df.shape}")
        if df.empty:
            st.info("暂无数据：请上传 CSV，或确认默认路径存在可读文件。")
        else:
            st.dataframe(df.head(10), use_container_width=True)

        # 写回共享变量，供工具使用
        S.df = st.session_state.df
