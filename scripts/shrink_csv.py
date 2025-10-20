# scripts/shrink_csv.py
import os
import pandas as pd
from pathlib import Path

SRC = Path("dataset/daily_quarter.csv")             # 原始CSV（>100MB）
OUT = Path("dataset/daily_quarter_last_month.csv")  # 精简后的CSV

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="gbk")

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"找不到文件：{SRC.resolve()}")

    df = load_csv(SRC)

    if "trade_date" not in df.columns:
        raise ValueError("CSV 中未找到列 trade_date，请确认表头。")

    # 统一为字符串 → 转 datetime（YYYYMMDD）
    s = df["trade_date"].astype(str)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if dt.isna().all():
        # 若整列转不出日期，尝试自动解析（容错）
        dt = pd.to_datetime(s, errors="coerce")

    max_dt = dt.max()
    if pd.isna(max_dt):
        raise ValueError("无法解析 trade_date 为日期，请检查数据格式。")

    # 最近 31 天窗口（相对数据里最大的日期）
    start_dt = max_dt - pd.Timedelta(days=100)
    mask = dt >= start_dt
    df_last_month = df.loc[mask].copy()

    # 保存
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_last_month.to_csv(OUT, index=False)

    # 计算文件大小（MB）
    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"已生成：{OUT}  行数：{len(df_last_month)}  大小：{size_mb:.2f} MB")
    print(f"时间范围：{start_dt.strftime('%Y-%m-%d')} ~ {max_dt.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()

# PS D:\Stock\Stock_Chat_V2> python scripts\shrink_csv.py
# 已生成：dataset\daily_quarter_last_month.csv  行数：351811  大小：39.84 MB
# 时间范围：2025-07-12 ~ 2025-10-20
# PS D:\Stock\Stock_Chat_V2> Move-Item -Path dataset\daily_quarter_last_month.csv -Destination dataset\daily_quarter.csv -Force