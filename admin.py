# 主程序，判断市场形态， 如是动量的则走动量因子计算，如是反转的，则按照沪深300中证500/1000分别选择因子和权重

from stockdata_ops import stock_data
import sys
import pandas as pd
from datetime import timedelta

from momentum import Momentum
from reversal import Reversal

from scipy.stats import spearmanr

# ==================== 3. 辅助函数: 准备横截面数据 ====================
def prepare_cross_sectional_data(df_history, stock_list, target_date):
    """
    生成 df_today —— 指定日期、指定股票池的因子横截面数据
    
    :param df_history: datefram with all the tech features.
    :param stock_list: list, 股票代码列表 (如 CSI500 成分股)
    :param target_date: str, 'YYYY-MM-DD'
    :return: pd.DataFrame, index=stock_code, columns=factors
    """
    
    # 复制df_history以避免修改原始数据
    df_copy = df_history.copy()

    # 如果df_history的索引是MultiIndex，获取日期级别的值
    if isinstance(df_copy.index, pd.MultiIndex):
        df_copy = df_copy.reset_index()  # 重置索引，使日期和其他级别成为普通列
    
    # 转换target_date为datetime类型
    target_dt = pd.to_datetime(target_date)
    
    # 筛选最近30天的数据
    cutoff_date = target_dt - pd.DateOffset(days=30)
    df_recent = df_copy[df_copy['date'] >= cutoff_date]
    
    # 筛选指定股票池的数据
    df_recent = df_recent[df_recent['code'].isin(stock_list)]
    
    # 从最近30天数据中，对每个code取最新的记录
    df_latest = df_recent.groupby('code').last()
    
    # 选取所需的列
    required_cols = ['mom_5', 'mom_10', 'mom_20', 'turn_5', 'rsi_14', 'volatility_20']
    available_cols = [col for col in required_cols if col in df_latest.columns]
    df_result = df_latest[available_cols].copy()
    
    return df_result

def get_market_regime_v2(index_close, current_date, df_history, index_name='CSI500'):
    """
    V2: 结合指数走势 + 动量因子有效性 (IC) 来判断
    """
    # --- 第一步：先用老方法判断大方向 ---
    thresholds = {'CSI300': -0.065, 'CSI500': -0.05, 'CSI1000': -0.09}
    threshold = thresholds.get(index_name, -0.05)
    
    close_today = index_close.loc[current_date]
    close_20d_ago = index_close.shift(20).loc[current_date]
    returns_20d = close_today / close_20d_ago - 1 if not pd.isna(close_20d_ago) else 0
    
    is_down_market = returns_20d <= threshold
    
    # --- 第二步：计算最近1个月动量因子的平均IC ---
    # 假设 df_history 已经有 future_return 和 mom_20
    # df_recent = df_history[df_history['date'] >= pd.to_datetime(current_date) - pd.DateOffset(months=1)]
    day_before_month = pd.to_datetime(current_date) - pd.DateOffset(months=1)

    df_recent = df_history[df_history.index.get_level_values(1) >= day_before_month]
    
    valid_data = df_recent[['mom_20', 'future_return']].dropna()
    if len(valid_data) > 50:
        ic_mom20, _ = spearmanr(valid_data['mom_20'], valid_data['future_return'])
        is_momentum_valid = ic_mom20 > 0.02  # 设定一个有效阈值
    else:
        is_momentum_valid = False
    
    # --- 第三步：综合判断 ---
    if is_down_market:
        return 'reversal'  # 大盘暴跌，肯定是反转环境
    elif is_momentum_valid:
        return 'momentum'  # 大盘不差，且动量有效
    else:
        return 'reversal'  # 大盘不差，但动量无效 -> 内部是反转/轮动市

# ==================== 5. 主程序 (已修复!) ====================
# CSI300/500/1000分别操作

if __name__ == "__main__":
    
    database = stock_data()
    latest_date = database.total_dataset.index.max()

    until_date_str = sys.argv[1]
    until_date = pd.to_datetime(until_date_str) 

    while database.calendar[database.calendar['calendar_date'] == until_date_str]["is_trading_day"].all() == 0:
        until_date -= timedelta(days=1)
        until_date_str = until_date.strftime("%Y-%m-%d")
    
    if latest_date < until_date:
        print(f"⚠️ 数据尚未更新至{until_date_str}，请稍等...")
        exit()

    working_dataset = database.set_working_dataset(until_date_str)
    
    print(f"🚀 最新数据日期: {until_date_str}")

    for name, code in database.index_mapping.items():
        # 读取指数历史数据，主要使用close价
        df_stock_index = pd.read_csv(database.base_dir / name / f'{code}.csv', index_col='date')

        index_close = df_stock_index['close']
   
        stock_list = database.stock_map[name]
        feature_cols = database.feature_columns.get(name,[])
        database.set_pool(name)

        print("🔄 正在计算动量因子...")

        df_history =  database.get_predict_dataset(working_dataset, stock_list, feature_cols)
        N = 5
        df_history['future_return'] = (
                    df_history.groupby('code')['close'].transform(lambda x: x.shift(-N) / x - 1) # 简单收益率
                    # .apply(lambda x: np.log(x.shift(-N) / x)) # 对数收益率
        )    

        regime = get_market_regime_v2(index_close, until_date_str, df_history, name)
        print(f"📊 {name} 当前市场状态: {regime}")
    
        if regime == 'reversal':
            revs = Reversal(name)  # 你原有的反转策略类
            
            tops = revs.predict(df_history, until_date_str)
            print(tops)
        else:
            selector = Momentum()  # 新的动量策略类
            # 训练
            selector.fit(df_history, future_return_col='future_return', date_col='date')

            # Step 4: ⭐ 关键修复! 生成 df_today ⭐
            print("🔍 正在准备今日选股数据...")
            df_today = prepare_cross_sectional_data(df_history, stock_list, until_date_str)             
        
            if df_today.empty:
                print("❌ 今日无有效数据，无法选股。")
            else:
                # Step 5: ⭐ 关键修复! 计算 score ⭐
                # get_composite_score 会自动处理 RSI 的符号问题
                df_today['score'] = selector.get_composite_score(df_today)
            
                # Step 6: 选股
                top_stocks = df_today.nlargest(5, 'score') # 演示用Top 2
                print("\n🎯 最终选股结果:")
                print(top_stocks[['score']])