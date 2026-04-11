'''用于开发调试， 手工运行'''
import pandas as pd
from selector import Selector
from datetime import datetime   

from baostock_ops import BaostockOps

def should_activate_strategy(index_close_series, current_date, lookback_days=20, threshold=-0.05):
    """
    判断是否激活反转策略
    :param index_close_series: 中证500每日收盘价 (pd.Series, index=date)
    :param current_date: 当前日期 (str or datetime)
    :param lookback_days: 回看天数
    :param threshold: 跌幅阈值 (默认-5%)
    :return: bool
    """
    # 确保数据按日期排序
    index_close_series = index_close_series.sort_index()
    
    # 找到当前日期的位置
    if current_date not in index_close_series.index:
        raise ValueError(f"日期 {current_date} 不在指数数据中")
    
    current_idx = index_close_series.index.get_loc(current_date)
    if current_idx < lookback_days:
        return False  # 数据不足，不交易
    
    start_idx = current_idx - lookback_days
    start_price = index_close_series.iloc[start_idx]
    current_price = index_close_series.iloc[current_idx]
    
    cumulative_return = current_price / start_price - 1
    print(cumulative_return, threshold)
    return cumulative_return <= threshold
def main():

    program = Selector()

    # today = datetime.now().strftime("%Y-%m-%d")
    
    stock_ops = BaostockOps()
    stock_ops.update_dataset()
    stock_ops.update_index()
    
    # ch = input("press enter to continue.....")
    for stock_pool in (["zz500", "zz1000"]):
        idx = stock_ops.index_mapping[stock_pool]
        df_stockindex = pd.read_csv(stock_ops.working_dir / f"{idx}.csv",index_col="date")
        last_day = df_stockindex.index.max()
        activate = should_activate_strategy(df_stockindex['close'], last_day)
        if activate:
            df = program.make_dataframe(stock_pool=stock_pool)
            program.predict(stock_pool=stock_pool, df_predict=df, val_end=last_day)
        else:
            print("过去20个交易日累计跌幅 未超 5%, 不激活选股策略")

if __name__ == "__main__":
    main()