import pandas as pd
# import numpy as np
from datetime import datetime, timedelta
import argparse

from pathlib import Path

from baostock_ops import BaostockOps

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='股票筛选工具 - 从沪深300和中证500成分股中筛选股票')
    
    # 互斥参数组：today 和 history 二选一
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--today', action='store_true', help='运行当天选股（默认）')
    group.add_argument('--history', action='store_true', help='运行历史回测')
    
    # 添加其他可能有用的参数
    parser.add_argument('--start-date', type=str, help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='回测结束日期 (YYYY-MM-DD)', 
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--top-n', type=int, default=10, help='每类选出的股票数量 (默认: 10)')
    parser.add_argument('--lookback-days', type=int, default=120, help='计算因子所需的历史天数 (默认: 120)')
    
    args = parser.parse_args()
    
    # 如果没有指定任何模式，默认使用today
    if not args.today and not args.history:
        args.today = True
    
    return args

def align_stock_to_calendar(df_stock, calendar):
    # 1. 确保 df_stock 的 index 是 DatetimeIndex
    if not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index)

    # 2. 确保 calendar 是 DatetimeIndex
    if not isinstance(calendar, pd.DatetimeIndex):
        calendar = pd.to_datetime(calendar)

    # 3. 去重（防止重复日期）
    df_stock = df_stock[~df_stock.index.duplicated(keep='first')]
    # df_stock.to_csv('stock_data.csv')
    # 4. 重新索引到统一日历
    df_aligned = df_stock.reindex(calendar)
    # df_aligned.to_csv('aligned.csv', index=False)

    # 5. 填充：价格类前向填充（限5天），成交量填0
    # price_cols = ['open', 'high', 'low', 'close']
    # if all(col in df_aligned.columns for col in price_cols):
        # df_aligned[price_cols] = df_aligned[price_cols].fillna(method="ffill", limit=5)

    for col in df_stock.columns:    #['date','code']:
        if col in ['date','code']:
            continue
        elif col in ['volume','turn']:         # 成交量数据填充0, 没有交易
            df_aligned[col] = df_aligned[col].fillna(0)
        else:      # 价格数据向前填充，价格无变化,limit放大些，考虑春节休市时间，免得麻烦
            df_aligned[col] = df_aligned[col].ffill(limit=20)
    # df_aligned.to_csv('aligned2.csv', index=False)
    # if 'volume' in df_aligned.columns:
        # df_aligned['volume'] = df_aligned['volume'].fillna(0)

    return df_aligned

class EasyProfitScreener:
    """
    从指定指数成分股中筛选未来5-20天“容易盈利”的股票
    输入: 日线数据 (含 OHLCV, turnover, peTTM, psTTM, pcfNcfTTM, pbMRQ, roe)
    输出: 每个指数内 Composite Score Top N 的股票
    """
    
    def __init__(self, 
                 index_components,
                #  benchmark_returns,
                 lookback_days=120,
                 top_n=10):
        """
        :param index_components: dict, {'CSI300': [list of stocks], 'CSI500': [...]}
        :param benchmark_returns: pd.Series, index=date, value=指数日收益率 (用于相对强度)
        :param lookback_days: 计算因子所需的历史天数
        :param top_n: 每个池子选出的股票数量
        """
        self.index_components = index_components
        # self.benchmark_returns = benchmark_returns
        self.lookback_days = lookback_days
        self.top_n = top_n
        
        # 定义要计算的因子
        self.factor_names = [
            'momentum_20',      # 动量
            'reversal_5',       # 反转
            'low_volatility_30',# 波动稳定性 (取负，所以叫 low_vol)
            'volume_price_ratio', # 量价配合
            'relative_strength' # 相对强度
        ]
    
    def calculate_factors(self, df_stock, benchmark_returns):
        """为单只股票计算所有因子"""
        df = df_stock.copy().sort_index()  # 确保按日期排序
        if len(df) < self.lookback_days:
            return pd.Series(index=self.factor_names, dtype=float)
        
        # --- 1. 动量因子: 过去20日累计收益 ---
        returns = df['close'].pct_change()
        mom_20 = returns.rolling(20).sum().iloc[-1]
        
        # --- 2. 反转因子: 近5日跌幅最大但未破位 ---
        # 我们用 -max_drawdown_5 来表示"超跌程度"
        price_5d = df['close'].iloc[-5:]
        max_dd_5 = (price_5d.min() / price_5d.iloc[0]) - 1
        reversal_5 = -max_dd_5  # 越负越好，所以取负
        
        # --- 3. 波动稳定性: 近30日波动率越低越好 ---
        vol_30 = returns.rolling(30).std().iloc[-1]
        low_vol_30 = -vol_30  # 越低越好，取负
        
        # --- 4. 量价配合: 上涨日成交量 vs 下跌日 ---
        recent_ret = returns.iloc[-5:]
        recent_vol = df['volume'].iloc[-5:]
        up_days = recent_ret > 0
        down_days = recent_ret < 0
        
        if up_days.sum() == 0 or down_days.sum() == 0:
            vol_ratio = 0.0
        else:
            up_vol_avg = recent_vol[up_days].mean()
            down_vol_avg = recent_vol[down_days].mean()
            vol_ratio = up_vol_avg / (down_vol_avg + 1e-6)
        volume_price_ratio = vol_ratio
        
        # --- 5. 相对强度: 相对于大盘的超额收益 ---
        # 需要外部传入 benchmark_returns
        stock_return_20 = df['close'].iloc[-1] / df['close'].iloc[-21] - 1
        bench_return_20 = self._get_benchmark_return(df.index[-1], benchmark_returns)
        relative_strength = stock_return_20 - bench_return_20
        
        return pd.Series({
            'momentum_20': mom_20,
            'reversal_5': reversal_5,
            'low_volatility_30': low_vol_30,
            'volume_price_ratio': volume_price_ratio,
            'relative_strength': relative_strength
        })
    
#    def _get_benchmark_return(self, date):
#        """获取指定日期对应的20日指数收益"""
#        try:
#            end_idx = self.benchmark_returns.index.get_loc(date)
#            start_idx = end_idx - 20
#            if start_idx < 0:
#                return 0.0
#            cum_return = (1 + self.benchmark_returns.iloc[start_idx:end_idx]).prod() - 1
#            return cum_return
#        except KeyError:
#            return 0.0
    
    def _get_benchmark_return(self, date, benchmark_returns):
        """获取指定日期对应的20日指数收益"""
        try:
            end_idx = benchmark_returns.index.get_loc(date)
            start_idx = end_idx - 20
            if start_idx < 0:
                return 0.0
            cum_return = (1 + benchmark_returns.iloc[start_idx:end_idx]).prod() - 1
            return cum_return
        except KeyError:
            return 0.0
    
    def cross_sectional_zscore(self, factor_df):
        """对每个因子做横截面Z-Score标准化"""
        zscore_df = factor_df.copy()
        for col in self.factor_names:
            mean_val = factor_df[col].mean()
            std_val = factor_df[col].std()
            if std_val > 1e-8:
                zscore_df[col] = (factor_df[col] - mean_val) / std_val
            else:
                zscore_df[col] = 0.0
        return zscore_df
    
    def screen(self, all_stocks_data, target_date):
        """
        主筛选函数
        :param all_stocks_data: dict, {stock_code: pd.DataFrame (daily data)}
        :param target_date: str, 'YYYY-MM-DD', 筛选该日期的数据
        :return: dict, {'CSI300': [top stocks], 'CSI500': [top stocks]}
        """
        results = {}

        target_date  = pd.to_datetime(target_date)
        
        for index_name, (stock_list, benchmark_returns) in self.index_components.items():
            print(f"\n🔍 正在处理 {index_name} 股票池...")
            
            # 收集该指数内所有股票的因子
            factor_records = []
            valid_stocks = []
            
            for stock in stock_list:
                # if stock not in all_stocks_data:
                    # continue
                df = all_stocks_data[stock]

                df_until = df[df.index <= target_date]

                if len(df_until) < self.lookback_days:
                    continue
                
                factors = self.calculate_factors(df_until, benchmark_returns)
                if factors.isna().all():
                    continue
                    
                factor_records.append(factors)
                valid_stocks.append(stock)
            
            if not factor_records:
                print(f"  ⚠️  {index_name} 无有效数据")
                results[index_name] = []
                continue
            
            # 构建因子DataFrame
            factor_df = pd.DataFrame(factor_records, index=valid_stocks)
            
            # 横截面Z-Score标准化
            zscore_df = self.cross_sectional_zscore(factor_df)
            
            # 等权合成综合得分
            zscore_df['composite_score'] = zscore_df[self.factor_names].mean(axis=1)
            
            # 选出Top N
            top_stocks = zscore_df.nlargest(self.top_n, 'composite_score')
            # 修改结果存储方式，保存(stock, score)元组
            results[index_name] = [(stock, zscore_df.loc[stock, 'composite_score']) for stock in top_stocks.index]
            
#            print(f"  ✅ {index_name} Top {self.top_n}:")
#            for i, (stock, score) in enumerate(results[index_name], 1):
#                print(f"    {i}. {stock} (Score: {score:.4f})")
        
        return results

    # --- 5. 保存结果到CSV文件 ---
def save_results(folder, top_picks, target_date):
    # 创建要保存的数据列表
    csv_rows = []
    for index, stocks in top_picks.items():
        for stock_tuple in stocks:
            if isinstance(stock_tuple, tuple):
                stock, score = stock_tuple
                csv_rows.append({
                    'date': target_date,
                    'index': index,
                    'code': stock,
                    'composite_score': score
                })
    
    # 按日期降序排列（新日期在上面）
    csv_rows = sorted(csv_rows, key=lambda x: x['date'], reverse=True)
    
    # 转换为DataFrame
    df_to_save = pd.DataFrame(csv_rows)
    
    # 定义文件路径
    csv_file_path = folder / "picks.csv"
    
    # 检查文件是否存在，以决定是否需要写入表头
    if not csv_file_path.exists():
        # 文件不存在，写入数据并包含表头
        df_to_save.to_csv(csv_file_path, index=False)
    else:
        # 文件存在，追加数据，不包含表头
        df_to_save.to_csv(csv_file_path, index=False, header=False, mode='a')
    
    print(f"\n💾 结果已保存到: {csv_file_path}")

def get_fridays(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    #判断start是否是周五，如不是，则往后推，直到找到第一个周五
    while start.weekday() != 4 and start <= end:
        start += timedelta(days=1)
    
    fridays = []
    while start <= end:
        fridays.append(start.strftime("%Y-%m-%d"))
        start += timedelta(days=7)

    return fridays

# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    print(f"运行模式: {'今日选股' if args.today else '历史回测'}")
    
    # --- 1. 准备数据 (你需要替换成你的真实数据) ---

    ops = BaostockOps(home="./")

    target_date = datetime.now().strftime('%Y-%m-%d')

    # 获取最新数据日期
    latest_data_date = ops.total_dataset.index.max()
    target_date = min(target_date, datetime.strftime(latest_data_date,"%Y-%m-%d"))

    csi300_list = pd.read_csv('csi300_list.csv')['code'].tolist()
    csi500_list = pd.read_csv('csi500_list.csv')['code'].tolist()
    csi1000_list = pd.read_csv('csi1000_list.csv')['code'].tolist()

    base_dir = ops.base_dir

    mapping = {
        "CSI300": "sh.000300",
        "CSI500": "sh.000905",
        "CSI1000": "sh.000852"
    }
    
    csi300 = pd.read_csv(base_dir / f"{mapping['CSI300']}.csv", index_col='date', parse_dates=True)
    csi500 = pd.read_csv(base_dir / f"{mapping['CSI500']}.csv", index_col='date', parse_dates=True)
    csi1000 = pd.read_csv(base_dir / f"{mapping['CSI1000']}.csv", index_col='date', parse_dates=True)
    
    # 示例：指数成分股
    index_components = {
        'CSI300': (csi300_list, csi300['pctChg']),          #['000001.SZ', '600000.SH', '601318.SH', ...],  # 你的沪深300成分股列表
        'CSI500': (csi500_list, csi500['pctChg']),          #['002475.SZ', '300750.SZ', '688981.SH', ...]   # 你的中证500成分股列表
        'CSI1000': (csi1000_list, csi1000['pctChg'])           #['002475.SZ', '300750.SZ', '688981.SH', ...]   # 你的中证1000成分股列表
    }
    

    # --- 2. 初始化筛选器 ---
    screener = EasyProfitScreener(
        index_components=index_components,
        # benchmark_returns=benchmark_returns_csi500,
        lookback_days=args.lookback_days,
        top_n=args.top_n
    )
    
    all_stocks_list = index_components['CSI300'][0] + index_components['CSI500'][0] + index_components['CSI1000'][0]

    print(f"读取截至 {target_date} 的数据...")

    # 获取所有要查询的股票代码
    
    # 使用向量化操作一次性过滤所有需要的股票数据
    mask = ops.total_dataset['code'].isin(all_stocks_list)
    filtered_data = ops.total_dataset[mask]
    
    # 按股票代码分组，创建字典
    all_stocks_data = {}
    for stock_code, group in filtered_data.groupby('code'):
        df = group.copy()
        
        # 注意：total_dataset已经将date设为索引，所以这里不再重复设置或重置
        # 确保索引是 DatetimeIndex 以符合后续计算因子的要求
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        all_stocks_data[stock_code] = df

    # 根据参数选择运行模式
    if args.today:
        # 仅运行当天选股
        print(f"执行今日选股: {target_date}")
        top_picks = screener.screen(all_stocks_data, pd.to_datetime(target_date))
        save_results(base_dir, top_picks, target_date)
    elif args.history:
        # 运行历史回测
        start_date = args.start_date or "2025-01-01"
        end_date = args.end_date
        fridays = get_fridays(start_date=start_date, end_date=end_date)
        
        print(f"执行历史回测，从 {start_date} 到 {end_date}")
        for trading_date in fridays:
            print(f"{trading_date} 筛选开始...")

            top_picks = screener.screen(all_stocks_data, pd.to_datetime(trading_date))
        
            save_results(base_dir, top_picks, trading_date)
    else:
        pass
    
    # --- 4. 输出结果 ---
#    print("\n" + "="*50)
#    print("🎯 最终选股结果 (供下周参考)")
#    print("="*50)
#    for index, stocks in top_picks.items():
#        print(f"\n{index} Top 10:")
#        for stock_tuple in stocks:
#            if isinstance(stock_tuple, tuple):  # 如果是(stock, score)元组
#                stock, score = stock_tuple
#                print(f"  - {stock} (Score: {score:.4f})")
#            else:  # 兼容旧的数据结构
#                print(f"  - {stock_tuple}")
    