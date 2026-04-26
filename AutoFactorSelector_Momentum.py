import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# from baostock_ops import BaostockOps, Calendar
from datetime import datetime, timedelta

from pathlib import Path
from stockdata_ops import stock_data

class AutoFactorSelector_Momentum:
    """
    动量策略专用因子选择器（适用于上涨市）
    核心逻辑：捕捉价格趋势、成交量配合、相对强度
    """
    
    def __init__(self, 
                 ic_threshold=0.02,
                 pval_threshold=0.05,
                 min_obs=200,
                 lookback_months=3):  # 动量风格切换快，只看最近3个月
        self.ic_threshold = ic_threshold
        self.pval_threshold = pval_threshold
        self.min_obs = min_obs
        self.lookback_months = lookback_months
        
        # 🎯 动量策略核心因子池
        self.base_factor_cols = [
            'mom_5',      # 过去5日涨幅（短期动量）
            'mom_10',     # 过去10日涨幅（中期动量）
            'mom_20',     # 过去20日涨幅（主流动量窗口）
            'turn_5',     # 近期换手率（确认资金流入）
            'rsi_14',     # RSI (超买超卖，用于过滤极端值)
            'volatility_20' # 波动率（高波动常伴随趋势）
        ]
        
        self.selected_factors = []
        self.weights = {}
        self.history_ic = pd.DataFrame()
    
    def fit(self, df, future_return_col='future_return', date_col='date'):
        """训练动量因子权重"""
        print(f"🔍 为动量策略筛选因子 (回溯最近 {self.lookback_months} 个月)...")
        
        # 数据预处理（同前）
        if date_col not in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=date_col)
            else:
                df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        unique_months = df['month'].unique()
        selected_months = sorted(unique_months)[-self.lookback_months:] if len(unique_months) >= self.lookback_months else sorted(unique_months)
        df_recent = df[df['month'].isin(selected_months)]
        
        monthly_ic = []
        for month, group in df_recent.groupby('month'):
            if len(group) < self.min_obs: 
                continue
            ic_dict = {'month': month}
            for col in self.base_factor_cols:
                if col not in group.columns: 
                    continue
                valid_data = group[[col, future_return_col]].dropna()
                if len(valid_data) < 50:
                    ic, pval = np.nan, 1.0
                else:
                    ic, pval = spearmanr(valid_data[col], valid_data[future_return_col])
                ic_dict[f'{col}_ic'] = ic
                ic_dict[f'{col}_pval'] = pval
            monthly_ic.append(ic_dict)
        
        if not monthly_ic:
            raise ValueError("无有效数据")
        ic_df = pd.DataFrame(monthly_ic).set_index('month')
        self.history_ic = ic_df
        
        # --- 计算平均IC并筛选 ---
        avg_ic = ic_df.filter(like='_ic').mean()
        avg_pval = ic_df.filter(like='_pval').mean()
        
        selected = []
        ic_values = []
        
        print("\n📊 动量因子 IC 报告:")
        for col in self.base_factor_cols:
            ic_key = f'{col}_ic'
            pval_key = f'{col}_pval'
            ic_val = avg_ic.get(ic_key, np.nan)
            pval = avg_pval.get(pval_key, 1.0)
            
            status = "❌ 无效"
            if not np.isnan(ic_val) and ic_val > self.ic_threshold and pval < self.pval_threshold:
                selected.append(col)
                ic_values.append(ic_val)
                status = "✅ 选中"
            print(f"  {col:<12}: IC={ic_val:.4f}, P={pval:.3f} -> {status}")
        
        # --- 权重分配：动量策略天然偏好中期动量 ---
        if selected:
            total_ic = sum(ic_values)
            raw_weights = {col: ic / total_ic for col, ic in zip(selected, ic_values)}
            
            # 🚀 强化 mom_10/mom_20：如果它们被选中，确保合计权重 > 50%
            mom_cols = [c for c in selected if c in ['mom_10', 'mom_20']]
            current_mom_weight = sum(raw_weights.get(c, 0) for c in mom_cols)
            if current_mom_weight < 0.5 and mom_cols:
                print("  ⚡ 强化中期动量权重至50%+")
                other_cols = [c for c in selected if c not in mom_cols]
                target_mom = 0.6
                scale_up = target_mom / current_mom_weight if current_mom_weight > 0 else 1.0
                for c in mom_cols:
                    raw_weights[c] *= scale_up
                remaining = 1.0 - target_mom
                current_other = sum(raw_weights.get(c, 0) for c in other_cols)
                if current_other > 0:
                    scale_down = remaining / current_other
                    for c in other_cols:
                        raw_weights[c] *= scale_down
            
            self.weights = raw_weights
            self.selected_factors = selected
        else:
            # Fallback: 经典动量组合
            print("\n⚠️ 启用默认动量权重")
            self.selected_factors = ['mom_20', 'turn_5']
            self.weights = {'mom_20': 0.7, 'turn_5': 0.3}
    
    def get_composite_score(self, df):
        score = pd.Series(0.0, index=df.index)
        for col in self.selected_factors:
            if col in df.columns:
                # 特别处理 RSI：我们希望 RSI 不要太高（避免追高），所以取负号
                if col == 'rsi_14':
                    score -= self.weights[col] * df[col]  # 越低越好（未超买）
                else:
                    score += self.weights[col] * df[col]
        return score
    
    def report(self):
        print("\n" + "="*40)
        print("📈 动量策略因子报告")
        print("="*40)
        print(f"选中因子: {self.selected_factors}")
        print("最终权重 (强调中期趋势):")
        for k, v in self.weights.items():
            sign = "-" if k == 'rsi_14' else "+"
            print(f"  - {sign}{k}: {abs(v):.4f}")
        print("="*40)
        


def calculate_momentum_factors(df):
    """为动量策略计算必要因子"""
    df = df.sort_values(['code', 'date'])
    
    # 动量因子 (过去N日收益率)
    df['mom_5'] = df.groupby('code')['close'].pct_change(5)
    df['mom_10'] = df.groupby('code')['close'].pct_change(10)
    df['mom_20'] = df.groupby('code')['close'].pct_change(20)
    
    # RSI (14日)
    def calc_rsi(group):
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # 使用apply计算RSI，并确保索引正确
    df['rsi_14'] = df.groupby('code', group_keys=False).apply(calc_rsi, include_groups=False)

    # 收益率
    df['ret_1'] = df.groupby('code')['close'].pct_change()

    # 换手率
    df['turn_5'] = df.groupby('code')['turn'].transform(lambda x: x.rolling(5).mean())

    # 波动率
    df['volatility_20'] = df.groupby('code')['ret_1'].transform(lambda x: x.rolling(20).std())

    return df

# ==================== 3. 辅助函数: 准备横截面数据 ====================
def prepare_cross_sectional_data(all_stock_daily, stock_pool, target_date):
    """
    生成 df_today —— 指定日期、指定股票池的因子横截面数据
    
    :param all_stock_daily: dict, {stock_code: daily_df}
    :param stock_pool: list, 股票代码列表 (如 CSI500 成分股)
    :param target_date: str, 'YYYY-MM-DD'
    :return: pd.DataFrame, index=stock_code, columns=factors
    """
    records = []
    for stock in stock_pool:
        if stock not in all_stock_daily:
            continue
        df = all_stock_daily[stock]
        if target_date not in df.index:
            continue
        
        # 获取截至 target_date 的历史数据（至少需要30天）
        df_hist = df.loc[:target_date].tail(30)
        if len(df_hist) < 25:
            continue
        
        # 提取最新一行的因子值
        latest_row = df_hist.iloc[-1][['mom_5', 'mom_10', 'mom_20', 'turn_5', 'rsi_14', 'volatility_20']]
        latest_row['stock'] = stock
        records.append(latest_row)
    
    if not records:
        return pd.DataFrame()
    
    df_cross = pd.DataFrame(records).set_index('stock')
    return df_cross


# 1. 判断市场状态
def get_market_regime(index_close, current_date, index_name='CSI500'):
    thresholds = {'CSI300': -0.065, 'CSI500': -0.05, 'CSI1000': -0.09}
    threshold = thresholds.get(index_name, -0.05)
    
    close_today = index_close.loc[current_date]
    close_20d_ago = index_close.shift(20).loc[current_date]
    
    if pd.isna(close_20d_ago):
        # 如果20天前的数据不存在，回退到可用的最早数据
        available_dates = index_close.loc[:current_date].index
        if len(available_dates) <= 1:
            return 'momentum' # 默认
        close_20d_ago = index_close.loc[available_dates[-21]] if len(available_dates) > 20 else index_close.iloc[0]
    
    returns_20d = close_today / close_20d_ago - 1
    if returns_20d <= threshold:
        return 'reversal'
    else:
        return 'momentum' 


# ==================== 5. 主程序 (已修复!) ====================
# CSI300/500/1000分别操作

if __name__ == "__main__":
    
    # trading_day = datetime.now().strftime("%Y-%m-%d")
    trading_day = '2026-03-31'
    
    database = stock_data(trading_day)
    while not database.calendar.is_trading_day(trading_day):
        day_in_dt = datetime.strptime(trading_day, "%Y-%m-%d")
        day_in_dt = day_in_dt - timedelta(days=1)
        trading_day = day_in_dt.strftime("%Y-%m-%d")
    

    for name, code in database.index_mapping.items():
        # 读取指数历史数据，主要使用close价
        df_stock_index = pd.read_csv(database.base_dir / name / f'{code}.csv', index_col='date')

        index_close = df_stock_index['close']
        regime = get_market_regime(index_close, trading_day, name)
        print(f"📊 当前市场状态: {regime}")
    
        if regime == 'reversal':
            # selector = AutoFactorSelector_Reversal()  # 你原有的反转策略类
            selector = None
            continue
        else:
            selector = AutoFactorSelector_Momentum()  # 新的动量策略类
   
        stock_list = database.stock_map[name]
        mask = database.dataset['code'].isin(stock_list)
        filtered_data = database.dataset[mask]
    
        all_stock_daily = dict()
        
        for stock_code, group in filtered_data.groupby('code'):
            df = group.copy()
        
            # 注意：total_dataset已经将date设为索引，所以这里不再重复设置或重置
            # 确保索引是 DatetimeIndex 以符合后续计算因子的要求
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            all_stock_daily[stock_code] = df
        
            # Step 1: 计算所有股票的动量因子
        print("🔄 正在计算动量因子...")
        for stock, df in all_stock_daily.items():
            # all_stock_daily[stock] = calculate_momentum_factors(df)
            all_stock_daily[stock] = database.compute_features(df)
            
            # 构建用于训练的面板数据 df_history
            # 这里简化处理：将所有股票的数据合并成一个长表
        history_list = []
        for stock, df in all_stock_daily.items():
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            df_copy['stock'] = stock
            history_list.append(df_copy)
        df_history = pd.concat(history_list, ignore_index=True)
    
        # 添加未来收益标签 (这里用未来5日收益作为示例)
        df_history = df_history.sort_values(['stock', 'date'])
        # df_history['future_return'] = df_history.groupby('stock')['close'].pct_change(-5).
        N = 5
        df_history['future_return'] = (
                    df_history.groupby('stock')['close'].transform(lambda x: x.shift(-N) / x - 1) # 简单收益率
                    # .apply(lambda x: np.log(x.shift(-N) / x)) # 对数收益率
        )    
        # 训练
        selector.fit(df_history, future_return_col='future_return', date_col='date')

    
        # Step 4: ⭐ 关键修复! 生成 df_today ⭐
        print("🔍 正在准备今日选股数据...")
        df_today = prepare_cross_sectional_data(all_stock_daily, stock_list, trading_day)             
        
        if df_today.empty:
            print("❌ 今日无有效数据，无法选股。")
        else:
            # Step 5: ⭐ 关键修复! 计算 score ⭐
            # get_composite_score 会自动处理 RSI 的符号问题
            df_today['score'] = selector.get_composite_score(df_today)
            
            # Step 6: 选股
            top_stocks = df_today.nlargest(10, 'score') # 演示用Top 2
            print("\n🎯 最终选股结果:")
            print(top_stocks[['score']])

#if __name__ == '__main__':
#    # 测试
#    today = '2026-03-19'
#    # today = datetime.now().strftime('%Y-%m-%d')
#
#    ops = stock_data("./")
#    for name, code in ops.index_mapping.items():
#        df_stock_index = pd.read_csv(ops.base_dir / name / f'{code}.csv', index_col='date')
#
#        index_close = df_stock_index['close']
#        regime = get_market_regime(index_close, today, name)
#    
#        stock_list = ops.stock_list[name]
#        
#        # 使用向量化操作一次性过滤所有需要的股票数据
#
#        mask = ops.total_dataset['code'].isin(stock_list)
#        df_history = ops.total_dataset[mask]
#        
#            
##        if regime == 'reversal':
##            selector = AutoFactorSelector_Reversal()  # 你原有的反转策略类
##        else:
##            selector = AutoFactorSelector_Momentum()  # 新的动量策略类
#    
#        selector = AutoFactorSelector_Momentum()  # 新的动量策略类
#        # 训练 & 选股
#        # df_history = df_panel.groupby('code').apply(ops.compute_features, include_groups=False)
#        selector.fit(df_history)
#        
#        # 4. ⭐ 关键：生成 df_today ⭐
#        df_today = prepare_cross_sectional_data(
#            all_stock_daily=all_stock_daily,
#            # stock_pool=csi500_components,  # 或合并沪深300+中证500
#            stock_pool=stock_list,  # 或合并沪深300+中证500
#            date=today,
#            factor_calculator=calc_factors
#        )
#        
#        # 5. 打分 & 选股
#        if not df_today.empty:
#            df_today['score'] = selector.get_composite_score(df_today)
#            top_stocks = df_today.nlargest(10, 'score').index.tolist()
#            print("Top 10:", top_stocks)
#        else:
#            print("⚠️ 无有效数据")
#        
#
#
#        # df_today = selector.get_composite_score(df_today)
#        # top_stocks = df_today.nlargest(10, 'score')