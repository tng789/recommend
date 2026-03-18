import pandas as pd

def merge_seasonal_factors(df_daily, df_seasonal, factor_cols=['dupontROE'], date_col='date', stock_col='stock_code'):
    """
    将季频因子（如 ROE）合并到日频数据中
    关键：使用 pubDate 作为生效日期，并向前填充 (ffill)
    """
    # 1. 确保日期格式正确
    df_seasonal['pubDate'] = pd.to_datetime(df_seasonal['pubDate'])
    df_daily[date_col] = pd.to_datetime(df_daily[date_col])
    
    # 2. 只保留需要的列
    keep_cols = [stock_col, 'pubDate'] + factor_cols
    df_seasonal_clean = df_seasonal[keep_cols].copy()
    
    # 3. 重命名 pubDate 为 date，以便合并
    df_seasonal_clean.rename(columns={'pubDate': 'date'}, inplace=True)
    
    # 4. 合并数据 (左连接，保留所有交易日)
    # 注意：这里是将财报数据“挂”在发布日当天
    df_merged = pd.merge(df_daily, df_seasonal_clean, on=[stock_col, 'date'], how='left')
    
    # 5. 按股票分组，对因子进行前向填充 (ffill)
    # 这意味着：4月25日发布的一季报数据，会填充到 4月25日及之后的每一天，直到下一次更新
    for col in factor_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged.groupby(stock_col)[col].fillna(method='ffill')
            
    # 6. (可选) 如果刚开始没有财报数据，可能会全是 NaN，可以根据策略决定是丢弃还是填0
    # 这里我们暂时保留 NaN，后续因子计算时会自动处理
    
    return df_merged