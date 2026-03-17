import baostock as bs
import pandas as pd

import lightgbm as lgb
# 获取某只股票日线（示例：贵州茅台）
df = ak.stock_zh_a_hist(symbol="600519", period="daily", 
                        start_date="20180101", end_date="20251231")
df.rename(columns={'日期': 'date', '收盘': 'close', '开盘': 'open',
                   '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)


def add_features(df):
    # 基础价格特征
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    
    # 移动平均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # 均线位置（是否在均线上方）
    df['price_above_ma20'] = (df['close'] > df['ma20']).astype(int)
    
    # 波动率
    df['volatility_20'] = df['ret_1'].rolling(20).std()
    
    # 成交量变化
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 相对强弱（vs 沪深300，需额外获取指数数据）
    # df['rs'] = df['ret_5'] - benchmark_ret_5
    
    return df

def create_label(df, forward_days=5, threshold=0.03):
    # 未来5日收益率
    future_ret = df['close'].shift(-forward_days) / df['close'] - 1
    # 如果收益 > 3%，标记为1（潜力股）
    df['label'] = (future_ret > threshold).astype(int)
    return df
# 假设你有全市场股票数据（多只股票拼接，带'stock_code'列）
# 这里简化为单只股票演示，实际需合并多股

df = add_features(df)
df = create_label(df)

# 特征列
feature_cols = ['open', 'high', 'low', 'volume', 'ma5', 'ma20', 'ma60',
                'price_above_ma20', 'volatility_20', 'volume_ratio']

# 去掉最后5行（标签为NaN）
df = df.dropna()

X = df[feature_cols]
y = df['label']

split_date = '2024-01-01'
X_train = X[X.index < split_date]
y_train = y[y.index < split_date]
X_val = X[X.index >= split_date]
y_val = y[y.index >= split_date]


train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',          # AUC 更关注排序能力
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)


# 对当前所有候选股票（比如中证500成分股）计算得分
stock_scores = {}
for stock in candidate_stocks:
    df_latest = get_latest_data(stock)  # 获取最近60天数据
    df_feat = add_features(df_latest)
    X_today = df_feat[feature_cols].iloc[[-1]]  # 最新一天特征
    prob = model.predict(X_today)[0]
    stock_scores[stock] = prob

# 选出概率最高的10只
selected = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("今日推荐股票:", [s[0] for s in selected])