import pandas as pd
import numpy as np
import baostock as bs
# import akshare as ak
from datetime import datetime, timedelta
from pathlib import Path    
import lightgbm as lgb
from tqdm import tqdm
#---------------------------
# 示例：对齐多只股票到统一交易日历
#---------------------------
# 2. 对每只股票对齐
def align_stock_to_calendar(df_stock, calendar):
    # 1. 确保 df_stock 的 index 是 DatetimeIndex
    if not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index)
    
    # 2. 确保 calendar 是 DatetimeIndex
    if not isinstance(calendar, pd.DatetimeIndex):
        calendar = pd.to_datetime(calendar)
    
    # 3. 去重（防止重复日期）
    df_stock = df_stock[~df_stock.index.duplicated(keep='first')]
    df_stock.to_csv('stock_data.csv')
    # 4. 重新索引到统一日历
    df_aligned = df_stock.reindex(calendar)
    # df_aligned.to_csv('aligned.csv', index=False)
    
    # 5. 填充：价格类前向填充（限5天），成交量填0
    price_cols = ['open', 'high', 'low', 'close']
    # if all(col in df_aligned.columns for col in price_cols):
        # df_aligned[price_cols] = df_aligned[price_cols].fillna(method="ffill", limit=5)
    
    for col in df_stock.columns:    #['date','code']:
        if col  in price_cols:
            df_aligned[col] = df_aligned[col].ffill(limit=5)
        if col == 'volume':
            df_aligned[col] = df_aligned[col].fillna(0)
    # df_aligned.to_csv('aligned2.csv', index=False)
    # if 'volume' in df_aligned.columns:
        # df_aligned['volume'] = df_aligned['volume'].fillna(0)
    
    return df_aligned
#    # df_stock = df_stock[df_stock.index.duplicated(keep='first')]  # 去重
#    df_aligned = df_stock.reindex(calendar)
#    # 价格前向填充（最多5天，避免长期停牌）
#    price_cols = ['open', 'high', 'low', 'close']
#    # df_aligned[price_cols] = df_aligned[price_cols].fillna(method='ffill', limit=5)         #待查实
#    for col in price_cols:
#        if col not in price_cols:
#            df_aligned[col] = df_aligned[col].ffillna(limit=5)
#    # 成交量填0
#    df_aligned['volume'] = df_aligned['volume'].fillna(0)
#    return df_aligned
#---------------------------    
# 更新股票数据      
#----------------------------
def update_stock_data(code:str)->None:

    home_dir = Path(".") /'.working'/ code
    home_dir.mkdir(parents=True, exist_ok=True)


    
    # 获取股票历史数据,如果本地磁盘没有，则直接从baostock获取；若有，则取本地磁盘数据的最后一天的下一天，以此为起始日。截止日均为今日。
    datafile = home_dir/f"{code}.csv"
    today = datetime.now().strftime("%Y-%m-%d")
    if not  datafile.exists():
        df = pd.DataFrame()
        start_date = "1999-01-01"
    else:
        df = pd.read_csv(datafile,parse_dates=True,skip_blank_lines=True)
        df = df.dropna(how='all')

        df['date'] = df['date'].str.replace("/","-")

        last_date = df.iloc[-1]['date']
        start_date = datetime.strptime(last_date,"%Y-%m-%d") + timedelta(days=1)
        start_date = start_date.strftime("%Y-%m-%d")

    #取出之后合并
    end_date = today
    if start_date > end_date:
        new_transaction = pd.DataFrame()
    else:
        new_transaction = fetch_stocks(code, start_date, end_date)
        # days = new_transaction.shape[0]

    if new_transaction.shape[0] == 0:
        return df

    df= pd.concat([df, new_transaction], axis=0).drop_duplicates()
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    df.to_csv(datafile, index=False, date_format='%Y-%m-%d',encoding="gbk")


    return df
#----------------------------------------------
# 添加特征
#----------------------------------------------

#def add_features(df):
#    # 基础价格特征
#    df['ret_1'] = df['close'].pct_change(1)
#    df['ret_5'] = df['close'].pct_change(5)
#    
#    # 移动平均线
#    df['ma5'] = df['close'].rolling(5).mean()
#    df['ma20'] = df['close'].rolling(20).mean()
#    df['ma60'] = df['close'].rolling(60).mean()
#    
#    # 均线位置（是否在均线上方）
#    df['price_above_ma20'] = (df['close'] > df['ma20']).astype(int)
#    
#    # 波动率
#    df['volatility_20'] = df['ret_1'].rolling(20).std()
#    
#    # 成交量变化
#    df['volume_ma5'] = df['volume'].rolling(5).mean()
#    df['volume_ratio'] = df['volume'] / df['volume_ma5']
#    
#    # 相对强弱（vs 沪深300，需额外获取指数数据）
#    # df['rs'] = df['ret_5'] - benchmark_ret_5
#    
#    return df

#---------------------------
# 从baostock 获取股票历史K线数据
#---------------------------
def convert_to_float(df:pd.DataFrame)->pd.DataFrame:
    df = df.replace("", 0)
    # 明确指定数值列的类型
    for col in df.columns:
        if col not in ['date', 'code']:
            df[col] = df[col].astype('float64')  # 使用 float64 而不是默认的 float
    return df
def fetch_stocks(code:str, start_date:str, end_date:str, freq = 'd')->pd.DataFrame:

    cols = ",".join(['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turn','peTTM','psTTM','pcfNcfTTM','pbMRQ'])
    
    empty_df = pd.DataFrame()

    entry = bs.login()
    if entry.error_code != '0':
        print(entry.error_msg)
        bs.logout()
        # return empty_df
    
    rs = bs.query_history_k_data_plus(
            code,
            cols,
            start_date = start_date, 
            end_date   = end_date, 
            frequency  = freq,
            adjustflag= "2"      #复权类型，默认不复权：3；1：后复权；2：前复权。 固定不变。
    )

    if rs.error_code != '0':
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
        bs.logout()
        return empty_df

    print("data feteched from baostock")
    data_list = []
    while rs.next():
        # 获取一条记录，将记录合并在一起
        bs_data = rs.get_row_data()
        data_list.append(bs_data)

    df = pd.DataFrame(data_list, columns=rs.fields)
    if df.shape[0] != 0:  
        # 删去成交量为零的行，重置索引
        df = convert_to_float(df)
        df = df.replace(0,np.nan).dropna()
        df.reset_index(drop=True, inplace=True)

        df.sort_values(by=['date'], ascending=True, inplace=True)

        print("the last date of ohlcv: ", df.iloc[-1]['date'])

    bs.logout()
    return df

def prepare_dataset(
    df_panel: pd.DataFrame,
    freq: str = 'daily',          # 'daily' or 'weekly'
    forward_days: int = 5,        # 预测未来N天收益
    threshold: float = 0.03,      # 收益阈值（3%）
    min_history: int = 60         # 至少需要60天历史计算因子
):
    """
    输入:
        df_panel: 长格式DataFrame，必须包含列:
            ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume',
             'turn', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            index 应为 date（或至少有 date 列）
    
    输出:
        X: 特征矩阵 (DataFrame)
        y: 二值标签 (Series)
        feature_cols: 特征列名列表
    """
    # 确保 date 是索引
    if 'date' in df_panel.columns:
        df_panel = df_panel.set_index('date')
    df_panel = df_panel.sort_index()

    # ====== 1. 重采样到目标频率（如果是周频）======
    if freq == 'weekly':
        # 每周五作为代表（可根据需要调整）
        df_panel = df_panel.groupby(['stock_code']).resample('W-FRI').last()
        df_panel = df_panel.reset_index(level=0)  # 保留 stock_code

    # ====== 2. 按股票分组计算技术因子 ======
    def compute_features(group):
        group = group.sort_index()
        close = group['close']
        volume = group['volume']
        turn = group['turn']
        
        # 收益率
        group['ret_1'] = close.pct_change()
        
        # 动量
        group['mom_5'] = close.pct_change(5)
        group['mom_20'] = close.pct_change(20)
        
        # 均线
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        group['ma_diff_5_20'] = (ma5 - ma20) / (ma20 + 1e-6)
        group['price_vs_ma20'] = (close - ma20) / (ma20 + 1e-6)
        
        # 波动率
        group['volatility_20'] = group['ret_1'].rolling(20).std()
        
        # 成交量 & 换手率
        vol_ma5 = volume.rolling(5).mean()
        group['volume_ratio'] = volume / (vol_ma5 + 1e-6)
        group['turn_5'] = turn.rolling(5).mean()
        
        # 基本面（直接使用，但做倒数处理）
        group['pe_inv'] = 1.0 / (group['peTTM'] + 1)
        group['pb_inv'] = 1.0 / (group['pbMRQ'] + 1)
        group['ps_inv'] = 1.0 / (group['psTTM'] + 1)
        group['pcf_inv'] = 1.0 / (group['pcfNcfTTM'] + 1)
        group['val_score'] = (
            group[['pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv']].mean(axis=1)
        )
        
        return group

    # 使用更明确的方式处理分组计算，避免 FutureWarning
    df_feat_list = []
    for stock_code, group in df_panel.groupby('stock_code'):
        computed = compute_features(group)
        df_feat_list.append(computed)
    
    df_feat = pd.concat(df_feat_list)
    
    # ====== 3. 构建标签 ======
    def add_label_and_return(group):
        future_ret = group['close'].shift(-forward_days) / group['close'] - 1
        group['future_return'] = future_ret      #  ←←← 新增：连续收益
        group['label'] = (future_ret > threshold).astype('int64')  # 明确指定整数类型
        return group
    
    if 'stock_code' in df_feat.columns:
        df_feat = df_feat.drop(columns=['stock_code'])
    
    # 同样使用显式循环避免警告
    df_labeled_list = []
    for stock_code, group in df_feat.groupby('stock_code'):
        labeled = add_label_and_return(group)
        df_labeled_list.append(labeled)
    
    df_labeled = pd.concat(df_labeled_list)
    
    # ====== 4. 删除不足历史的行 ======
    df_clean = df_labeled.dropna(subset=[
        'mom_20', 'volatility_20', 'val_score', 'label'
    ])
    
    # ====== 5. 横截面标准化（关键！）======
    feature_cols = [
        'mom_5', 'mom_20', 'ma_diff_5_20', 'price_vs_ma20',
        'volatility_20', 'volume_ratio', 'turn_5',
        'pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv', 'val_score'
    ]
    
    # 对每个交易日，对所有股票做 Z-Score
    # 明确指定数值类型，避免类型推断警告
    df_clean.loc[:, feature_cols] = df_clean.groupby(level=0)[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    ).astype('float64')  # 明确指定浮点类型
    
    # ====== 6. 最终清理 ======
    final_df = df_clean.dropna(subset=feature_cols + ['label', 'future_return'])
    
    final_df.to_csv('final_df.csv', index='date', encoding="gbk")
    return final_df, feature_cols

#-----------------------
# 获取沪深300，中证500，上证50
#-----------------------
def get_master_list(code:str)->pd.DataFrame:
       # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    if code == 'hs300':
        rs = bs.query_hs300_stocks()
    elif code == 'sz50':
        rs = bs.query_sz50_stocks()
    elif code == 'zz500':
        rs = bs.query_zz500_stocks()
    else:
        print("code error")
        lg = bs.logout()
        return pd.DataFrame()

    print('query error_code:'+rs.error_code)
    print('query error_msg:'+rs.error_msg)
    
    # 打印结果集
    master_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        master_list.append(rs.get_row_data())
    result = pd.DataFrame(master_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv(f'{code}_stocks.csv', index=False, encoding="gbk" )
    # 登出系统
    bs.logout()

    return result 

def get_trading_days(start_date:str, end_date:str="")->pd.DataFrame:
    # 登录
    if end_date == "":
        end_date = datetime.now().strftime("%Y-%m-%d")
        
    bs.login()

    rs = bs.query_trade_dates(start_date=start_date,  end_date=end_date)
    # trading_days 是 DatetimeIndex-like array
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    df_calendar = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()
    return df_calendar

#-----------------------
# 获取最新数据,废弃不用
#-----------------------
def get_latest_data(code:str, end_date:str="", window_size=60):
    # df = pd.read_csv(f'.working/{code}/{code}.csv', encoding="gbk")

    df_final = pd.read_csv('final_df.csv', encoding="gbk")
    df = df_final.loc[df_final['code']==code]
    
    # print(df.columns.to_list(), df.shape)
    return df.iloc[-window_size:]
def main():

    hs300 = pd.read_csv('hs300_stocks.csv', encoding="gbk")
    stock_list = hs300['code'].tolist()

    # stock_list = ["sh.600000", "sh.600009", "sh.600010", "sh.600011", "sh.600015"]
    #if Path('index.csv').exists():
    #    sz_index = pd.read_csv('index.csv')
    #else:
    #    sz_index = get_trading_days(start_date="2000-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
    sz_index = pd.read_csv('index.csv')
    
    all_trading_days = pd.to_datetime(sz_index['date']).sort_values().unique()

    # 3. 合并所有股票
    all_dfs = []
    
    # stock_list = stock_list[:5 ]
    #for code in tqdm(stock_list, desc="prepare dataset"):                # BY YANG...  50只股票  
    #    # df = fetch_stocks(code,start_date="2000-01-01",end_date=datetime.now().strftime("%Y-%m-%d"))  # 返回 index=date 的 DataFrame
    #    df = pd.read_csv(f".working/{code}/{code}.csv")
    #    # df = update_stock_data(code)
    #    df['stock_code'] = code
    #    # df = pd.read_csv(f"{code}.csv")
    #    # df.to_csv(f"{code}.csv", index=False)
    #    df.set_index('date', inplace=True)
    #    
    #    df_aligned = align_stock_to_calendar(df, all_trading_days)
    #    # df_aligned.to_csv(f"{code}.aligned.csv", index=False)
    #    all_dfs.append(df_aligned)
    #
    #df_panel = pd.concat(all_dfs).sort_index()  # 按日期排序
    # df_panel.to_csv("all.csv",index='date')

    df_final = pd.read_csv('final_df.csv', encoding="gbk", index_col='date')
    feature_cols = ['mom_5', 'mom_20', 'ma_diff_5_20', 'price_vs_ma20',
                    'volatility_20', 'volume_ratio', 'turn_5',
                    'pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv', 'val_score']
    # df_final,feature_cols = prepare_dataset(df_panel,threshold=0.1)
    # print(X.shape, y.shape)
    X = df_final[feature_cols]
    y = df_final['label']

    # y_continuous = df_final['future_return']  # ←←← 关键！
    
    # X = X.droplevel(level=[0,1])                  # 直接读文件后，这个地方就不用了。
    # y = y.droplevel(level=[0,1])

    split_date = '2024-01-01'
    val_end    = "2025-06-30" 
    X_train = X[X.index < split_date]                   # BY YANG...  大X,小y,  why?
    y_train = y[y.index < split_date]
    X_val = X[(X.index >= split_date) & (X.index < val_end)]
    y_val = y[(y.index >= split_date) & (y.index < val_end)]

    y_continuous = df_final[(y.index >= split_date) & (y.index < val_end)]['future_return']  # ←←← 关键！
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': ['auc','binary_logloss'],          # AUC 更关注排序能力
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'is_unbalance': True,          # 或 'scale_pos_weight': pos_weight,
        'verbose': 1,
        'seed': 42,
        'max_depth': -1,               # 不限制深度（由 num_leaves 控制）
        'min_data_in_leaf': 50        # 每个叶子至少50个样本（防过拟合）
    }
    
#    model = lgb.train(
#        params,
#        train_data,
#        valid_sets=[val_data],
#        num_boost_round=500,
#        callbacks=[lgb.early_stopping(stopping_rounds=50),
#        lgb.log_evaluation(50) ],
#        
#    )
#    print("Best iteration:", model.best_iteration)
#    model.save_model('model.lgb') 

    model = lgb.Booster(model_file='model.lgb')
    # 1. 特征重要性（看哪些因子真有用）
    lgb.plot_importance(model, max_num_features=15)
    
    # 2. 预测概率分布
    preds = model.predict(X_val)
    print("预测概率均值:", preds.mean())  # 应接近正样本比例（如 0.2）
    
    y_true_return = y_continuous
    # 3. 计算 Rank IC（核心！）
    from scipy.stats import spearmanr
    ic, p_value = spearmanr(preds,y_true_return)  # 注意：这里 y 应是连续收益，不是0/1

    print(f"Rank IC: {ic:.4f}")

    print(f"P-value: {p_value:.4f} (应 < 0.05 才显著)")

    # 对当前所有候选股票（比如中证500成分股）计算得分
    stock_scores = {}
    for stock in stock_list[:10]:
        # df_latest = get_latest_data(stock,window_size=60)  # 获取最近60天数据
        # df_feat = add_features(df_latest)
        df_stock = df_final.loc[df_final['code']==stock]
        # print(df_code.columns.to_list(), df_code.shape)

        X_today = df_stock[feature_cols].iloc[[-1]]  # 最新一天特征
        # print(X_today)
        
        prob = model.predict(X_today)[0]
        stock_scores[stock] = prob
        print(stock, prob) 
    # 选出概率最高的10只
    selected = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    print("今日推荐股票:", [s[0] for s in selected])
if __name__ == '__main__':
    main()
