import pandas as pd
import numpy as np

from pathlib import Path
from baostock_ops import BaostockOps,my_calendar
from datetime import datetime,timedelta

class stock_data:
    index_mapping = {
        "CSI300": "sh.000300",
        "CSI500": "sh.000905",
        "CSI1000": "sh.000852"
    }
    very_beginning = "2020-01-01"
    feature_columns = {'CSI500':[ 'reversal_5', 'pb_inv', 'ps_inv','pe_inv','volatility_20', 'turn_5'],
                    'CSI300':[ 'reversal_5', 'pb_inv', 'ps_inv','pe_inv','volatility_20', 'turn_5'],        
                    'CSI1000':['reversal_5',  'pb_inv', 'ps_inv','turn_5']
                    }

    def __init__(self):
        self.base_dir = Path("local") 
        self.calendar = my_calendar(self.base_dir)

        self.stock_map = self.update_stock_map()

        self.total_dataset = self.read_parquet()
        self.total_dataset.set_index("date", inplace=True)                      # 设置索引为date列, 这是所有的数据，从very beginning开始到今天
        
        self.working_dataset = None                                             # working_dataset是分析的数据，初始时置为None，用set_working_dataset设置
        
        self.calendar = my_calendar(self.base_dir).calendar
        self.datasource = BaostockOps(self.base_dir)

        self.stock_pool = None
        
    def set_pool(self, stock_pool:str):
        self.stock_pool = stock_pool
    def read_parquet(self):
        # 读取base_dir及其子目录中所有匹配total_*.parquet的文件
        parquet_files = list(self.base_dir.rglob("total_*.parquet"))
        if not parquet_files:
            # 如果在子目录中没有找到，尝试在当前目录查找
            parquet_files = list(self.base_dir.glob("total_*.parquet"))
        
        if parquet_files:
            # 读取所有匹配的parquet文件并合并
            dfs = [pd.read_parquet(file) for file in parquet_files]
            total_dataset = pd.concat(dfs, ignore_index=True)
        else:
            # raise FileNotFoundError(f"No parquet files matching 'total_*.parquet' found in {self.base_dir} or its subdirectories")
            total_dataset = BaostockOps().refresh_dataset(self.stock_list, start_date=self.very_beginning)
            self.save_parquet(total_dataset)

        return  total_dataset

    def save_parquet(self, df:pd.DataFrame, date_col:str="date")->None:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 筛选2020-01-01之后的数据
        df_filtered = df[df[date_col] >= self.very_beginning]
        
        # 检查是否有符合条件的数据
        if df_filtered.empty:
            print("没有找到2020-01-01之后的数据")
        else:
            print(f"找到 {len(df_filtered)} 条2020-01-01之后的数据")
            
            # 按年份分组并保存到不同的parquet文件
            for year, group in df_filtered.groupby(df_filtered[date_col].dt.year):
                output_file = Path(self.base_dir) / f"total_{year}.parquet"
                group.to_parquet(output_file)
                print(f"Saved {output_file} with {len(group)} rows")
                
    def set_working_dataset(self, until_date:str)->None:
        until = datetime.strptime(until_date, "%Y-%m-%d")
        self.working_dataset = self.total_dataset[self.total_dataset.index <= until]   # 只保留指定日期之后的数据,这是程序要分析的数据
        return self.working_dataset

    def update_dataset(self, start_date_str:str)->None:
        '''如是更新整个库的数据，则从very beginning开始更新，如是日常，则从指定日期开始更新，保险起见，取20天数据
        以覆盖当年内最长假期'''
        # start_date = pd.to_datetime(today) - pd.to_timedelta(20, unit='d')
        # start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')

        stock_list = []
        for name, members in self.stock_map.items():
            stock_list = stock_list + members

        dataset = BaostockOps().refresh_dataset(stock_list, start_date_str)
        
        if start_date_str == self.very_beginning:
            # 原数据丢弃
            self.total_dataset = dataset
        else:
            self.total_dataset.reset_index(inplace=True)
            self.total_dataset = pd.concat([self.total_dataset, dataset], ignore_index=True).drop_duplicates()
        # self.total_dataset.set_index("date", inplace=True)

        self.save_parquet(self.total_dataset)
        self.total_dataset.set_index("date", inplace=True)
        
    def update_index(self):
        BaostockOps().update_index(self.index_mapping, self.base_dir)
    def update_stock_map(self):
        stock_map = dict()
        for name, _ in self.index_mapping.items():
            members = pd.read_csv(self.base_dir / name / f"{name}_list.csv")['code'].to_list()
            stock_map[name] = members
             
        return stock_map
    def compute_features(self, group):
        group = group.sort_index()
        close = group['close']
        # high  = group['high']
        # low   = group['low']
        volume = group['volume']
        turn  = group['turn']

        # 收益率
        group['ret_1'] = close.pct_change()

        # 动量因子 (过去N日收益率)
        group['mom_5'] = close.pct_change(5)
        group['mom_10'] = close.pct_change(10)
        group['mom_20'] = close.pct_change(20)

        # === 2. 新增：反转因子（核心！）===
        group['reversal_5'] = -group['mom_5']  # 短期反转信号

        # 均线  （可选保留，但IC弱）===
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

         # === 5. 基本面：前向填充（关键！）===
        #  fundamentals = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
        # for col in fundamentals:
        #     if col in group.columns:
        #         group[col] = group[col].fillna(method='ffill')

        # 基本面（直接使用，但做倒数处理）
        group['pe_inv'] = 1.0 / (group['peTTM'] + 1)
        group['pb_inv'] = 1.0 / (group['pbMRQ'] + 1)
        group['ps_inv'] = 1.0 / (group['psTTM'] + 1)
        group['pcf_inv'] = 1.0 / (group['pcfNcfTTM'] + 1)
        group['val_score'] = (
            group[['pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv']].mean(axis=1)
        )

        # 动量因子 (过去N日收益率)
        #df['mom_5'] = df.groupby('stock_code')['close'].pct_change(5)
        #df['mom_10'] = df.groupby('stock_code')['close'].pct_change(10)
        #df['mom_20'] = df.groupby('stock_code')['close'].pct_change(20)
        
        # RSI (14日)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        group['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 收益率
        group['ret_1'] = close.pct_change()
    
        # 换手率、波动率等可复用你已有的计算
        # group['turn_5'] = turn.rolling(5).mean()
    
        # 波动率
        # group['volatility_20'] = group['ret_1'].rolling(20).std()

        return group

    def align_stock_to_calendar(self, df_stock, calendar):
        '''用于原始数据按照日历对齐，方便计算技术指标tech indicators，也叫做features'''
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

        return df_aligned

    def get_predict_dataset(self, working_dataset:pd.DataFrame, stock_list:list[str], feature_cols:list[str]):
        '''准备日k线数据，成分股的原始日K线数据，按交易日历对齐,其实日期为数据集的开始日期, 截止日期为当天
        total_dataset: 截止到当天的交易所日历的最后交易日的数据，来构建预测数据集。长度不限。
        比如：今天2025-10-10，则从2024-10-10开始，到2025-10-10结束
        stock_pool: 股票池， CSI300, CSI500, CSI1000
        '''

        start_date = working_dataset.index.min()              # 获取股票池的开始日期作为数据集的开始日期,日期已是索引。
        start_date_str = start_date.strftime("%Y-%m-%d")
        df_trading_days = self.calendar[self.calendar['calendar_date'] >= start_date_str].copy()

        all_trading_days = pd.to_datetime(df_trading_days.loc[df_trading_days['is_trading_day']==1]['calendar_date']).sort_values().unique()

        all_dfs = []
        

        # 使用向量化操作一次性过滤所有需要的股票数据
        mask = working_dataset['code'].isin(stock_list)
        filtered_data = working_dataset[mask]
        
        for stock_code, group in filtered_data.groupby('code'):
            df = group.copy()

            # 与证交所交易日对齐，个股不交易的日子向前填充，即 若下一日不交易，则今日数据填入
            df_aligned = self.align_stock_to_calendar(df, all_trading_days)

            all_dfs.append(df_aligned)

        df_panel = pd.concat(all_dfs).sort_index()  # 按日期排序

        df_predict = self.prepare_dataset(df_panel, self.stock_pool)          #计算技术指标，zscore归一化

        return df_predict

    def prepare_dataset(self,
        df_panel: pd.DataFrame,
        stock_pool: str,            # 输入的股票池名称，如'CSI300','CSI500'等
        freq: str = 'daily',          # 'daily' or 'weekly',weekly没用上
        # forward_days: int = 10,        # 预测未来N天收益,5天表示1周, 
        # threshold: float = 0.03,      # 收益阈值（3%），按照麻雀战术，超2.5%的收益就要卖出, 这两项在计算futereturn的时候用
        min_history: int = 60        # 至少需要60天历史计算因子,未用上
                        
    ):
        """
        输入:
            df_panel: 长格式DataFrame，必须包含列:
                ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume',
                 'turn', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
                index 应为 date（或至少有 date 列）

        输出:
            无，
            保存回测和预测用的数据集后，退出
            #X: 特征矩阵 (DataFrame)
            #y: 二值标签 (Series)
            #feature_cols: 特征列名列表
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

        df_feat = df_panel.groupby('code').apply(self.compute_features, include_groups=False)

        # ====== 3. 构建标签, 加上future_return列，用于计算因子 ======
#        def add_label_and_return(group):
#            future_ret = group['close'].shift(-forward_days) / group['close'] - 1
#            group['future_return'] = future_ret      #  ←←← 新增：连续收益
#            group['label'] = (future_ret > threshold).astype(int)
#            return group


        # df_predict = df_feat.copy()
        # df_backtest = df_feat.groupby('code').apply(add_label_and_return)             # BY YANG...耗时较长

        # 删除缺失值, 但一般不会出现
        # df_backtest = df_backtest.dropna(subset=[ 'mom_20', 'volatility_20', 'val_score', 'label' ])
        # 检查列是否存在后再进行 dropna 操作，防止 KeyError
        cols_to_check = ['mom_20', 'volatility_20', 'pb_inv']
        existing_cols = [col for col in cols_to_check if col in df_feat.columns]
        df_predict  = df_feat.dropna(subset=existing_cols)

#        feature_cols = [
#            'reversal_5',          # 替代 mom_5
#            'pb_inv',              # 强价值因子
#            'ps_inv',              # 强价值因子
#            'pe_inv',              # 辅助
#            'volatility_20',       # 风险控制
#            'turn_5'               # 流动性
#            # 'value_reversal'    # 可单独测试，也可作为组合输入
#        ]

        feature_cols = self.feature_columns.get(stock_pool, [])
        
        # ====== 5. 横截面标准化（关键！）======
        # 对每个交易日，对所有股票做 Z-Score
        def robust_zscore(x):
            if x.isna().all():
                return np.full_like(x, np.nan)
            mean = x.mean()
            std = x.std()
            if std == 0:
                return np.zeros_like(x)
            return (x - mean) / (std + 1e-6)

        # df_backtest.loc[:,feature_cols] = df_backtest.groupby(level=0)[feature_cols].transform(robust_zscore)
        df_predict.loc[:,feature_cols] = df_predict.groupby(level=0)[feature_cols].transform(robust_zscore)

        # df_clean[feature_cols] = df_clean.groupby(level=0)[feature_cols].apply(
            # lambda x: (x - x.mean()) / (x.std() + 1e-6)
            # robust_zscore
        # )

        # 在标准化前，对每个因子做 ±3 标准差缩尾
        for col in feature_cols:
#            df_backtest.loc[:, col] = df_backtest.groupby(level=0)[col].transform(
#                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
#            )
            df_predict.loc[:, col] = df_predict.groupby(level=0)[col].transform(
                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
            )

        # ====== 6. 最终清理 ======
        # 回测数据中最后10天因为前向计算future_return, 故这10天不会有数据。可以丢弃。也可以不丢，在计算因子时再处理
        # df_backtest = df_backtest.dropna(subset=feature_cols + ['label','future_return'])
        # 预测用数据，不需要future_return列和label列,丢弃空行即可。
        df_predict  = df_predict.dropna(subset=feature_cols)
#        if not is_predict_mode:
#            final_df = df_clean.dropna(subset=feature_cols + ['label', 'future_return'])
#        else:
#            final_df = df_clean.dropna(subset=feature_cols)
#            # print(f"Future return std: {final_df['future_return'].std():.4f}")
#        for col in feature_cols:
#            ic = spearmanr(df_backtest[col], df_backtest['future_return'])[0]
#            print(f"{col}: Rank IC = {ic:.4f}")

        # 保存数据。以便查验
        # df_backtest.to_csv(self.base_dir / stock_pool / "backtest.dataset.csv", index=True)
        # df_predict.to_csv(self.base_dir / stock_pool / "predict.dataset.csv", index=True)

        return df_predict


#for test
if __name__ == "__main__":
    today = datetime.now()          #.strftime("%Y-%m-%d")
    database = stock_data()
    # database.update_dataset(start_date_str="2024-01-01")
    df = database.total_dataset
    if not df.empty:
        database.update_dataset(datetime.strftime(today-timedelta(days=20),'%Y-%m-%d'))
        print(df.shape)
    database.update_index()
#    for key, value in ops.stock_list.items():
#        print(key, len(value))
    
    # df_feat = df.groupby('code').apply(ops.compute_features, include_groups=False)