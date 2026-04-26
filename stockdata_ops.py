import pandas as pd
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

    def __init__(self, date:str):
        self.base_dir = Path("local") 
        self.calendar = my_calendar(self.base_dir)

        self.stock_map = self.update_stock_map()

        self.total_dataset = self.read_parquet()
        self.total_dataset.set_index("date", inplace=True)
        
        until = datetime.strptime(date, "%Y-%m-%d")
        self.dataset = self.total_dataset[self.total_dataset.index <= until]
        
        self.calendar = my_calendar(self.base_dir)
        self.datasource = BaostockOps(self.base_dir)

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
                
    def update_dataset(self, start_date_str:str)->None:

        # start_date = pd.to_datetime(today) - pd.to_timedelta(20, unit='d')
        # start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')

        stock_list = []
        for name, members in self.stock_map.items():
            stock_list = stock_list + members

        dataset = BaostockOps().refresh_dataset(stock_list, start_date_str)
        
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

    def prepare_dataset(self,
        df_panel: pd.DataFrame,
        stock_pool: str,
        freq: str = 'daily',
    ):
        pass

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


#for test
if __name__ == "__main__":
    today = datetime.now()
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