import numpy as np
import baostock as bs
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm import tqdm
import json

from scipy.stats import spearmanr

from AutoFactorSelector import AutoFactorSelector

class Selector(object):
    working_dir = Path(".working")
    base_dir = Path(".local")
    feature_cols = ['mom_5', 'mom_20', 'ma_diff_5_20', 'price_vs_ma20',
                    'volatility_20', 'volume_ratio', 'turn_5',
                    'pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv', 'val_score']
    def __init__(self):
        self.calendar = self.load_calendar()

    def init_stock_pool(self,name:str, stock_list:list[str]=[]):
        if name in ("hs300","sz50","zz500"):
            self.get_master_list(self.name)
        
        if len(stock_list):
            pool_dir = self.base_dir / name
            pool_dir.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(stock_list, columns=['code_name'])
            df.to_csv(pool_dir/f"{name}_stocks.csv")
    #
    #---
    # 判断今日是否为交易日
    #---
    def is_trading_day(self,day:str)->bool:
        '''从self.calendar取得day所对应的is_trading_day的值，为1返回True，否则返回False
        '''
        today = datetime.now().strftime("%Y-%m-%d")
        if day > today or day < "2000-01-01": 
            return False

        self.calendar['calendar_date']
        trading = self.calendar.loc[self.calendar['calendar_date']==day]['is_trading_day']
        return trading == 1


    def update_stock_list(self)->list[str]:
        '''self.working_dir路径下取得所有csv的文件名，不含扩展名，写入列表变量，返回该列表。'''
        csv_files = []
        for file_path in self.working_dir.glob("*.csv"):
            csv_files.append(file_path.stem)  # stem属性是文件名不包含扩展名
        return csv_files
    # -------------------------------------
    # 加载预测结果, 数据保存在csv文件中，下同。 是否搬进数据库，后面再说
    # -------------------------------------
    def load_predictions(self):
        if self.prediction_file.exists():
            return pd.read_csv(self.prediction_file, index_col='date')
        else:
            predictions = pd.DataFrame(columns=['date']+ self.stock_list)
            # predictions.set_index('date', inplace=True)
            predictions.to_csv(self.prediction_file, index=False)
            return predictions
    #---------------------------
    # 从baostock 获取股票历史K线数据
    #---------------------------
    def _convert_to_float(self, df:pd.DataFrame)->pd.DataFrame:
        df = df.replace("", 0)
        for col in df.columns:
            if col not in ['date', 'code']:
                df[col] = df[col].astype(float)
        return df
    def _fetch_stocks(self, code:str, start_date:str, end_date:str, freq = 'd')->pd.DataFrame:
    
        cols = ",".join(['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turn','peTTM','psTTM','pcfNcfTTM','pbMRQ'])
        empty_df = pd.DataFrame()
        
        rs = bs.query_history_k_data_plus(
                code,
                cols,
                start_date = start_date, 
                end_date   = end_date, 
                frequency  = freq,
                adjustflag= "2"      #复权类型，默认不复权：3；1：后复权；2：前复权。 固定不变。
        )
    
        if rs.error_code != '0':
            print('query_history_k_data_plus respond error_msg:'+rs.error_msg)
            return empty_df
    
        data_list = []
        while rs.next():
            # 获取一条记录，将记录合并在一起
            bs_data = rs.get_row_data()
            data_list.append(bs_data)
    
        df = pd.DataFrame(data_list, columns=rs.fields)
        if df.shape[0] != 0:  
            # 删去成交量为零的行，重置索引
            df = self._convert_to_float(df)
            # df = df.replace(0,np.nan).dropna()
            df.reset_index(drop=True, inplace=True)
    
            df.sort_values(by=['date'], ascending=True, inplace=True)
    
            print(f"the last date of {code} ohlcv: {df.iloc[-1]['date']}")
        else:
            print(f"no new ohlcv data for {code}")
    
        return df

    #---------------------------    
    # 更新股票数据，增量方式更新，不需要整个下载全部数据      
    #----------------------------
    def _update_stock_data(self, code:str)->None:
    
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取股票历史数据,如果本地磁盘没有，则直接从baostock获取；若有，则取本地磁盘数据的最后一天的下一天，以此为起始日。截止日均为今日。
        datafile = self.working_dir/f"{code}.csv"
        today = datetime.now().strftime("%Y-%m-%d")
        if not  datafile.exists():
            df = pd.DataFrame()
            start_date = "2000-01-01"
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
            new_transaction = self._fetch_stocks(code, start_date, end_date)
            # days = new_transaction.shape[0]
    
        # 没有取到数据，则返回原先的数据, 有就拼接起来
        if new_transaction.shape[0] == 0:
            return df
    
        df = pd.concat([df, new_transaction], axis=0).drop_duplicates()
        # 重置索引
        df.reset_index(drop=True, inplace=True)
        df.to_csv(datafile, index=False, date_format='%Y-%m-%d',encoding="gbk")
    
        return df
    def update_dataset(self)->None:

        stock_list = self.update_stock_list()

        entry = bs.login()          #一次登录，取多条数据
        if entry.error_code != '0':
            print(entry.error_msg)
            bs.logout()
            return 

        for code in stock_list:
            self._update_stock_data(code)
        bs.logout()

    def prepare_dataset(self,
        df_panel: pd.DataFrame,
        stock_pool: str,
        freq: str = 'daily',          # 'daily' or 'weekly',weekly没用上
        forward_days: int = 10,        # 预测未来N天收益,5天表示1周
        threshold: float = 0.03,      # 收益阈值（3%），按照麻雀战术，超2.5%的收益就要卖出
        min_history: int = 60        # 至少需要60天历史计算因子,未用上
        # is_predict_mode: bool = False     # ← 新增参数
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
    
        # ====== 2. 按股票分组计算技术因子 ======
        def compute_features(group):
            group = group.sort_index()
            close = group['close']
            # high  = group['high']
            # low   = group['low']
            volume = group['volume']
            turn  = group['turn']
            
            # 收益率
            group['ret_1'] = close.pct_change()
            
            # 动量
            group['mom_5'] = close.pct_change(5)
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
            
#            # === 5. 基本面：前向填充（关键！）===
#            fundamentals = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
#            for col in fundamentals:
#                if col in group.columns:
#                    group[col] = group[col].fillna(method='ffill')

            # 基本面（直接使用，但做倒数处理）
            group['pe_inv'] = 1.0 / (group['peTTM'] + 1)
            group['pb_inv'] = 1.0 / (group['pbMRQ'] + 1)
            group['ps_inv'] = 1.0 / (group['psTTM'] + 1)
            group['pcf_inv'] = 1.0 / (group['pcfNcfTTM'] + 1)
            group['val_score'] = (
                group[['pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv']].mean(axis=1)
            )
            
            return group
    
        df_feat = df_panel.groupby('code').apply(compute_features, # BY YANG...添加include_groups=False
                                                        include_groups=False)  #消除futureWarning
        
        # ====== 3. 构建标签, 加上future_return列，用于计算因子 ======
        def add_label_and_return(group):
            future_ret = group['close'].shift(-forward_days) / group['close'] - 1
            group['future_return'] = future_ret      #  ←←← 新增：连续收益
            group['label'] = (future_ret > threshold).astype(int)
            return group
        
        if 'stock_code' in df_feat.columns:                 # 这里多出了一个stock_code列
            df_feat = df_feat.drop(columns=['stock_code'])

        df_predict = df_feat.copy()
        df_backtest = df_feat.groupby('code').apply(add_label_and_return)

        # 删除缺失值, 但一般不会出现
        df_backtest = df_backtest.dropna(subset=[ 'mom_20', 'volatility_20', 'val_score', 'label' ])
        df_predict  = df_feat.dropna(subset=['mom_20', 'volatility_20', 'pb_inv']) 
        
#        if not is_predict_mode:
#            # df_labeled = df_feat.groupby('stock_code').apply(add_label_and_return)
#            df_clean = df_feat.groupby('stock_code').apply(add_label_and_return)
#
#            # ====== 4. 删除不足历史的行 ======
#            #df_clean = df_labeled.dropna(subset=[
#            #    'mom_20', 'volatility_20', 'val_score', 'label'
#            #    ])
#        else:
#            # ====== 预测模式：不加标签，保留所有最新数据 ======
#            df_clean = df_feat.copy()
#            # 只删除因子计算所需的最小历史缺失（如 mom_20 需20天）
#            df_clean = df_clean.dropna(subset=['mom_20', 'volatility_20', 'pb_inv']) 
        
#        feature_cols = [
#            'mom_5', 'mom_20', 'ma_diff_5_20', 'price_vs_ma20',
#            'volatility_20', 'volume_ratio', 'turn_5',
#            'pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv', 'val_score'
#        ]
        
        feature_cols = [
            'reversal_5',          # 替代 mom_5
            'pb_inv',              # 强价值因子
            'ps_inv',              # 强价值因子
            'pe_inv',              # 辅助
            'volatility_20',       # 风险控制
            'turn_5'               # 流动性
            # 'value_reversal'    # 可单独测试，也可作为组合输入
        ]
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

        df_backtest.loc[:,feature_cols] = df_backtest.groupby(level=0)[feature_cols].transform(robust_zscore)
        df_predict.loc[:,feature_cols] = df_predict.groupby(level=0)[feature_cols].transform(robust_zscore)
        # df_clean[feature_cols] = df_clean.groupby(level=0)[feature_cols].apply(
            # lambda x: (x - x.mean()) / (x.std() + 1e-6)
            # robust_zscore
        # )

        # 在标准化前，对每个因子做 ±3 标准差缩尾
        for col in feature_cols:
            df_backtest.loc[:, col] = df_backtest.groupby(level=0)[col].transform(
                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
            ) 
            df_predict.loc[:, col] = df_predict.groupby(level=0)[col].transform(
                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
            ) 
        
        # ====== 6. 最终清理 ======
        # 回测数据中最后10天因为前向计算future_return, 故这10天不会有数据。可以丢弃。也可以不丢，在计算因子时再处理
        df_backtest = df_backtest.dropna(subset=feature_cols + ['label','future_return'])       #问题在这儿
        # 预测用数据，不需要future_return列和label列,丢弃空行即可。
        df_predict  = df_predict.dropna(subset=feature_cols)
#        if not is_predict_mode:
#            final_df = df_clean.dropna(subset=feature_cols + ['label', 'future_return'])
#        else:
#            final_df = df_clean.dropna(subset=feature_cols)
#            # print(f"Future return std: {final_df['future_return'].std():.4f}")
        for col in feature_cols:
            ic = spearmanr(df_backtest[col], df_backtest['future_return'])[0]
            print(f"{col}: Rank IC = {ic:.4f}") 

        df_backtest.to_csv(self.base_dir / stock_pool / "backtest.dataset.csv", index=True)
        df_predict.to_csv(self.base_dir / stock_pool / "predict.dataset.csv", index=True)

        return
    
    #-----------------------
    # 获取沪深300，中证500，上证50股票列表
    #-----------------------
    def get_master_list(self,code:str)->pd.DataFrame:

        # 若列表存在，则直接读取后返回
        if Path(f'{code}_stocks.csv').exists():
            return pd.read_csv(f'{code}_stocks.csv',index_col=None)

           # 登陆系统
        lg = bs.login()
        if lg.error_code != '0':
            print(f"log into baostock: {lg.error_msg}")
            return pd.DataFrame()
        
        if code == 'hs300':
            rs = bs.query_hs300_stocks()
        elif code == 'sz50':
            rs = bs.query_sz50_stocks()
        elif code == 'zz500':
            rs = bs.query_zz500_stocks()
        else:
            print("code not supported")
            lg = bs.logout()
            return pd.DataFrame()
    
        if rs.error_code != '0':
            print(f"getting stock list respond error: {rs.error_msg}")
            return pd.DataFrame()
    
        # 打印结果集
        master_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            master_list.append(rs.get_row_data())
        result = pd.DataFrame(master_list, columns=rs.fields)
        # 结果集输出到csv文件
        result.to_csv(self.base_dir / code / f'{code}_stocks.csv', index=False, encoding="gbk" )
        # 登出系统
        bs.logout()
    
        return result 
    
    #---------------------------
    # 示例：对齐多只股票到统一交易日历
    #---------------------------
    # 2. 对每只股票对齐
    def align_stock_to_calendar(self, df_stock, calendar):
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
        # price_cols = ['open', 'high', 'low', 'close']
        # if all(col in df_aligned.columns for col in price_cols):
            # df_aligned[price_cols] = df_aligned[price_cols].fillna(method="ffill", limit=5)

        for col in df_stock.columns:    #['date','code']:
            if col in ['date','code']:
                continue
            elif col in ['volume','turn']:         # 成交量数据填充0, 没有交易
                df_aligned[col] = df_aligned[col].fillna(0)
            else:      # 价格数据向前填充，价格无变化,limit放大些，免得麻烦
                df_aligned[col] = df_aligned[col].ffill(limit=20)
        # df_aligned.to_csv('aligned2.csv', index=False)
        # if 'volume' in df_aligned.columns:
            # df_aligned['volume'] = df_aligned['volume'].fillna(0)
        
        return df_aligned
    
    # def get_trading_calender(self, start_date:str, end_date:str=""):
    def load_calendar(self, start_date:str = "2000-01-01", end_date:str=""):    
        # 登录
        if end_date == "":
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        def get_trading_days(start_date:str, end_date:str="")->pd.DataFrame:
            rs = bs.login()
            if rs.error_code != '0':
                print(f"log into baostock: {rs.error_msg}")
                return pd.DataFrame()

            rs = bs.query_trade_dates(start_date=start_date,  end_date=end_date)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            bs.logout()

            return df

        calendar = self.base_dir / "calendar.csv"
        if calendar.exists():
            df_calendar =  pd.read_csv(calendar, parse_dates=True)
            last_date = df_calendar['calendar_date'].max()
            if last_date == end_date:
                return df_calendar
            else:
                df_calendar = get_trading_days(start_date, end_date)
        else:
            df_calendar = get_trading_days(start_date, end_date)
            
        df_calendar.to_csv(calendar, index=False)
        return df_calendar
    
    def to_backtest_dataset(self,df:pd.DataFrame, date:str=""):
        '''删除数据集中日期小于本月的数据, 有利于计算市场因子
        '''
        if date=="":
            date_obj = datetime.now()
        else:
            date_obj = datetime.strptime(date,"%Y-%m-%d")

        start_date_of_month = date_obj.replace(day=1).strftime("%Y-%m-%d")    
        # 获取索引中日期的部分（假设第三层是日期）
        # date_index = df.index.get_level_values(2)
        # 直接对日期索引使用to_period方法并过滤掉本月的数据
        df = df[df.index < start_date_of_month]
        return df

    def get_stock_list(self,stock_pool:str)->list[str]:
        df = pd.read_csv(self.base_dir/stock_pool/f"{stock_pool}_stocks.csv")
        return df['code'].to_list()

    def last_month(self, date:str=""):
        if date == "":
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(date,"%Y-%m-%d")
        last_month = current_date - relativedelta(months=1)
        year  = last_month.year
        month = last_month.month
        return year,month

    # ------------------------------------------------------------------
    # 最终数据集，包含技术指标，并做横截面的归一化处理，用于训练和验证，和测试
    # 每日数据预测的输入数据也用这个函数来构建
    # ------------------------------------------------------------------
    def make_dataframe(self, stock_pool:str):
        '''date: 指定日期，即最近的一个交易日'''

        if not (self.base_dir/stock_pool).exists():
            return

        days = min(252, len(self.calendar))
        df_trading_days = self.calendar.iloc[-days:].copy()

        all_trading_days = pd.to_datetime(df_trading_days.loc[df_trading_days['is_trading_day']==1]['calendar_date']).sort_values().unique()
        
        all_dfs = []
        stock_list = self.get_stock_list(stock_pool=stock_pool)

        for code in tqdm(stock_list, desc="prepare dataset"):                
            stock_data_file = self.working_dir / f"{code}.csv"
            if not stock_data_file.exists():
                bs.login()
                self._update_stock_data(code)
                bs.logout()
            df = pd.read_csv(self.working_dir/f"{code}.csv")
#            days = 60
#            if purpose != "predict":                        # 预测时，只取最近60天的数据，其实已经够多
#                days = min(252, len(df))        
#
#            df = df.iloc[-days:].copy()
            df['stock_code'] = code
            df.set_index('date', inplace=True)
            
            # 与证交所交易日对齐，个股不交易的日子向前填充，即 若下一日不交易，则今日数据填入
            df_aligned = self.align_stock_to_calendar(df, all_trading_days)
            # df_aligned.to_csv(f"{code}.aligned.csv", index=False)
            all_dfs.append(df_aligned)
        
        df_panel = pd.concat(all_dfs).sort_index()  # 按日期排序

        # predict_mode = True if purpose == "predict" else False
        # df_final,feature_cols = 
        self.prepare_dataset(df_panel,stock_pool)

#        if not predict_mode:
#            for col in feature_cols:
#                ic = spearmanr(df_final[col], df_final['future_return'])[0]
#                print(f"{col}: Rank IC = {ic:.4f}")
#
#        df_final.to_csv(self.base_dir / stock_pool / f"{purpose}.csv", index=True)

        return 

    def cal_weights(self, stock_pool, feature_cols, date):
        factor = AutoFactorSelector(ic_threshold=0.02)
    
        # 取上月数据生成的backtest数据集
        df_backtest = pd.read_csv(program.base_dir/ stock_pool/ "backtest.dataset.csv", index_col='date')
        # date = f"{year}-{month:02d}-01"         # 用当前的日期，取截止到上个月底的数据
        df_final = self.to_backtest_dataset(df_backtest, date=date)
        #
        factor.fit(
            df_final, 
            factor_cols=['reversal_5', 'pb_inv', 'ps_inv', 'pe_inv', 'volatility_20']
        )
        factor.report()
        weights = factor.weights
        year, month = self.last_month(date)
        json.dump(weights, open(self.base_dir / stock_pool / f"weights.{year}.{month}.json", "w"))
        return weights
        
    def predict(self, stock_pool, val_end:str):
        df_final = pd.read_csv(self.base_dir /stock_pool/'predict.dataset.csv', index_col='date')
        feature_cols = [
            'code',    
            'reversal_5',          # 替代 mom_5
            'pb_inv',              # 强价值因子
            'ps_inv',              # 强价值因子
            'pe_inv',              # 辅助
            'volatility_20',       # 风险控制
            'turn_5',               # 流动性
            # 'future_return'
            # 'value_reversal'    # 可单独测试，也可作为组合输入
        ]
        X = df_final[feature_cols]
        X = X[X.index <= val_end]
        last_date = X.index.max()           # 用于预测的数据集的最后一天，作为计算因子所用数据集的参照

        start_date = datetime.strptime(last_date,"%Y-%m-%d") + timedelta(days=-60)
        start_date = start_date.strftime("%Y-%m-%d")
        X_val = X[ (X.index >= start_date) & (X.index <=last_date)]

        df_val = X_val.copy()

        # 保存到csv用于验证,待正式发布时，可以删除。
        df_val.to_csv(self.base_dir /stock_pool/f'predict.val.{val_end}.csv', index=True)

        if 'code' not in df_val.columns:
            # 如果 stock_code 在 MultiIndex 的第二层
            df_val = df_val.reset_index(level=1)  # 假设 index 是 (date, stock_code)

        year, month = self.last_month(date=last_date)
        weights_file = self.base_dir / stock_pool / f"weights.{year}.{month}.json"
        print(f"Using weights from {weights_file}")

        if not weights_file.exists():
            print(f"Warning: weights file {weights_file} not found, calculating weights...")
            weights = self.cal_weights(stock_pool, feature_cols, last_date)
        else:
            weights = json.load(open(weights_file, "r"))
        
        composite_score = pd.Series(0.0, index=df_val.index)
        for col in weights:
            if col in X_val.columns:
                composite_score += weights[col] * X_val[col]
            else:
                print(f"Warning: {col} not in input data")
        df_val['composite_score'] = composite_score

#        df_val['composite_score'] = (w_ps_inv * X_val['ps_inv'] + w_pb_inv * X_val['pb_inv'] +
#                                     w_reversal_5 * X_val['reversal_5']  + w_pe_inv*X_val['pe_inv'] +
#                                     w_volatility_20 * X_val['volatility_20'] + w_turn_5 * X_val['turn_5'])
            #+
            # 0.2 * X_val['reversal_5']
        # df_val['composite_score'] = 1.0 * X_val['reversal_5']
                                     
        # 2. 按交易日分组，对股票进行排名（降序：高分排前）
        # df_val['rank_pct'] = df_val.groupby(level=0)['composite_score'].rank(pct=True, ascending=False)
        
        # 3. 按日期分组，对每只股票排名（1 = 最高分）
        df_val['rank'] = df_val.groupby(level=0)['composite_score'].rank(method='min', ascending=False).astype(int)
        
        # 3. 可选：转换为“Top N”信号,获取每个交易日的 Top N 股票（例如 Top 10）
        TOP_N = 10
        top_stocks = df_val[df_val['rank'] <= TOP_N].copy()
        # df_val['is_top10'] = df_val['rank_pct'] <= 0.1  # 做多前10%
        # print(df_val['is_top10'])

        # 5. 【关键】按日期和排名排序，方便查看
        top_stocks = top_stocks.sort_values(['date', 'rank'])
        # top_stocks.to_csv('top_stocks.csv')
        # df_val['is_bottom10'] = df_val['rank_pct'] >= 0.9  # 做空后10%（或仅过滤）

        # 6. 输出示例：打印最近一个交易日的 Top 10
        last_date = top_stocks.index.get_level_values('date').max()
        print(f"\n📅 预测日期: {last_date}")
        
        df_predict = top_stocks[top_stocks.index.get_level_values('date') == last_date][['code', 'composite_score', 'rank']]
        print(df_predict)
        df_predict.to_csv(self.base_dir / stock_pool / f"predict.{last_date}.csv") 

        prediction_history = self.base_dir / stock_pool / "predictions.csv"
        if prediction_history.exists():
            df_history = pd.read_csv(prediction_history, index_col='date')  
            if last_date in df_history.index:
                print(f"{last_date} 的预测结果已存在，无需重复记录")
                return
            # 重置df_predict的索引以匹配历史数据结构
            # df_predict.reset_index(inplace=True)    
            # df_history_reset = df_history.reset_index()
            # 合并数据
            df_combined = pd.concat([df_predict, df_history], axis=0)
            # 去除重复项
            df_combined = df_combined.drop_duplicates(subset=['code','rank', 'composite_score'],keep='last')
            # 按日期排序，最新的在最前面
            df_combined = df_combined.sort_values(['date'], ascending=False)
            df_predict = df_combined.copy()
            
        # print(df_predict)
        df_predict.to_csv(prediction_history, index=True)
        
        
        # 统计 Top 10% 股票中，未来5天收益 ≥3% 的比例
        # top10 = df_val[df_val['is_top10']]
        # print(top10.head())

        # top10.to_csv('top10.csv')
        
        # hit_rate = (top10['future_return'] >= 0.03).mean()
        # print(f"Top 10% 股票中，5日涨幅≥3% 的比例: {hit_rate:.2%}")
        # ic, pval = spearmanr(score, y_continuous)
        # print(f"Composite Score Rank IC: {ic:.4f}")
        

# --------------------------------------------------------------------    
#    def train(self, split_date:str="2024-01-01", val_end:str="2025-12-31"):
#        # df_final, feature_cols = self.make_dataframe(purpose="train")
#        # ch = input("Press Enter to continue...")
#
#        df_final = pd.read_csv('final_df.csv', encoding="gbk", index_col='date')
#        feature_cols = [
#            'reversal_5',          # 替代 mom_5
#            'pb_inv',              # 强价值因子
#            'ps_inv',              # 强价值因子
#            'pe_inv',              # 辅助
#            'volatility_20',       # 风险控制
#            'turn_5'               # 流动性
#            # 'value_reversal'    # 可单独测试，也可作为组合输入
#        ]
#        X = df_final[feature_cols]
#        y = df_final['label']
#    
#        # y_continuous = df_final['future_return']  # ←←← 关键！
#        
#        # X = X.droplevel(level=[0,1])                  # 直接读文件后，这个地方就不用了。
#        # y = y.droplevel(level=[0,1])
#    
#        # split_date = '2024-01-01'
#        # val_end    = "2025-06-30" 
#        X_train = X[X.index < split_date]                   # BY YANG...  大X,小y,  why?
#        y_train = y[y.index < split_date]
#        X_val = X[(X.index >= split_date) & (X.index < val_end)]
#        y_val = y[(y.index >= split_date) & (y.index < val_end)]
#    
#        y_continuous = df_final[(y.index >= split_date) & (y.index < val_end)]['future_return']  # ←←← 关键！
#        
#        train_data = lgb.Dataset(X_train, label=y_train)
#        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
##--------------------------------------------------------------------
#        score = (
#            0.444 * X_val['ps_inv'] +
#            0.556 * X_val['pb_inv'] #+
#            # 0.2 * X_val['reversal_5']
#            )
#
#        ic, pval = spearmanr(score, y_continuous)
#        print(f"Composite Score Rank IC: {ic:.4f}")
## --------------------------------------------------------------------    
#        # ch = input("Press Enter to continue...")
#
#        params = {
#            'objective': 'regression',
#            # 'metric': ['auc','binary_logloss'],          # AUC 更关注排序能力
#            'metric': 'rmse',          # AUC 更关注排序能力
#            'boosting_type': 'gbdt',
#            'num_leaves': 31,
#            'learning_rate': 0.05,
#            'feature_fraction': 0.8,
#            'bagging_fraction': 0.8,
#            'bagging_freq': 5,
#            'is_unbalance': True,          # 或 'scale_pos_weight': pos_weight,
#            'verbose': 1,
#            'seed': 42,
#            'max_depth': -1,               # 不限制深度（由 num_leaves 控制）
#            'min_data_in_leaf': 100        # 每个叶子至少50个样本（防过拟合）
#        }
#        
#        model = lgb.train(
#            params,
#            train_data,
#            # lgb.Dataset(X_train, y_train_continuous),  # ← 连续收益
#            # valid_sets=[lgb.Dataset(X_val, y_val_continuous)],
#            valid_sets=[val_data],
#            num_boost_round=200,
#            callbacks=[ #lgb.early_stopping(stopping_rounds=50),
#            lgb.log_evaluation(20) ],
#            
#        )
#        print("Best iteration:", model.best_iteration)
#        model.save_model(self.model_file)
#        lgb.plot_importance(model, max_num_features=15)
#    
#        # 2. 预测概率分布
#        preds = model.predict(X_val)
#        print("预测概率均值:", preds.mean())  # 应接近正样本比例（如 0.2）
#    
#        y_true_return = y_continuous
#        # 3. 计算 Rank IC（核心！）
#        ic, p_value = spearmanr(preds,y_true_return)  # 注意：这里 y 应是连续收益，不是0/1
#
#        print(f"Rank IC: {ic:.4f}")
#        print(f"P-value: {p_value:.4f} (应 < 0.05 才显著)" )
#        
#        self.model = model
        
#    def daily_update(self):
#        '''更新当日k线数据： 按照stock_list获取股票当日k线数据，并更新本地数据文件，添加技术指标，制作成可以训练的数据集。
#            换句话讲，就是制作成可以训练的数据集。 方便后续模型更新。'''
#        today = datetime.now().strftime("%Y-%m-%d")
#        data_file_today = self.my_dir/f"for_predict_{today}.csv"
#
#        # self.update_portfolio()
#        # 提取60天数据，
#        df, feature_cols = self.make_dataframe(purpose="predict")
#        # df['date'] = today         
#
#        df.to_csv(data_file_today, index=False, date_format='%Y-%m-%d',encoding="gbk")
#

    def get_portfolio(self)->list[str]:
        '''获取self.base_dir目录下的第一级目录名，写入列表后，返回该列表
        '''
        portfolio_list = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                portfolio_list.append(item.name)
        return portfolio_list
    def daily(self):
        '''每日例行更新数据，按照股票池或者组合更新回测和预测数据集，并执行预测，并更新预测结果，写入csv文件 
        '''
        today = datetime.now().strftime("%Y-%m-%d")
        self.calendar = self.load_calendar()
        
        if not self.is_trading_day(today):              # 非交易日
            return

        self.update_dataset()
        
        portfolio_name = self.get_portfolio()

        for stock_pool in portfolio_name:
           self.make_dataframe(stock_pool=stock_pool, purpose="backtest")
           self.make_dataframe(stock_pool=stock_pool, purpose="predict")
           self.predict(stock_pool = stock_pool,val_end=today)

        
#        today = datetime.now().strftime("%Y-%m-%d")                 #  self.df_final.iloc[-1]['date']
#        data_file_today = self.my_dir/f"for_predict_{today}.csv"
#
#        df_for_prediction = pd.read_csv(data_file_today, encoding="gbk") 
#        
        # 如果当天的数据已经存在于prediction_file中，则不再重复计算
#        if data_file_today.exists():
#            self.df_predictions = pd.read_csv(self.prediction_file, index_col=0)
#            if today in self.df_predictions.index:
#                print(f"{today} 的预测结果已存在，无需重复计算")
#                return
        
#        for stock in self.stock_list:
#            # df_latest = get_latest_data(stock,window_size=60)  # 获取最近60天数据
#            # df_feat = add_features(df_latest)
#            df_stock = df_for_prediction.loc[df_for_prediction['code']==stock]
#            # print(df_code.columns.to_list(), df_code.shape)
#    
#            X_today = df_stock[self.feature_cols].iloc[[-1]]  # 最新一天特征
#            # print(X_today)
#            
#            prob = self.model.predict(X_today)[0]
#            stock_scores[stock] = prob
#            print(stock, prob) 
#        #
#        # 全部预测完毕后，保存至self.df_predictions,再保存至文件
#        new_predictions = pd.Series(stock_scores)
#        self.df_predictions.loc[today] = new_predictions
#        self.df_predictions.to_csv(self.prediction_file)

if __name__ == "__main__":
    
    program = Selector()

    # program.update_dataset()
#    program.update_portfolio()
#    
#    ch = input("press enter to continue.....")
    # factor = AutoFactorSelector(ic_threshold=0.02)
#    program.make_dataframe(stock_pool="zz500",purpose="backtest")
#
#    # 取上月数据生成的backtest数据集
#    year,month = program.last_month()
#    df_historical = pd.read_csv(program.base_dir/ "zz500" /f"backtest.{year}.{month}.csv", index_col='date')
#    #
#    factor.fit(
#        df_historical, 
#        factor_cols=['reversal_5', 'pb_inv', 'ps_inv', 'pe_inv', 'volatility_20']
#    )
#    factor.report()
#    #weights = factor.weights
#
#    
#    ch = input("press enter to continue.....")

    # program.make_dataframe(stock_pool="zz500")

    # program.make_dataframe(stock_pool="zz500",purpose="predict")
    program.make_dataframe(stock_pool="zz500")
#    
    program.predict(stock_pool="zz500", val_end="2026-02-01")
    program.predict(stock_pool="zz500", val_end="2026-02-02")
    program.predict(stock_pool="zz500", val_end="2026-02-03")
    program.predict(stock_pool="zz500", val_end="2026-02-04")
    program.predict(stock_pool="zz500", val_end="2026-02-05")
    program.predict(stock_pool="zz500", val_end="2026-02-06")
    program.predict(stock_pool="zz500", val_end="2026-02-13")
    # program.predict(stock_pool="zz500", split_date="2024-12-31", val_end="2026-01-15")
    # program.predict(stock_pool="zz500", split_date="2024-12-31", val_end="2026-02-01")
    # program.predict(stock_pool="zz500", split_date="2024-12-31", val_end="2025-12-01")
    # program.predict(stock_pool="zz500", split_date="2024-12-31", val_end="2025-11-01")
    # program.predict(stock_pool="zz500", split_date="2024-12-31", val_end="2025-12-15")
#    
    
    # mygbm.train()
    # -----------------------------------
    # 如果当日ohlcv数据不存在，则更新当日数据，否则不更新
    # mygbm.daily_update()
    #-----------------------------------
    #预测，如果prediction_file中已经存在当日的预测结果，则不再重复计算

    # mygbm.daily_predict()
    