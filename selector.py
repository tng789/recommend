import numpy as np
import baostock as bs
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm import tqdm
import json

from baostock_ops import BaostockOps

from AutoFactorSelector import AutoFactorSelector       #for zz500
from AutoFactorSelector_HS300 import AutoFactorSelector_HS300               #for hs300
from AutoFactorSelector_CS1000 import AutoFactorSelector_CS1000             # for zz1000


class Selector(object):
    working_dir = Path(".working")
    base_dir = Path(".local")
    feature_cols = ['mom_5', 'mom_20', 'ma_diff_5_20', 'price_vs_ma20',
                    'volatility_20', 'volume_ratio', 'turn_5',
                    'pe_inv', 'pb_inv', 'ps_inv', 'pcf_inv', 'val_score']
    def __init__(self):
        self.calendar = BaostockOps().load_calendar()

    def init_stock_pool(self,name:str, stock_list:list[str]=[]):
        if name in ("hs300","sz50","zz500"):
            self.get_master_list(name)

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

        # self.calendar['calendar_date']
        trading = self.calendar.loc[self.calendar['calendar_date']==day]['is_trading_day']
        return trading.all() == 1

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
    def compute_features(self,group):
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

        return group

    # 并行处理优化：使用多进程加速分组计算
    def apply_compute_features(self,args):
        """辅助函数，用于在进程池中应用compute_features"""
        name, group = args
        result = self.compute_features(group)
        return result

    def prepare_dataset(self,
        df_panel: pd.DataFrame,
        stock_pool: str,
        freq: str = 'daily',          # 'daily' or 'weekly',weekly没用上
        # forward_days: int = 10,        # 预测未来N天收益,5天表示1周
        # threshold: float = 0.03,      # 收益阈值（3%），按照麻雀战术，超2.5%的收益就要卖出
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

        # 获取CPU核心数
        # num_cores = max(1, mp.cpu_count() - 1)  # 保留一个核心给其他任务

#        if num_cores > 1 and len(df_panel) > 1000:  # 只有在数据量大时才使用并行
#            # 准备数据用于并行处理
#            grouped = df_panel.groupby('code')
#            stock_groups = [(name, group) for name, group in grouped]
#
#            # 使用进程池并行处理
#            with mp.Pool(processes=num_cores) as pool:
#                results = pool.map(self.apply_compute_features, stock_groups)
#
#            # 合并结果
#            df_feat = pd.concat(results)
#        else:
#            # 数据量小时使用普通方法
#            df_feat = df_panel.groupby('code').apply(self.compute_features, include_groups=False)
        df_feat = df_panel.groupby('code').apply(self.compute_features, include_groups=False)

        # ====== 3. 构建标签, 加上future_return列，用于计算因子 ======
#        def add_label_and_return(group):
#            future_ret = group['close'].shift(-forward_days) / group['close'] - 1
#            group['future_return'] = future_ret      #  ←←← 新增：连续收益
#            group['label'] = (future_ret > threshold).astype(int)
#            return group

        if 'stock_code' in df_feat.columns:                 # 这里多出了一个stock_code列
            df_feat = df_feat.drop(columns=['stock_code'])

        # df_predict = df_feat.copy()
        # df_backtest = df_feat.groupby('code').apply(add_label_and_return)             # BY YANG...耗时较长

        # 删除缺失值, 但一般不会出现
        # df_backtest = df_backtest.dropna(subset=[ 'mom_20', 'volatility_20', 'val_score', 'label' ])
        # 检查列是否存在后再进行 dropna 操作，防止 KeyError
        cols_to_check = ['mom_20', 'volatility_20', 'pb_inv']
        existing_cols = [col for col in cols_to_check if col in df_feat.columns]
        df_predict  = df_feat.dropna(subset=existing_cols)

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

    def prepare_dataset_zz1000(self,
        df_panel: pd.DataFrame,
        stock_pool: str,
        freq: str = 'daily',          # 'daily' or 'weekly',weekly没用上
        # forward_days: int = 10,        # 预测未来N天收益,5天表示1周
        # threshold: float = 0.03,      # 收益阈值（3%），按照麻雀战术，超2.5%的收益就要卖出
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

        df_feat = df_panel.groupby('code').apply(self.compute_features, include_groups=False)

        if 'stock_code' in df_feat.columns:                 # 这里多出了一个stock_code列
            df_feat = df_feat.drop(columns=['stock_code'])

        cols_to_check = ['mom_20', 'volatility_20', 'pb_inv']
        existing_cols = [col for col in cols_to_check if col in df_feat.columns]
        df_predict  = df_feat.dropna(subset=existing_cols)

        feature_cols = [
            'reversal_5',          # 替代 mom_5
            'pb_inv',              # 强价值因子
            'ps_inv',              # 强价值因子
            'turn_5'               # 流动性
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

        df_predict.loc[:,feature_cols] = df_predict.groupby(level=0)[feature_cols].transform(robust_zscore)

        # 在标准化前，对每个因子做 ±3 标准差缩尾
        for col in feature_cols:
            df_predict.loc[:, col] = df_predict.groupby(level=0)[col].transform(
                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
            )

        # ====== 6. 最终清理 ======
        # 回测数据中最后10天因为前向计算future_return, 故这10天不会有数据。可以丢弃。也可以不丢，在计算因子时再处理
        # 预测用数据，不需要future_return列和label列,丢弃空行即可。
        df_predict  = df_predict.dropna(subset=feature_cols)

        # 保存数据。以便查验
        # df_backtest.to_csv(self.base_dir / stock_pool / "backtest.dataset.csv", index=True)
        # df_predict.to_csv(self.base_dir / stock_pool / "predict.dataset.csv", index=True)

        return df_predict

    #-----------------------
    # 获取沪深300，中证500，上证50股票列表
    # baostock 获取的股票列表不算最近新，本函数弃用。列表跟新不频繁，所以手动更新
    #-----------------------
    def get_master_list(self,code:str)->pd.DataFrame:

        # 若列表存在，则直接读取后返回
        stock_list_file = self.base_dir / code / f'{code}_stocks.csv'
        if stock_list_file.exists():
            return pd.read_csv(stock_list_file, index_col=None)

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

    def to_backtest_dataset(self,df:pd.DataFrame, date:str=""):
        '''删除数据集中日期小于本月的数据, 有利于计算市场因子
        已经废弃不用，日后删除
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
        '''取date所指定日期的上一个月的年份和月份， 返回年份和月份'''
        if date == "":
            current_date = datetime.now()
        else:
            current_date = datetime.strptime(date,"%Y-%m-%d")
        last_month = current_date - relativedelta(months=1)
        year  = last_month.year
        month = last_month.month
        return year,month

    def make_backtest_dataset(self, df_predict:pd.DataFrame, date, forward_days=15, threshold=0.04) -> pd.DataFrame:
        '''中证500作为默认样本，预测未来15天收益超4%,而中证1000则不同'''
        def add_label_and_return(group):
            # forward_days = 10
            # threshold = 0.3
            future_ret = group['close'].shift(-forward_days) / group['close'] - 1
            group['future_return'] = future_ret      #  ←←← 新增：连续收益
            group['label'] = (future_ret > threshold).astype(int)
            return group

        year, month = date.year, date.month
        first_day_of_month = datetime(year, month, 1)
        df_predict = df_predict[df_predict.index < first_day_of_month]

        df_backtest = df_predict.groupby('code').apply(add_label_and_return, include_groups=False)

        # 删除缺失值
        # df_backtest = df_backtest.dropna(subset=feature_cols + ['label','future_return'])
        df_backtest = df_backtest.dropna(subset=[ 'mom_20', 'volatility_20', 'val_score', 'label', 'future_return' ])
        return df_backtest

    # ------------------------------------------------------------------
    # 最终数据集，包含技术指标，并做横截面的归一化处理，用于训练和验证，和测试
    # 每日数据预测的输入数据也用这个函数来构建
    # ------------------------------------------------------------------
    def make_dataframe(self, stock_pool:str):
        '''stock_pool: 股票池，截止到当天的交易所日历的最后交易日的1年数据，来构建预测数据集
        比如：今天2025-10-10，则从2024-10-10开始，到2025-10-10结束
        '''

        today = datetime.now().strftime("%Y-%m-%d")

        # 如果是当天的第二次预测，原始数据文件就不用再创建了，节约时间
        predict_panel_file = self.base_dir / stock_pool / f"predict_panel_{today}.csv"
        if predict_panel_file.exists():
            print("Data panel exists, no need to create again...")
            df_panel = pd.read_csv(predict_panel_file, parse_dates = True, index_col = 'date')
            df_predict = self.prepare_dataset(df_panel,stock_pool)
            return df_predict

        # 删除老的文件
        for file_path in (self.base_dir / stock_pool).glob("predict_panel_*.csv"):
            print(file_path)
            file_path.unlink()

        span = 3            # 年份跨度，1年就够了，为回溯测试，取3年的数据
        if not (self.base_dir/stock_pool).exists():
            return

        # 取得日历的最后一天，往前数1年 span，取得那年的一月1日为数据集的开始日期
        end_date_str = self.calendar['calendar_date'].max()
        y,*md = end_date_str.split("-")
        start_date = f"{str(int(y) - span)}-01-01"

        df_trading_days = self.calendar[self.calendar['calendar_date'] >= start_date].copy()

        all_trading_days = pd.to_datetime(df_trading_days.loc[df_trading_days['is_trading_day']==1]['calendar_date']).sort_values().unique()

        all_dfs = []
        stock_list = self.get_stock_list(stock_pool=stock_pool)

        for code in tqdm(stock_list, desc="prepare dataset"):
            stock_data_file = self.working_dir / f"{code}.csv"
            if not stock_data_file.exists():
                bs.login()
                BaostockOps()._update_stock_data(code)
                bs.logout()
            df = pd.read_csv(self.working_dir/f"{code}.csv")
            # df['stock_code'] = code
            df.set_index('date', inplace=True)

            # 与证交所交易日对齐，个股不交易的日子向前填充，即 若下一日不交易，则今日数据填入
            df_aligned = self.align_stock_to_calendar(df, all_trading_days)
            # df_aligned.to_csv(f"{code}.aligned.csv", index=False)
            all_dfs.append(df_aligned)

        df_panel = pd.concat(all_dfs).sort_index()  # 按日期排序

        df_panel.to_csv(self.base_dir/ stock_pool / f"predict_panel_{today}.csv", index = True)

        df_predict = self.prepare_dataset(df_panel,stock_pool)
        return df_predict

#        if not predict_mode:.
#            for col in feature_cols:
#                ic = spearmanr(df_final[col], df_final['future_return'])[0]
#                print(f"{col}: Rank IC = {ic:.4f}")
#

    def cal_weights(self, df_predict:pd.DataFrame, stock_pool:str, feature_cols, date):
        last_date = df_predict.index.max()
        if stock_pool == "zz500":
            factor = AutoFactorSelector(ic_threshold=0.02)
            # 10日收益阈值3%, 为默认值
            df_backtest = self.make_backtest_dataset(df_predict, last_date)
        elif stock_pool == "hs300":                 #这个还没有好,要加上ROE数据
            factor = AutoFactorSelector_HS300()
            df_backtest = self.make_backtest_dataset(df_predict, last_date)
        elif stock_pool == "zz1000":
            factor = AutoFactorSelector_CS1000()
            # 5日收益阈值5%~8%,暂设6%
            # 持有期缩短：中证500拿5-10天，中证1000可能3-5天就要止盈，因为轮动太快。
            # 止盈阈值提高：小盘股弹性大，3%可能只是起步，可以设为5%-8%。
            df_backtest = self.make_backtest_dataset(df_predict, last_date, forward_days=5,threshold=0.06)
        else:
            raise ValueError("Invalid stock pool")

        # 取上月数据生成的backtest数据集
        # df_backtest = pd.read_csv(program.base_dir/ stock_pool/ "backtest.dataset.csv", index_col='date')
        # date = f"{year}-{month:02d}-01"         # 用当前的日期，取截止到上个月底的数据
        # df_final = self.to_backtest_dataset(df_backtest, date=date)
        #

        # 保存backtest数据集, 以便查验，可不保存
        # df_backtest.to_csv(self.base_dir / stock_pool / "backtest.dataset.csv")

        factor.fit(
            df_backtest,
            # factor_cols= feature_cols           #['reversal_5', 'pb_inv', 'ps_inv', 'pe_inv', 'volatility_20']
        )
        factor.report()
        weights = factor.weights
        sameday_last_month = date - relativedelta(months=1)
        year,month = sameday_last_month.year, sameday_last_month.month
        json.dump(weights, open(self.base_dir / stock_pool / f"weights.{year}.{month}.json", "w"))
        return weights

    def predict(self, stock_pool, df_predict, val_end:str):
        # df_predict = pd.read_csv(self.base_dir /stock_pool/'predict.dataset.csv', index_col='date')
        # df_predict.set_index('date', inplace=True)

        if 'code' not in df_predict.columns:            # code 也要成为index之一？
            # 如果 stock_code 在 MultiIndex 的第二层
            df_predict = df_predict.reset_index(level=0)  # 假设 index 是 (date, stock_code)

        # 计算composite score的
        if stock_pool == "zz500":
            feature_cols = [
            'reversal_5',          # 替代 mom_5
            'pb_inv',              # 强价值因子
            'ps_inv',              # 强价值因子
            'pe_inv',              # 辅助
            'volatility_20',       # 风险控制
            'turn_5',               # 流动性
            # 'future_return'
            # 'value_reversal'    # 可单独测试，也可作为组合输入
        ]
        elif stock_pool == 'zz1000':
            feature_cols = [
            'reversal_5',          # 强动量因子
            'pb_inv',              # 强价值因子
            'ps_inv',              # 强价值因子
            'turn_5',               # 流动性
            ]
        elif stock_pool == 'hs300':             # roe呢？？？？
            feature_cols = [
            'pe_inv',       # 🔥 核心：市盈率倒数 (大盘股最有效因子之一)
            'pb_inv',       # 🔥 核心：市净率倒数
            'ps_inv',       # 辅助：市销率
            'volatility_20',# 🔥 核心：低波异象 (在大盘股中极有效)
            'reversal_5',   # ⚠️ 观察：短期反转在大盘股效果不稳定，由数据决定去留
            'turn_5'        # ⚠️ 辅助：流动性，权重通常较低
            ]
        else:
            raise ValueError('Invalid stock pool')

        X = df_predict[['code','close'] + feature_cols]
        X = X[X.index <= val_end]
        last_date = X.index.max()           # 用于预测的数据集的最后一天，作为计算因子所用数据集的参照

        # start_date = datetime.strptime(last_date,"%Y-%m-%d") + timedelta(days=-60)
        # 预测数据只取至今（数据集中最新日期）的60天
        start_date = last_date + timedelta(days=-60)
        start_date = start_date.strftime("%Y-%m-%d")
        X_val = X[ (X.index >= start_date) & (X.index <=last_date)]

        df_val = X_val.copy()

        # 保存到csv用于验证,待正式发布时，可以删除。
        # df_val.to_csv(self.base_dir /stock_pool/f'predict.val.{val_end}.csv', index=True)
        last_month = last_date - relativedelta(months=1)
        year  = last_month.year
        month = last_month.month

        # year, month = self.last_month(date=last_date)
        weights_file = self.base_dir / stock_pool / f"weights.{year}.{month}.json"
        print(f"Using weights from {weights_file}")

        # 取得或计算因子的权重
        if not weights_file.exists():
            print(f"Warning: weights file {weights_file} not found, calculating weights...")
            weights = self.cal_weights(df_predict,stock_pool, feature_cols, last_date)
        else:
            weights = json.load(open(weights_file, "r"))

        # 计算综合得分，composite score
        composite_score = pd.Series(0.0, index=df_val.index)
        for col in weights:
            if col in X_val.columns:
                composite_score += weights[col] * X_val[col]
            else:
                print(f"Warning: {col} not in input data")
        df_val['composite_score'] = composite_score.round(5)

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

        df_predict = top_stocks[top_stocks.index.get_level_values('date') == last_date][['code', 'close', 'composite_score', 'rank']]

        df_stockname = pd.read_csv("stock_industry.csv")
        # df_stockname.set_index('code')

        df_predict = df_predict.join(df_stockname.set_index('code')[['code_name', 'industry']], on='code')

        print(df_predict)

        prediction_history_file = self.base_dir / stock_pool / "predictions.csv"
        if prediction_history_file.exists():
            df_history = pd.read_csv(prediction_history_file, parse_dates=True, index_col='date')
#            if datetime.strftime(last_date,"%Y-%m-%d") in df_history.index:
#                print(f"{last_date} 的预测结果已存在，无需重复记录")
#                return
            # 重置df_predict的索引以匹配历史数据结构
            # df_predict.reset_index(inplace=True)
            # df_history_reset = df_history.reset_index()
            # 合并数据
            df_predict.index = df_predict.index.astype(str)
            df_history.index = df_history.index.astype(str)
            df_combined = pd.concat([df_predict, df_history], axis=0)
            # 保持索引为字符串类型，仅保留日期部分
#            if hasattr(df_combined.index, 'date'):
#                # 如果索引是datetime类型，先提取日期部分再转为字符串
#                df_combined.index = df_combined.index.date.astype(str)
#            else:
#                # 如果已经是字符串类型，保持原样
#                df_combined.index = df_combined.index.astype(str)
            # 去除重复项
            df_combined = df_combined.drop_duplicates(subset=['code','rank', 'composite_score'],keep='last')
            # 按日期排序，最新的在最前面
            # df_combined = df_combined.sort_values(['date'], ascending=False)
            df_predict = df_combined.copy()

        df_predict.to_csv(prediction_history_file, encoding='utf-8',index=True)
        print("predictions saved.")

        # 统计 Top 10% 股票中，未来5天收益 ≥3% 的比例
        # top10 = df_val[df_val['is_top10']]
        # print(top10.head())

        # top10.to_csv('top10.csv')

        # hit_rate = (top10['future_return'] >= 0.03).mean()
        # print(f"Top 10% 股票中，5日涨幅≥3% 的比例: {hit_rate:.2%}")
        # ic, pval = spearmanr(score, y_continuous)
        # print(f"Composite Score Rank IC: {ic:.4f}")

    def get_portfolio(self)->list[str]:
        '''获取self.base_dir目录下的第一级目录名，写入列表后，返回该列表
        '''
        portfolio_list = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                portfolio_list.append(item.name)
        return portfolio_list
