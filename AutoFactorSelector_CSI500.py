import pandas as pd
import numpy as np
from scipy.stats import spearmanr

class AutoFactorSelector_CSI500:
    def __init__(self, ic_threshold=0.02, pval_threshold=0.05, min_obs=100):
        self.ic_threshold = ic_threshold
        self.pval_threshold = pval_threshold
        self.min_obs = min_obs
        self.selected_factors = []
        self.weights = {}
        self.history_ic = pd.DataFrame()
    
    def fit(self, df, factor_cols=['reversal_5', 'pb_inv', 'ps_inv', 'pe_inv', 'volatility_20'], future_return_col='future_return', date_col='date'):
        """
        df: DataFrame，包含日期、所有因子、future_return
        factor_cols: 候选因子列表，如 ['mom_5', 'pb_inv', ...]
        """
        # 确保日期列存在
        if date_col not in df.columns:
            # 检查索引名称是否与现有列冲突
            index_names = df.index.names
            conflicting_names = [name for name in index_names if name in df.columns and name is not None]
            if conflicting_names:
                # 如果有冲突，drop 重复的索引级别
                df = df.reset_index(level=conflicting_names, drop=True)
                # 然后重新添加剩余的索引
                remaining_levels = [name for name in index_names if name not in conflicting_names and name is not None]
                if remaining_levels:
                    df = df.reset_index()
            else:
                df = df.reset_index()
        
        # 按月分组
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['month'] = df[date_col].dt.to_period('M')
        monthly_ic = []
        
        for month, group in df.groupby('month'):
            if len(group) < self.min_obs:
                continue
                
            ic_dict = {'month': month}
            for col in factor_cols:
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
        
        # 转为 DataFrame
        ic_df = pd.DataFrame(monthly_ic).set_index('month')
        self.history_ic = ic_df
        
        # 使用最近1个月的数据选择因子（可改为滚动3个月平均）
        latest_ic = ic_df.iloc[-1]
        
        selected = []
        ic_values = []
        
        for col in factor_cols:
            ic_key = f'{col}_ic'
            pval_key = f'{col}_pval'
            
            if ic_key in latest_ic.index:
                ic_val = latest_ic[ic_key]
                pval = latest_ic[pval_key]
                
                # 选择：IC显著为正，且统计显著
                if ic_val > self.ic_threshold and pval < self.pval_threshold:
                    selected.append(col)
                    ic_values.append(ic_val)
        
        # 计算权重：按IC比例分配（IC越大，权重越高）
        if selected:
            total_ic = sum(ic_values)
            self.weights = {
                col: ic / total_ic for col, ic in zip(selected, ic_values)
            }
            self.selected_factors = selected
        else:
            # 若无有效因子，回退到默认（可自定义）
            print("⚠️ 无显著因子，使用默认价值+反转组合")
            self.selected_factors = ['pb_inv', 'ps_inv', 'reversal_5']
            self.weights = {'pb_inv': 0.4, 'ps_inv': 0.4, 'reversal_5': 0.2}
    
    def get_composite_score(self, df):
        """计算 Composite Score"""
        score = pd.Series(0.0, index=df.index)
        for col in self.selected_factors:
            if col in df.columns:
                score += self.weights[col] * df[col]
            else:
                print(f"Warning: {col} not in input data")
        return score
    
    def report(self):
        """打印当前选择结果"""
        print("\n📊 自动因子选择器报告:")
        print(f"选中因子: {self.selected_factors}")
        print(f"对应权重: {self.weights}")
        if not self.history_ic.empty:
            print("\n最近一个月因子 IC:")
            latest = self.history_ic.iloc[-1]
            for col in self.selected_factors:
                ic = latest.get(f'{col}_ic', np.nan)
                pval = latest.get(f'{col}_pval', np.nan)
                print(f"  {col}: IC={ic:.4f}, p={pval:.3f}")
                

# df_historical = pd.read_csv(".local/final_df_train.csv")
# 每月1日运行（使用截至上月末的数据）
# selector = AutoFactorSelector(ic_threshold=0.02)
# selector.fit(
#    df_historical, 
#    factor_cols=['reversal_5', 'pb_inv', 'ps_inv', 'pe_inv', 'volatility_20']
#)
# selector.report()
#score = selector.get_composite_score(df_historical)
#weights = selector.weights
#for k,v in weights.items():
#    print(f"{k}: {v}")