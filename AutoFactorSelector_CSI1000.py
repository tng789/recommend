import pandas as pd
import numpy as np
from scipy.stats import spearmanr

class AutoFactorSelector_CSI1000:
    """
    专为中证1000 (小盘股) 设计的自动因子选择器
    特点：
    1. 默认剔除 PE (因小盘股盈利不稳定)
    2. 优先强化 反转 (Reversal) 和 换手 (Turnover) 因子
    3. 对噪音更敏感，设置更严格的显著性门槛
    """
    
    def __init__(self, 
                 ic_threshold=0.025,      # 小盘股噪音大，IC 门槛略调高
                 pval_threshold=0.05,     # 显著性水平
                 min_obs=200,             # 小盘股样本多，要求更多样本
                 lookback_months=6):      # 小盘股风格切换快，只看最近半年
        self.ic_threshold = ic_threshold
        self.pval_threshold = pval_threshold
        self.min_obs = min_obs
        self.lookback_months = lookback_months
        
        # 🎯 中证1000 核心候选因子池 (已主动剔除 pe_inv)
        # 逻辑：小盘股看情绪 (反转)、看资金 (换手)、看销售/资产 (PS/PB)，不看利润 (PE)
        self.base_factor_cols = [
            'reversal_5',   # 核心：超跌反弹
            'ps_inv',       # 核心：市销率倒数 (比PE更稳)
            'pb_inv',       # 辅助：市净率倒数
            'turn_5',       # 核心：流动性确认
            # 'volatility_20' # 默认不选，除非数据证明低波在小盘股有效
        ]
        
        self.selected_factors = []
        self.weights = {}
        self.history_ic = pd.DataFrame()
    
    def fit(self, df, future_return_col='future_return', date_col='date'):
        """
        使用最近 N 个月的数据计算因子 IC 并确定权重
        """
        print(f"🔍 开始为中证1000筛选因子 (回溯最近 {self.lookback_months} 个月)...")
        
        # 确保日期列存在
        if date_col not in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=date_col)
            else:
                df = df.reset_index()
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # 只取最近 N 个月的数据
        unique_months = df['month'].unique()
        if len(unique_months) < self.lookback_months:
            print(f"⚠️ 警告：数据只有 {len(unique_months)} 个月，少于设定的 {self.lookback_months} 个月，将使用全部数据。")
            selected_months = sorted(unique_months)
        else:
            selected_months = sorted(unique_months)[-self.lookback_months:]
            
        df_recent = df[df['month'].isin(selected_months)]
        
        monthly_ic = []
        
        # 按月计算 IC
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
            raise ValueError("没有足够的有效数据来计算 IC。")
            
        ic_df = pd.DataFrame(monthly_ic).set_index('month')
        self.history_ic = ic_df
        
        # --- 核心策略：计算平均 IC 并筛选 ---
        # 对小盘股，我们使用最近 N 个月的平均 IC，而不是单月，以平滑噪音
        avg_ic = ic_df.filter(like='_ic').mean()
        avg_pval = ic_df.filter(like='_pval').mean()
        
        selected = []
        ic_values = []
        
        print("\n📊 因子 IC 分析报告 (中证1000):")
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
            
            print(f"  {col:<15}: IC={ic_val:.4f}, P={pval:.3f} -> {status}")
        
        # --- 权重分配策略 (针对小盘股优化) ---
        if selected:
            # 基础权重：按 IC 比例分配
            total_ic = sum(ic_values)
            raw_weights = {col: ic / total_ic for col, ic in zip(selected, ic_values)}
            
            # 🚀 特殊增强逻辑：如果 reversal_5 被选中，强制提升其权重下限
            # 逻辑：小盘股反转效应通常是主导因子
            if 'reversal_5' in raw_weights:
                current_w = raw_weights['reversal_5']
                if current_w < 0.35: # 如果自然计算出的权重低于 35%
                    # 从其他因子借权，简单处理：重新归一化，保证 reversal 至少 0.4
                    print("  ⚡ 触发小盘股增强策略：强制提升 reversal_5 权重至 40%+")
                    raw_weights['reversal_5'] = 0.45
                    # 剩余权重按比例分配给其他因子
                    remaining_cols = [c for c in selected if c != 'reversal_5']
                    remaining_sum = sum(raw_weights[c] for c in remaining_cols)
                    if remaining_sum > 0:
                        scale = (1.0 - 0.45) / remaining_sum
                        for c in remaining_cols:
                            raw_weights[c] *= scale
            
            self.weights = raw_weights
            self.selected_factors = selected
        else:
            #  fallback: 如果没有任何因子显著，使用经验权重 (小盘股经典配置)
            print("\n⚠️ 未检测到显著因子，启用中证1000 默认经验权重 (强反转 + 高换手)")
            self.selected_factors = ['reversal_5', 'ps_inv', 'turn_5']
            self.weights = {
                'reversal_5': 0.50, # 强反转
                'ps_inv': 0.30,     # 低 PS
                'turn_5': 0.20      # 高换手
            }
    
    def get_composite_score(self, df):
        """计算 Composite Score"""
        if not self.selected_factors:
            raise ValueError("尚未运行 fit() 方法选择因子。")
            
        score = pd.Series(0.0, index=df.index)
        valid_count = 0
        
        for col in self.selected_factors:
            if col in df.columns:
                score += self.weights[col] * df[col]
                valid_count += 1
            else:
                # 在中证1000中，如果缺少关键因子，可能需要警告
                if col in ['reversal_5', 'ps_inv']:
                    print(f"⚠️ 严重警告：核心因子 {col} 不在输入数据中！")
        
        if valid_count == 0:
            return pd.Series(0.0, index=df.index)
            
        return score
    
    def report(self):
        """打印报告"""
        print("\n" + "="*40)
        print("🏆 中证1000 自动因子选择器报告")
        print("="*40)
        print(f"选中因子: {self.selected_factors}")
        print("最终权重:")
        for k, v in self.weights.items():
            print(f"  - {k}: {v:.4f}")
        print("="*40)
        
# 实例化 (针对中证1000优化)
#selector_1000 = AutoFactorSelector_CS1000(lookback_months=6)

# 训练 (使用过去6个月的中证1000数据)
# 假设 df_cs1000 是你的中证1000 历史数据
#selector_1000.fit(df_cs1000)

# 查看报告
#selector_1000.report()

# 生成得分 (用于今日预测)
# df_today_cs1000 是今日的中证1000 因子数据
#df_today_cs1000['composite_score'] = selector_1000.get_composite_score(df_today_cs1000)

# 选股
#top_stocks = df_today_cs1000.nlargest(5, 'composite_score') # 小盘股建议选少一点，如前5
#print(top_stocks[['stock_code', 'composite_score']])