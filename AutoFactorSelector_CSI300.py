import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 需要加入ROE数据，并修改代码

class AutoFactorSelector_CSI300:
    """
    专为沪深300 (大盘蓝筹) 设计的自动因子选择器
    特点：
    1. 核心依赖：估值 (PE/PB) + 质量 (隐含在低波/稳定中)
    2. 弱化短期反转，警惕“接飞刀”，更倾向于“低波”或“趋势”
    3. 长周期回溯 (12个月)，适应机构主导的稳定风格
    4. 若数据包含ROE，可轻松扩展 (代码预留逻辑)
    """
    
    def __init__(self, 
                 ic_threshold=0.015,      # 大盘股信号稳，门槛可略低，追求覆盖面
                 pval_threshold=0.05,     
                 min_obs=300,             # 样本要求高
                 lookback_months=12):     # 大盘股风格稳定，看长一点
        self.ic_threshold = ic_threshold
        self.pval_threshold = pval_threshold
        self.min_obs = min_obs
        self.lookback_months = lookback_months
        
        # 🎯 沪深300 核心候选因子池
        # 逻辑：
        # 1. PE/PB 是定价锚 (必须)
        # 2. 低波动 (Volatility) 在大盘股中是显著Alpha (防御+复利)
        # 3. 反转 (Reversal) 需谨慎，仅在超跌时有效，权重由数据决定
        # 4. 换手 (Turn) 权重降低，大盘股不靠换手率驱动
        self.base_factor_cols = [
            'pe_inv',       # 🔥 核心：市盈率倒数 (大盘股最有效因子之一)
            'pb_inv',       # 🔥 核心：市净率倒数
            'ps_inv',       # 辅助：市销率
            'volatility_20',# 🔥 核心：低波异象 (在大盘股中极有效)
            'reversal_5',   # ⚠️ 观察：短期反转在大盘股效果不稳定，由数据决定去留
            'turn_5'        # ⚠️ 辅助：流动性，权重通常较低
        ]
        
        self.selected_factors = []
        self.weights = {}
        self.history_ic = pd.DataFrame()
    
    def fit(self, df, future_return_col='future_return', date_col='date'):
        """
        使用最近 12 个月的数据计算因子 IC 并确定权重
        """
        print(f"🔍 开始为沪深300筛选因子 (回溯最近 {self.lookback_months} 个月)...")
        
        # 数据预处理
        if date_col not in df.columns:
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=date_col)
            else:
                df = df.reset_index()
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # 选取最近 N 个月
        unique_months = df['month'].unique()
        if len(unique_months) < self.lookback_months:
            print(f"⚠️ 警告：数据只有 {len(unique_months)} 个月，将使用全部数据。")
            selected_months = sorted(unique_months)
        else:
            selected_months = sorted(unique_months)[-self.lookback_months:]
            
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
            raise ValueError("没有足够的有效数据来计算 IC。")
            
        ic_df = pd.DataFrame(monthly_ic).set_index('month')
        self.history_ic = ic_df
        
        # --- 计算平均 IC ---
        avg_ic = ic_df.filter(like='_ic').mean()
        avg_pval = ic_df.filter(like='_pval').mean()
        
        selected = []
        ic_values = []
        
        print("\n📊 因子 IC 分析报告 (沪深300):")
        for col in self.base_factor_cols:
            ic_key = f'{col}_ic'
            pval_key = f'{col}_pval'
            
            ic_val = avg_ic.get(ic_key, np.nan)
            pval = avg_pval.get(pval_key, 1.0)
            
            status = "❌ 无效"
            # 沪深300中，只要IC>0且显著，我们就保留，因为大盘股因子很珍贵
            if not np.isnan(ic_val) and ic_val > self.ic_threshold and pval < self.pval_threshold:
                selected.append(col)
                ic_values.append(ic_val)
                status = "✅ 选中"
            elif not np.isnan(ic_val) and ic_val > 0: 
                # 即使未达阈值，如果是正相关且是大盘股核心因子(PE/PB/low_vol)，也酌情保留
                if col in ['pe_inv', 'pb_inv', 'volatility_20']:
                     selected.append(col)
                     ic_values.append(max(ic_val, 0.01)) # 赋予一个最小正权重
                     status = "⚠️ 保留 (核心因子)"
            
            print(f"  {col:<15}: IC={ic_val:.4f}, P={pval:.3f} -> {status}")
        
        # --- 权重分配策略 (针对大盘股优化) ---
        if selected:
            total_ic = sum(ic_values)
            raw_weights = {col: ic / total_ic for col, ic in zip(selected, ic_values)}
            
            # 🛡️ 特殊增强逻辑：确保“估值+低波”的主导地位
            # 逻辑：沪深300是价值投资的战场，必须保证 PE/PB/LowVol 的总权重 > 60%
            value_vol_cols = [c for c in selected if c in ['pe_inv', 'pb_inv', 'volatility_20']]
            current_value_vol_weight = sum(raw_weights.get(c, 0) for c in value_vol_cols)
            
            if current_value_vol_weight < 0.6:
                print(f"  ⚡ 触发大盘股增强策略：强制提升 [估值+低波] 组合权重至 60%+")
                # 简单重平衡：将其他因子（如反转、换手）的权重压缩，让位给价值因子
                other_cols = [c for c in selected if c not in value_vol_cols]
                
                # 目标：价值类占 0.65，其他类占 0.35
                target_value_weight = 0.65
                
                # 重新计算价值类内部比例
                value_ic_sum = sum(ic_values[selected.index(c)] for c in value_vol_cols if c in selected) # 简化处理，直接用当前权重比例
                # 更简单的做法：直接按比例缩放
                scale_up = target_value_weight / current_value_vol_weight
                for c in value_vol_cols:
                    raw_weights[c] *= scale_up
                
                # 压缩其他类
                remaining_weight = 1.0 - target_value_weight
                current_other_weight = sum(raw_weights.get(c, 0) for c in other_cols)
                if current_other_weight > 0:
                    scale_down = remaining_weight / current_other_weight
                    for c in other_cols:
                        raw_weights[c] *= scale_down
            
            # 归一化检查 (防止浮点数误差)
            total_w = sum(raw_weights.values())
            self.weights = {k: v/total_w for k, v in raw_weights.items()}
            self.selected_factors = selected
            
        else:
            # Fallback: 沪深300经典价值策略
            print("\n⚠️ 未检测到显著因子，启用沪深300 默认经验权重 (深度价值 + 低波)")
            self.selected_factors = ['pe_inv', 'pb_inv', 'volatility_20']
            self.weights = {
                'pe_inv': 0.40,
                'pb_inv': 0.30,
                'volatility_20': 0.30
            }
    
    def get_composite_score(self, df):
        """计算 Composite Score"""
        if not self.selected_factors:
            raise ValueError("尚未运行 fit() 方法。")
            
        score = pd.Series(0.0, index=df.index)
        for col in self.selected_factors:
            if col in df.columns:
                score += self.weights[col] * df[col]
            else:
                if col in ['pe_inv', 'pb_inv']:
                    print(f"⚠️ 严重警告：核心价值因子 {col} 缺失！")
        return score
    
    def report(self):
        print("\n" + "="*40)
        print("🏆 沪深300 自动因子选择器报告")
        print("="*40)
        print(f"选中因子: {self.selected_factors}")
        print(f"最终权重 (注重价值与稳定):")
        for k, v in self.weights.items():
            print(f"  - {k}: {v:.4f}")
        print("="*40)