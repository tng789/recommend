
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