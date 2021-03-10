tic = time.time()

def feature_sel_5000(data):
    Y = data['sum_qty']
    X = data.drop('sum_qty', 1)
    x, x_TEST, y, y_TEST = train_test_split(X, Y, train_size=0.9, random_state=42)
    
    rf = RandomForestRegressor().fit(x,y)
    xgb = XGBRegressor().fit(x,y)
    etree = ExtraTreesRegressor().fit(x,y)
    gb = GradientBoostingRegressor().fit(x,y)
    lgb = LGBMRegressor().fit(x,y)
    dt = DecisionTreeRegressor().fit(x,y)
#     print(Y)

    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_
    etree_imp = etree.feature_importances_
    gb_imp = gb.feature_importances_
    lgb_imp= lgb.feature_importances_   
    dt_imp = dt.feature_importances_
#     print(rf_imp)
    
# # summarize feature importance
#     for i,v in enumerate(rf_imp):
#         print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
#         plt.bar([x for x in range(len(rf_imp))], rf_imp)
#         plt.show()
    
    d = {'features':(X.columns).to_numpy(),
    'rf':rf_imp,
     'xgb':xgb_imp,
     'etree':etree_imp,
     'gb':gb_imp,
     'lgb':lgb_imp,
         'dt':dt_imp
#    'knn':knn_imp,
    # 'catboost':lgb_imp
          #'cat':cat_imp/np.sum(cat_imp)
    }
    feat_table = pd.DataFrame(d)
    feat_table['lgb'] = feat_table['lgb']/(feat_table['lgb'].sum())
    feat_table['mean'] = feat_table.mean(axis=1)
# feat_table
    feat_table= feat_table.loc[feat_table['mean']>=0.01]
    selected = feat_table['features'].tolist()
    return selected