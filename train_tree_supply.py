tic = time.time()
counter = 0 

def find_best_rf(row): 
    if row.rf ==1:
        return row.yhat_rf
    else:   
        return 0
#eval['rf_best'] = eval.apply(find_best_rf, axis = 1)

def find_best_xgb(row): 
    if row.xgb ==1:
        return row.yhat_xgb
    else:   
        return 0
# eval['xgb_best'] = eval.apply(find_best_xgb, axis = 1)


def find_best_etree(row): 
    if row.etree ==1:
        return row.yhat_etree
    else:   
        return 0
# eval['etree_best'] = eval.apply(find_best_etree, axis = 1)


def find_best_gboost(row): 
    if row.gboost ==1:
        return row.yhat_gboost
    else:   
        return 0
# eval['gboost_best'] = eval.apply(find_best_gboost, axis = 1)

def find_best_lgbm(row): 
    if row.lgbm ==1:
        return row.yhat_lgbm
    else:   
        return 0
# eval['lgbm_best'] = eval.apply(find_best_lgbm, axis = 1)



def find_best_dt(row): 
    if row['dt'] ==1:
        return row.yhat_dt
    else:   
        return 0
# eval['dt_best'] = eval.apply(find_best_dt, axis = 1)

def find_best_ada(row): 
    if row.ada ==1:
        return row.yhat_ada
    else:   
        return 0
# eval['ada_best'] = eval.apply(find_best_ada, axis = 1)

def find_best_knn(row): 
    if row.knn ==1:
        return row.yhat_knn
    else:   
        return 0
    
def find_best_cat(row): 
    if row.cat ==1:
        return row.yhat_cat
    else:   
        return 0
# eval['knn_best'] = eval.apply(find_best_knn, axis = 1)



# subset = df3[df3['model'].isin(lsua)] # 13080 row 


lsa,lsb =[],[]


group1 = df3.groupby('model')
#ls_prophet_prediction= []
for g1 in group1.groups:
    groupm = group1.get_group(g1)
    print('start training model:',groupm['model'].unique())
    numerical = groupm[['sum_amount', 'sum_qty', 'plant', 'pressure',
       'windbearing', 'cloudcover', 'icon_clear-day', 'icon_cloudy',
       'icon_fog', 'icon_partly-cloudy-day', 'icon_partly-cloudy-night',
       'icon_rain', 'icon_sleet', 'icon_snow', 'icon_wind', 'moonphase',
       'humidity', 'windspeed', 'temperaturehigh', 'temperaturelow',
       'dewpoint', 'visibility', 'holiday_name_cny_day',
       'holiday_name_dragon_boat_day', 'holiday_name_labor_day',
       'holiday_name_mid_autumn_day', 'holiday_name_national_day',
       'holiday_name_new_year_day', 'holiday_name_tomb_swipe_day', 
       'model_code', 'base_price', 'promotion_price', 'lifestage',
       'total_stock', 'weight_kg', 'width_cm', 'length_cm', 'height_cm']]
    
    #df_cat=  groupm[['weekid', 'model_code','date']]
    
    length_df = groupm.shape[0]
    print('shape of the model:',length_df)
    lsweek = ['202044', '202045', '202046','202047', '202048', '202049', '202050', '202051', '202052','202053', '202101', '202102']
    
    
#     df_train = groupm.iloc[:length_df - 4]
#     df_val = groupm.iloc[len(groupm)-4:]
    df_train = groupm[~groupm['weekid'].isin(lsweek)]
    df_val = groupm[groupm['weekid'].isin(lsweek)]
    #print(df_val['weekid'])
    label = numerical.pop('sum_qty')
    features = numerical
    selected = feature_selection(features,label,numerical)
#     print(selected)
      
    if selected == []:
        #print('using all features')
        data = numerical
        train_df = df_train
        val_df = df_val
        
    else:
        #print('selected features are:',selected)
        #df_num = groupm[selected]
        #sel = selected
        #print('selected feature are:',selected)
        target = 'sum_qty'
        selected.append(target)
        train_df = df_train[selected]
        val_df = df_val[selected]
        
    Y_train = train_df.pop('sum_qty')
    X_train = train_df
    Y_valid = val_df.pop('sum_qty')
    X_valid = val_df
    
    rf =  RandomForestRegressor().fit(X_train,Y_train)
    xgb = XGBRegressor().fit(X_train,Y_train)
    etree = ExtraTreesRegressor().fit(X_train,Y_train)
    gboost = GradientBoostingRegressor().fit(X_train,Y_train)
    lgbm = LGBMRegressor().fit(X_train,Y_train)
    dt = DecisionTreeRegressor().fit(X_train,Y_train)
    knn = KNeighborsRegressor().fit(X_train,Y_train)
    ada = AdaBoostRegressor().fit(X_train,Y_train)
    cat = CatBoostRegressor().fit(X_train,Y_train,silent=True)
    
    ################################################################
   
    yhat_rf = rf.predict(X_valid)
    yhat_xgb = xgb.predict(X_valid)
    yhat_etree = etree.predict(X_valid)
    yhat_gboost = gboost.predict(X_valid)
    yhat_lgbm = lgbm.predict(X_valid)
    yhat_dt = dt.predict(X_valid)
    yhat_knn = knn.predict(X_valid)
    yhat_ada = ada.predict(X_valid)
    yhat_cat = cat.predict(X_valid)
   
    ################################################################
    
    d1 = {'week':df_val['weekid'].values,
          'model':df_val['model'].values,
          'y':df_val['sum_qty'].values,
          'yhat_rf':yhat_rf,
          'yhat_xgb':yhat_xgb,
          'yhat_etree':yhat_etree,
          'yhat_gboost':yhat_gboost,
          'yhat_lgbm':yhat_lgbm,
                    'yhat_dt':yhat_dt,
                    'yhat_knn':yhat_knn,
                    'yhat_ada':yhat_ada,
                    'yhat_cat':yhat_cat
    }
    ################################################################
    eval= pd.DataFrame(d1)
    eval['f_gap_rf'] = abs(eval['yhat_rf'] - eval['y']) /eval['yhat_rf']
    eval['f_gap_xgb'] = abs(eval['yhat_xgb'] - eval['y']) /eval['yhat_xgb']
    eval['f_gap_etree'] = abs(eval['yhat_etree'] - eval['y']) /eval['yhat_etree']
    eval['f_gap_gboost'] = abs(eval['yhat_gboost'] - eval['y']) /eval['yhat_gboost']
    eval['f_gap_lgbm'] = abs(eval['yhat_lgbm'] - eval['y']) /eval['yhat_lgbm']
    eval['f_gap_dt'] = abs(eval['yhat_dt'] - eval['y']) /eval['yhat_dt']
    eval['f_gap_knn'] = abs(eval['yhat_knn'] - eval['y']) /eval['yhat_knn']
    eval['f_gap_ada'] = abs(eval['yhat_ada'] - eval['y']) /eval['yhat_ada']
#     eval['f_gap_cat'] = abs(eval['yhat_cat'] - eval['y']) /eval['yhat_cat']
    
    eval['f_gap_best']= eval[['f_gap_rf','f_gap_xgb', 
          'f_gap_etree','f_gap_gboost','f_gap_lgbm','f_gap_dt',
        'f_gap_knn','f_gap_ada']].min(axis=1)
    
    
    eval['rf'] = np.where(eval["f_gap_best"] == eval["f_gap_rf"], 1,0)
    eval['xgb'] = np.where(eval["f_gap_best"] == eval["f_gap_xgb"], 1,0)
    eval['etree'] = np.where(eval["f_gap_best"] == eval["f_gap_etree"], 1,0)
    eval['gboost'] = np.where(eval["f_gap_best"] == eval["f_gap_gboost"], 1,0)
    eval['lgbm'] = np.where(eval["f_gap_best"] == eval["f_gap_lgbm"], 1,0)
    eval['dt'] = np.where(eval["f_gap_best"] == eval["f_gap_dt"], 1,0)
    eval['ada'] = np.where(eval["f_gap_best"] == eval["f_gap_ada"], 1,0)
    eval['knn'] = np.where(eval["f_gap_best"] == eval["f_gap_knn"], 1,0)
#     eval['cat'] = np.where(eval["f_gap_best"] == eval["f_gap_cat"], 1,0)
    
 
    eval['rf_best'] = eval.apply(find_best_rf, axis = 1)
    eval['xgb_best'] = eval.apply(find_best_xgb, axis = 1)
    eval['etree_best'] = eval.apply(find_best_etree, axis = 1)
    eval['gboost_best'] = eval.apply(find_best_gboost, axis = 1)
    eval['lgbm_best'] = eval.apply(find_best_lgbm, axis = 1)
    eval['dt_best'] = eval.apply(find_best_dt, axis = 1)
    eval['ada_best'] = eval.apply(find_best_ada, axis = 1)
    eval['knn_best'] = eval.apply(find_best_knn, axis = 1)
#     eval['cat_best'] = eval.apply(find_best_cat, axis = 1)
    eval['duplicate'] = eval['rf'] + eval['xgb']+ eval['etree']+ eval['gboost']+ eval['lgbm']+ eval['dt']+ eval['ada']+ eval['knn']+ eval['ada']
    eval['yhat_best'] = eval['rf_best'] + eval['xgb_best']+ eval['etree_best']+ eval['gboost_best']+ eval['lgbm_best']+ eval['dt_best']+ eval['ada_best']+ eval['knn_best']
    eval['yhat_best1'] = eval['yhat_best']/ eval['duplicate']
    eval = eval[['week','model','y','yhat_best1']]
    
    
    

    gap = eval.groupby(['model']).agg({'y': 'sum',  'yhat_best1': 'sum'}).reset_index()  

#      'yhat_dt': 'sum','yhat_knn': 'sum','yhat_ada': 'sum','yhat_cat': 'sum'
#                                           }).reset_index()    
#     gap['gap_rf'] = abs(gap['y']-gap['yhat_rf']) / gap['yhat_rf']
#     gap['gap_xgb'] = abs(gap['y']-gap['yhat_xgb']) / gap['yhat_xgb']
#     gap['gap_etree'] = abs(gap['y']-gap['yhat_etree']) / gap['yhat_etree']
#     gap['gap_gboost'] = abs(gap['y']-gap['yhat_gboost']) / gap['yhat_gboost']
#     gap['gap_lgbm'] = abs(gap['y']-gap['yhat_lgbm']) / gap['yhat_lgbm']
#     gap['gap_dt'] = abs(gap['y']-gap['yhat_dt']) / gap['yhat_dt']
#     gap['gap_knn'] = abs(gap['y']-gap['yhat_knn']) / gap['yhat_knn']
#     gap['gap_ada'] = abs(gap['y']-gap['yhat_ada']) / gap['yhat_ada']
#     gap['gap_cat'] = abs(gap['y']-gap['yhat_cat']) / gap['yhat_cat']
    
#     gap['gap_best']= gap[['gap_rf','gap_xgb', 
#           'gap_etree','gap_gboost','gap_lgbm','gap_dt',
#         'gap_knn','gap_ada','gap_cat']].min(axis=1)
    
    gap['prediction'] = abs(gap['y'] / gap['yhat_best1'])/gap['yhat_best1']
    counter+=1
    
#     gap = gap[['model','y','prediction','gap_best']]

    lsa.append(eval)
    lsb.append(gap)
    print('counter number:',counter)
    print('gap of the model is:',gap['prediction'].values)
    print('--------------------continue-----------------------')
    
    
    
#     ###############################################################
#     ls.append(gap)
# output = pd.concat(ls)
print('--------------------end-----------------------')

toc = time.time()
print(f"Runtime of the program is: {(toc - tic)/60}"+ ' '+'mins')    