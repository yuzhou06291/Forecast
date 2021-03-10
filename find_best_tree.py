print("Hello world")

import pandas as pd
import os
import numpy as np


df = pd.read_csv('C:/Users/backup/Desktop/case3_pre.csv')

eval = df
eval['f_gap_rf'] = abs(eval['yhat_rf'] - eval['y']) /eval['yhat_rf']
eval['f_gap_xgb'] = abs(eval['yhat_xgb'] - eval['y']) /eval['yhat_xgb']
eval['f_gap_etree'] = abs(eval['yhat_etree'] - eval['y']) /eval['yhat_etree']
eval['f_gap_gboost'] = abs(eval['yhat_gboost'] - eval['y']) /eval['yhat_gboost']
eval['f_gap_lgbm'] = abs(eval['yhat_lgbm'] - eval['y']) /eval['yhat_lgbm']
eval['f_gap_dt'] = abs(eval['yhat_dt'] - eval['y']) /eval['yhat_dt']
eval['f_gap_knn'] = abs(eval['yhat_knn'] - eval['y']) /eval['yhat_knn']
eval['f_gap_ada'] = abs(eval['yhat_ada'] - eval['y']) /eval['yhat_ada']
# eval['f_gap_cat'] = abs(eval['yhat_cat'] - eval['y'])

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
eval['duplicate'] = eval['rf'] + eval['xgb']+ eval['etree']+ eval['gboost']+ eval['lgbm']+ eval['dt']+ eval['ada']+ eval['knn']



def find_best_rf(row): 
  if row.rf ==1:
    return row.yhat_rf
  else:   
    return 0
eval['rf_best'] = eval.apply(find_best_rf, axis = 1)

def find_best_xgb(row): 
  if row.xgb ==1:
    return row.yhat_xgb
  else:   
    return 0
eval['xgb_best'] = eval.apply(find_best_xgb, axis = 1)


def find_best_etree(row): 
  if row.etree ==1:
    return row.yhat_etree
  else:   
    return 0
eval['etree_best'] = eval.apply(find_best_etree, axis = 1)


def find_best_gboost(row): 
  if row.gboost ==1:
    return row.yhat_gboost
  else:   
    return 0
eval['gboost_best'] = eval.apply(find_best_gboost, axis = 1)

def find_best_lgbm(row): 
  if row.lgbm ==1:
    return row.yhat_lgbm
  else:   
    return 0
eval['lgbm_best'] = eval.apply(find_best_lgbm, axis = 1)

def find_best_dt(row): 
  if row['dt'] ==1:
    return row.yhat_dt
  else:   
    return 0
eval['dt_best'] = eval.apply(find_best_dt, axis = 1)

def find_best_ada(row): 
  if row.ada ==1:
    return row.yhat_ada
  else:   
    return 0
eval['ada_best'] = eval.apply(find_best_ada, axis = 1)

def find_best_knn(row): 
  if row.knn ==1:
    return row.yhat_knn
  else:   
    return 0
eval['knn_best'] = eval.apply(find_best_knn, axis = 1)

eval['yhat_best'] = eval['rf_best'] + eval['xgb_best']+ eval['etree_best']+ eval['gboost_best']+ eval['lgbm_best']+ eval['dt_best']+ eval['ada_best']+ eval['knn_best']
eval['yhat_best1'] = eval['yhat_best']/ eval['duplicate']
eval = eval[['week','model','y','yhat_best1']]
