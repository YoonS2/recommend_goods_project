# ! pip install parmap
# ! pip install pyodbc

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 정확도 함수
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import multiprocessing as mp #병렬처리 
import parmap
from sklearn.preprocessing import LabelEncoder
import pyodbc


import warnings
warnings.filterwarnings(action='ignore')


# 하이브에서 데이터 불러오기. 매출원시

# --db2연결
import ibm_db_dbi
# conn = ibm_db_dbi.connect("DRIVER={IBM DB2 ODBC DRIVER}; Database=PGSDW; Hostname=10.56.36.47; Port=50001; PROTOCOL=TCPIP; UID=gsseldw; PWD=gsseldw01", "", "") 
conn = ibm_db_dbi.connect("DRIVER={IBM DB2 ODBC DRIVER}; Database=PGSDW; Hostname=10.56.36.47; Port=50001; PROTOCOL=TCPIP; UID=gsanldw; PWD=gsanldw01", "", "") 
sql_0401 = """
select *
from anl1_mart.danbi_group_total0401 
where gr=1
;
 """
# sql_0416 = """
# select *
# from anl1_mart.danbi_group_total0416 as a

# where a.gr=1
# ;
# """


# sql_0501 = """
# select *
# from anl1_mart.danbi_group_total0501
# where gr=1
# ;
# """
# sql_gooddm = """
# select bd_item_lcls_cd,BD_ITEM_LCLS_NM ,BD_ITEM_MCLS_CD ,BD_ITEM_MCLS_NM ,BD_ITEM_SCLS_CD  ,BD_ITEM_SCLS_NM ,BD_ITEM_DCLS_CD 
# ,BD_ITEM_DCLS_NM ,trim(BD_ITEM_CD) as good_cd   ,trim(BD_ITEM_NM) AS GOODS_NM
# from lake_offline.gsucdw_tb_dm_bd_item_base
# where bizu_cd='1' 
# ;

# """

sql_gooddm = """
select *,trim(bd_item_cd) as goods_cd, trim(bd_item_nm) as goods_nm
          ,BD_ITEM_MCLS_CD AS LINE_CD
          ,BD_ITEM_MCLS_NM AS LINE_NM
          ,BD_ITEM_SCLS_CD AS CLASS_CD
          ,BD_ITEM_SCLS_NM AS CLASS_NM
from lake_offline.gsucdw_tb_dm_bd_item_base
where bizu_cd='1' 
;

"""


# with pyodbc.connect("DSN=Sample Cloudera Hive DSN",autocommit=True) as conn : df = pd.read_sql(sql,conn)
with pyodbc.connect("DSN=CloudHI",autocommit=True) as conn:
    amt_0401 = pd.read_sql(sql_0401,conn)
#     amt_0416 = pd.read_sql(sql_0416,conn)
#     amt_0501 = pd.read_sql(sql_0501,conn)
    good_dm = pd.read_sql(sql_gooddm,conn)


  
  conn = ibm_db_dbi.connect("DRIVER={IBM DB2 ODBC DRIVER}; Database=PGSDW; Hostname=10.56.36.47; Port=50001; PROTOCOL=TCPIP; UID=gsanldw; PWD=gsanldw01", "", "") 
sql_origin01 = """
select *,trim(ORIGIN_BIZPL_CD) as origin_bizpl_cd2
from gsanldw.ts_ms_bizpl
order by origin_bizpl_cd, open_dt desc 
;

"""

sql_event = '''
    SELECT *
    FROM GSANLDW.danbi_event_tm2
    ;
    '''

event = pd.read_sql( sql_event, conn ) 

origin01 = pd.read_sql( sql_origin01 , conn ) 




## 데이터정리
origin02= origin01.groupby("ORIGIN_BIZPL_CD2").head(1)
origin02=origin02.drop(['ORIGIN_BIZPL_CD'],axis=1)
origin02.rename(columns={'ORIGIN_BIZPL_CD2':'origin_bizpl_cd','BIZPL_NM':'bizpl_nm'},inplace=True)

amt1=amt_0401.copy()


amt_line2=pd.merge(amt1,origin02[['origin_bizpl_cd','GOODS_REGION_CD']],left_on='origin_bizpl_cd',right_on='origin_bizpl_cd',how='left')
amt_line2=pd.merge(left=amt_line2,right=good_dm[['class_cd','class_nm','goods_cd','goods_nm']],on='goods_cd')



## 이벤트 테이블

amt_line3=pd.merge(amt_line2,event[['PRSNT_GOODS_CD','E21','E11','DIST']],left_on='goods_cd',right_on='PRSNT_GOODS_CD',how='left')
amt_line3['event']='없음'
amt_line3['event']=np.where(amt_line3['E21']>0,'2+1',amt_line3['event'])
amt_line3['event']=np.where(amt_line3['E11']>0,'1+1',amt_line3['event'])
amt_line3['event']=np.where(amt_line3['DIST']>0,'할인',amt_line3['event'])


## 이상치 제거

def outlier1(x) :
    d={}
    d['qty_mean']=x['qty'].mean()
    d['qty_std']=x['qty'].std()
#     d['qty_q3']=x['qty'].quantile(q=0.75)
#     d['qty_q1']=x['qty'].quantile(q=0.25)
    d['amt_mean']=x['amt'].mean()
    d['amt_std']=x['amt'].std()
#     d['amt_q3']=x['amt'].quantile(q=0.75)
#     d['amt_q1']=x['amt'].quantile(q=0.25)
#     return pd.Series(d, index=['qty_mean', 'qty_std', 'qty_q1','qty_q3','amt_mean', 'amt_std','amt_q1','amt_q3'])
    return pd.Series(d, index=['qty_mean', 'qty_std','amt_mean', 'amt_std'])



start=time.time()
hdr_out1=amt_line3.groupby(["origin_bizpl_cd","class_cd","event"],as_index=False).apply(outlier1)
hdr_out1=hdr_out1.reset_index()
hdr_out1['amt_up']=hdr_out1['amt_mean']+hdr_out1['amt_std']*2
hdr_out1['amt_lp']=hdr_out1['amt_mean']-hdr_out1['amt_std']*2
hdr_out1['qty_up']=hdr_out1['qty_mean']+hdr_out1['qty_std']*2
hdr_out1['qty_lp']=hdr_out1['qty_mean']-hdr_out1['qty_std']*2
print("time :", (time.time() - start)/60)  # 현재시각 - 시작시간 = 실행 시간



amt_line02=pd.merge(left=amt_line3,right=hdr_out1, on=['origin_bizpl_cd','class_cd','event'],how='left')
amt_line02['del1']=np.where((amt_line02['qty']<amt_line02['qty_lp'])|(amt_line02['qty']>amt_line02['qty_up']),1,0)
amt_line02['del2']=np.where((amt_line02['amt']<amt_line02['amt_lp'])|(amt_line02['amt']>amt_line02['amt_up']),1,0)
amt_line02['del3']=np.where(amt_line02['qty']>5,1,0)

remove_index1=amt_line02[(amt_line02['event'].isin(['1+1','2+1']))&((amt_line02['del1']==1) | (amt_line02['del2']==1))].index
remove_index2=amt_line02[(amt_line02['event'].isin(['없음','할인']))&(amt_line02['del3']==1) ].index


amt_line03=amt_line02.drop(remove_index1)
amt_line03=amt_line03.drop(remove_index2)
amt_line03=amt_line03.reset_index()

#기준설정

group="line_cd"
# group="gr"

amt_line04=amt_line03.groupby(["bizpl_cl_div_cd","origin_bizpl_cd","sale_day",group,"goods_cd","goods_nm","event"]).sum().reset_index()[['bizpl_cl_div_cd','origin_bizpl_cd','sale_day',group,'goods_cd','goods_nm','event','qty','amt']]
amt_line04['dqty']=amt_line04['qty']/amt_line04['sale_day']
amt_line04['damt']=amt_line04['amt']/amt_line04['sale_day']



#TF-IDF구하기
## TF
tf_max=amt_line04.groupby(["origin_bizpl_cd",group]).max().reset_index()[['origin_bizpl_cd',group,'dqty']]
tf_max.rename(columns={'dqty':'max_dqty'},inplace=True)

bizpl_tf=pd.merge(left=amt_line04,right=tf_max,how='left',on=['origin_bizpl_cd',group])[['bizpl_cl_div_cd','origin_bizpl_cd',group,'goods_cd','dqty','max_dqty']]
bizpl_tf['tf']=0.1+0.9*(bizpl_tf['dqty']/bizpl_tf['max_dqty'])

## IDF
def totalidf(x) :
    d={}
    d['total']=len(set(x[x['dqty']!=0]['origin_bizpl_cd']))

    return pd.Series(d, index=['total'])
def storeidf(x) :
    d={}
    d['store']=len(set(x['origin_bizpl_cd']))
    
idf_total=amt_line04[['bizpl_cl_div_cd','origin_bizpl_cd','dqty']].groupby(['bizpl_cl_div_cd']).apply(totalidf).reset_index()
amt04_idf=pd.merge(left=amt_line04[amt_line04['dqty']!=0],right=idf_total,on='bizpl_cl_div_cd',how='left')
idf_store=amt04_idf.groupby(["bizpl_cl_div_cd",group,"goods_cd","total"]).apply(lambda d: len(set(d.origin_bizpl_cd))).reset_index()

bizpl_idf=idf_store.copy()
bizpl_idf.rename(columns={0:'store'},inplace=True)
bizpl_idf['idf']=np.log(bizpl_idf['total']/(bizpl_idf['store']+1))+0.1
bizpl_idf['ratio']=bizpl_idf['store']/bizpl_idf['total']


# IDF 이상치 제거

def outlier2(x) :
    d={}
    d['idf_q3']=x['idf'].quantile(q=0.75)
    d['idf_q1']=x['idf'].quantile(q=0.25)
    return pd.Series(d, index=['idf_q3', 'idf_q1'])
  
idf_six=bizpl_idf.groupby(['bizpl_cl_div_cd',group]).apply(outlier2).reset_index()

idf_six['iqr']=idf_six['idf_q3']-idf_six['idf_q1']
idf_six['lp']=idf_six['idf_q1']-1.5*(idf_six['idf_q3']-idf_six['idf_q1'])
idf_six['up']=idf_six['idf_q3']+1.5*(idf_six['idf_q3']-idf_six['idf_q1'])


bizpl_idf01['del']=np.where(bizpl_idf01['idf']>bizpl_idf01['up'],1,0)
bizpl_idf01=bizpl_idf01[bizpl_idf01['del']==0]
bizpl_idf01=pd.merge(left=bizpl_idf,right=idf_six[['bizpl_cl_div_cd',group,'up']],on=['bizpl_cl_div_cd',group],how='left')

bizpl_tfidf=pd.merge(bizpl_tf, bizpl_idf01[['bizpl_cl_div_cd','goods_cd','idf']] ,on =['goods_cd','bizpl_cl_div_cd'],how='inner')
bizpl_tfidf=pd.merge(left=bizpl_tfidf,right=good_dm[['line_cd','line_nm','class_cd','class_nm','goods_cd','goods_nm']],on=['goods_cd'],how='left')
bizpl_tfidf['tfidf']=bizpl_tfidf['tf']*bizpl_tfidf['idf']

#온라인 상품 제외
bizpl_tfidf_01=bizpl_tfidf[~bizpl_tfidf['class_nm'].str.contains('온라인')]  
