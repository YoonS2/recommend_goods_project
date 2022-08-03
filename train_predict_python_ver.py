# --db2연결
import ibm_db_dbi
# conn = ibm_db_dbi.connect("DRIVER={IBM DB2 ODBC DRIVER}; Database=PGSDW; Hostname=10.56.36.47; Port=50001; PROTOCOL=TCPIP; UID=gsseldw; PWD=gsseldw01", "", "") 
conn = ibm_db_dbi.connect("DRIVER={IBM DB2 ODBC DRIVER}; Database=PGSDW; Hostname=10.56.36.47; Port=50001; PROTOCOL=TCPIP; UID=gsanldw; PWD=gsanldw01", "", "") 

#train set-- 유사점 구한 버전으로 미리 준비하기.. 
# query = '''
#     SELECT *
#     FROM GSANLDW.danbi_TRAIN_line01_20210401TM
#     ;
#     '''

# query = '''
#     SELECT *
#     FROM GSANLDW.danbi_TRAIN_GROUP22_20210401
#     ;
#     '''
# query = '''
# SELECT *
# from GSANLDW.DANBI_TRAIN_LINE39_tm2
# ;
# '''
query = '''
SELECT *
from GSANLDW.DANBI_TRAIN_SET_TM1018
;
'''

# query = '''
# SELECT *
# from GSANLDW.danbi_TRAIN_line01_OUT2
# ;
# '''
train_origin = pd.read_sql( query, conn ) 


#test set
# query = '''
#     SELECT *
#     FROM GSANLDW.danbi_TEST_line01_20210416TM
#     ;
#     '''
# query = '''
#     SELECT *
#     FROM GSANLDW.danbi_TRAIN_GROUP22_20210416
#     ;
#     '''
# query = '''
# SELECT *
# from GSANLDW.DANBI_TEST_line39
# ;
# '''

query = '''
SELECT *
from GSANLDW.DANBI_TEST_SET_TM1018
;
'''

# query = '''
# SELECT *
# from GSANLDW.danbi_TEST_line01_OUT
# ;
# '''


test = pd.read_sql( query, conn ) 


#test set
# query = '''
#     SELECT *
#     FROM GSANLDW.danbi_TEST_line01_20210416TM
#     ;
#     '''
query = '''
    SELECT *
    FROM GSANLDW.danbi_event_tm2
    ;
    '''

# query = '''
# SELECT *
# from GSANLDW.danbi_TEST_line01_OUT
# ;
# '''


event = pd.read_sql( query, conn ) 

#기존
query = '''
SELECT *
from GSANLDW.danbi_logistic2
;
'''


logistic = pd.read_sql( query, conn ) 



train=pd.merge(train_origin[['BF_BIZ','CLASS_NM','GOODS_CD','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA']],event[event['YYYYMM']=='202104'],left_on='GOODS_CD',right_on='PRSNT_GOODS_CD',how='left')
train=train[['BF_BIZ','CLASS_NM','GOODS_CD','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA','E21','E11','DIST','EVENT_DAY']]
train=pd.merge(train,event.loc[event['YYYYMM']=='202104',['PRSNT_GOODS_CD','E21','E11','DIST']],left_on='GOODS_CD',right_on='PRSNT_GOODS_CD',how='left')

train['E21']=train['E21_x'].fillna(0)
train['E11']=train['E11_x'].fillna(0)
train['DIST']=train['DIST_x'].fillna(0)
train['E21_af']=train['E21_y'].fillna(0)
train['E11_af']=train['E11_y'].fillna(0)
train['DIST_af']=train['DIST_y'].fillna(0)
train['EVENT_DAY']=train['EVENT_DAY'].fillna(0)
# train.loc[train['E21']>0 ,'E21']==1
# train.loc[train['E11']>0 ,'E11']==1
# train.loc[train['DIST']>0 ,'DIST']==1
train['E21'] = train["E21"].apply(lambda x: 1 if x >0 else x)
train['E11'] = train["E11"].apply(lambda x: 1 if x >0 else x)
train['DIST'] = train["DIST"].apply(lambda x: 1 if x >0 else x)
train['E21_af'] = train["E21_af"].apply(lambda x: 1 if x >0 else x)
train['E11_af'] = train["E11_af"].apply(lambda x: 1 if x >0 else x)
train['DIST_af'] = train["DIST_af"].apply(lambda x: 1 if x >0 else x)
train=train.drop(['EVENT_DAY','E11_x','E21_x','DIST_x','E11_y','E21_y','DIST_y','GOODS_CD','PRSNT_GOODS_CD'],axis=1)



test2=pd.merge(test[['BF_BIZ','CLASS_NM','GOODS_CD','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA']],event[event['YYYYMM']=='202104'],left_on='GOODS_CD',right_on='PRSNT_GOODS_CD',how='left')
test2=test2[['BF_BIZ','CLASS_NM','GOODS_CD','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA','E21','E11','DIST','EVENT_DAY']]
test2=pd.merge(test2,event.loc[event['YYYYMM']=='202105',['PRSNT_GOODS_CD','E21','E11','DIST']],left_on='GOODS_CD',right_on='PRSNT_GOODS_CD',how='left')

test2['E21']=test2['E21_x'].fillna(0)
test2['E11']=test2['E11_x'].fillna(0)
test2['DIST']=test2['DIST_x'].fillna(0)
test2['E21_af']=test2['E21_y'].fillna(0)
test2['E11_af']=test2['E11_y'].fillna(0)
test2['DIST_af']=test2['DIST_y'].fillna(0)
test2['EVENT_DAY']=test2['EVENT_DAY'].fillna(0)
# train.loc[train['E21']>0 ,'E21']==1
# train.loc[train['E11']>0 ,'E11']==1
# train.loc[train['DIST']>0 ,'DIST']==1
test2['E21'] = test2["E21"].apply(lambda x: 1 if x >0 else x)
test2['E11'] = test2["E11"].apply(lambda x: 1 if x >0 else x)
test2['DIST'] = test2["DIST"].apply(lambda x: 1 if x >0 else x)
test2['E21_af'] = test2["E21_af"].apply(lambda x: 1 if x >0 else x)
test2['E11_af'] = test2["E11_af"].apply(lambda x: 1 if x >0 else x)
test2['DIST_af'] = test2["DIST_af"].apply(lambda x: 1 if x >0 else x)
test2=test2.drop(['EVENT_DAY','E11_x','E21_x','DIST_x','E11_y','E21_y','DIST_y','GOODS_CD','PRSNT_GOODS_CD'],axis=1)


# train=train_origin[['BF_BIZ','CLASS_NM','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA']]
train=train_origin[['BF_BIZ','CLASS_NM','GOODS_NM','REAL_TFIDF','REAL_QTY','PRE_TFIDF','PRE_QTY','TA','TA2','TA3','TA4']]

train2=train.copy()
# train2['PRE_TFIDF']=train['REAL_TFIDF']
# train2['PRE_QTY']=train['REAL_QTY']


# test2=test[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_TFIDF','PRE_QTY','TA']]
test2=test[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_TFIDF','PRE_QTY','TA','TA2','TA3','TA4']]


# real로 학습할시 결측값 제거
train2=train2.dropna()



start = time.time()  # 시작 시간 저장
biz=list(set(train2['BF_BIZ']))
droplist=[]
for i in biz :
    orilen=len(train2.loc[train2['BF_BIZ']==i,:])
    ta1len=len(train2.loc[(train2['BF_BIZ']==i) & (train2['TA']==1),:])
    ta0len=len(train2.loc[(train2['BF_BIZ']==i) & ( train2['TA']==0),:])
    if (orilen==ta1len) | (orilen==ta0len) :
        droplist.append(i)
biz2=list(set(test2['BF_BIZ']))
droplist2=[]
for i in biz2 :
    orilen=len(test2.loc[test2['BF_BIZ']==i,:])
    ta1len=len(test2.loc[(test2['BF_BIZ']==i) & (test2['TA']==1),:])
    ta0len=len(test2.loc[(test2['BF_BIZ']==i) & ( test2['TA']==0),:])
    if (orilen==ta1len) | (orilen==ta0len) :
        droplist2.append(i)
print("time :", (time.time() - start)/60)  # 현재시각 - 시작시간 = 실행 시간


train_train2.groupby('BF_BIZ').sum()



for i in droplist :
    biz.remove(i)
for i in droplist2 :
    biz2.remove(i)
    
    
    train2=train2[train2['BF_BIZ'].isin(biz)]
test2=test2[test2['BF_BIZ'].isin(biz2)]
origin= pd.merge(train2, test2, left_on='BF_BIZ', right_on='BF_BIZ', how='inner')
# origin= pd.merge(train_real, test2, left_on='BF_BIZ', right_on='BF_BIZ', how='inner')
origin=list(set(origin['BF_BIZ']))



test3=test2[test2['BF_BIZ'].isin(origin)]
test3=test3.reset_index()
test3=test3.drop(['index'],axis=1)
train3=train2[train2['BF_BIZ'].isin(origin)]
train3=train3.reset_index()
train3=train3.drop(['index'],axis=1)


#정확도 비교 

train3['seg']='train'
test3['seg']='test'
alldata=pd.concat([train3,test3],axis=0)

originby1=alldata.groupby('BF_BIZ')
origingroup1=[group for name, group in originby1]
originname=[name for name, group in originby1]
x=dict(list(originby1))

# col1=['se','PRE_TFIDF','PRE_QTY']
col1=['PRE_QTY']



#랜덤포레스트
origin_name=originby1.nunique().index
rftotal_result=pd.DataFrame(columns=['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','rf_pred','rf_prob1'])

start = time.time()  # 시작 시간 저장
for i in range(len(origingroup1) ):


    #랜덤포레스트
    clf = RandomForestClassifier(random_state=123,max_features=1,n_estimators=30,max_depth=2)
    train_x=origingroup1[i].loc[origingroup1[i]['seg']=='train',:]
    test_x=origingroup1[i].loc[origingroup1[i]['seg']=='test',:]
    train_y=origingroup1[i].loc[origingroup1[i]['seg']=='train','TA']
    test_y=origingroup1[i].loc[origingroup1[i]['seg']=='test','TA']
#     le = LabelEncoder()
#     train_x['class_le']=le.fit_transform(train_x['CLASS_NM'])
#     test_x['class_le']=le.transform(test_x['CLASS_NM'])
#     lr = LogisticRegression(random_state=123) 
    scaler =StandardScaler()
#     sc_tr_feat=scaler.fit_transform(pd.DataFrame(data=train_x,columns=col1))
#     sc_te_feat=scaler.transform(pd.DataFrame(data=test_x ,columns=col1))
#     train_x, test_x, train_y, test_y = train_test_split(origingroup1[i], origingroup1[i].loc[:,'TA'], test_size = 0.3, random_state = 30)
    clf.fit(train_x.loc[:,col1],train_y) #학습
#     clf.fit(sc_tr_feat,train_y)
#     predict=pd.DataFrame(clf.predict(sc_te_feat),columns=['rf_pred'])
#     prob=pd.DataFrame(clf.predict_proba(sc_te_feat),columns=['rf_prob0','rf_prob1'])  # 분류확률
    
    predict = pd.DataFrame(clf.predict(test_x.loc[:,col1]),columns=['rf_pred']) # 예측
    prob=pd.DataFrame(clf.predict_proba(test_x.loc[:,col1]),columns=['rf_prob0','rf_prob1']) # 분류확률
    origin_nm=test_x.reset_index()[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','TA']]
      #결과합
    rf_result= pd.concat([origin_nm, predict,prob['rf_prob1']],axis=1)
    rftotal_result=pd.concat([rftotal_result,rf_result],axis=0)

print("time :", (time.time() - start)/60)  # 현재시각 - 시작시간 = 실행 시간


rftotal_result=pd.DataFrame(columns=['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','E21','E11','DIST','E21_af','E11_af','DIST_af','rf_pred','rf_prob1'])

# col1=['se','PRE_TFIDF','PRE_QTY']
col1=['PRE_QTY','E21','E11','DIST','E21_af','E11_af','DIST_af']

start = time.time()  # 시작 시간 저장
for i in range(len(origingroup1) ):


    #랜덤포레스트
    clf = RandomForestClassifier(random_state=123,max_features=1,n_estimators=30,max_depth=2)
    train_x=origingroup1[i].loc[origingroup1[i]['seg']=='train',:]
    test_x=origingroup1[i].loc[origingroup1[i]['seg']=='test',:]
    train_y=origingroup1[i].loc[origingroup1[i]['seg']=='train','TA']
    test_y=origingroup1[i].loc[origingroup1[i]['seg']=='test','TA']
#     le = LabelEncoder()
#     train_x['class_le']=le.fit_transform(train_x['CLASS_NM'])
#     test_x['class_le']=le.transform(test_x['CLASS_NM'])
#     lr = LogisticRegression(random_state=123) 
    scaler =StandardScaler()
#     sc_tr_feat=scaler.fit_transform(pd.DataFrame(data=train_x,columns=col1))
#     sc_te_feat=scaler.transform(pd.DataFrame(data=test_x ,columns=col1))
#     train_x, test_x, train_y, test_y = train_test_split(origingroup1[i], origingroup1[i].loc[:,'TA'], test_size = 0.3, random_state = 30)
    clf.fit(train_x.loc[:,col1],train_y) #학습
#     clf.fit(sc_tr_feat,train_y)
#     predict=pd.DataFrame(clf.predict(sc_te_feat),columns=['rf_pred'])
#     prob=pd.DataFrame(clf.predict_proba(sc_te_feat),columns=['rf_prob0','rf_prob1'])  # 분류확률
    
    predict = pd.DataFrame(clf.predict(test_x.loc[:,col1]),columns=['rf_pred']) # 예측
    prob=pd.DataFrame(clf.predict_proba(test_x.loc[:,col1]),columns=['rf_prob0','rf_prob1']) # 분류확률
    origin_nm=test_x.reset_index()[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','E21','E11','DIST','E21_af','E11_af','DIST_af','TA']]
      #결과합
    rf_result= pd.concat([origin_nm, predict,prob['rf_prob1']],axis=1)
    rftotal_result=pd.concat([rftotal_result,rf_result],axis=0)

print("time :", (time.time() - start)/60)  # 현재시각 - 시작시간 = 실행 시간

xx=rftotal_result.loc[rftotal_result['rf_pred']!=rftotal_result['TA'],:].reset_index()

xx.to_excel("D:/자료/업무/점포유형화/점포유형화고도화/팔림새유형/파일럿후로직수정/랜덤포레스트확률.xlsx")


## 로지스틱
lgtotal_result=pd.DataFrame(columns=['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','logi_pred','logi_prob1'])
lgtotal_result=pd.DataFrame(columns=['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','E21','E11','DIST','E21_af','E11_af','DIST_af','logi_pred','logi_prob1'])
col1=['PRE_QTY']
TA='TA3'

# col1=['se','PRE_TFIDF','PRE_QTY']
col1=['PRE_QTY','E21','E11','DIST','E21_af','E11_af','DIST_af']

from sklearn.preprocessing import MinMaxScaler
start = time.time()  # 시작 시간 저장
for i in range(len(origingroup1) ):
    #스케일링
    lr = LogisticRegression(random_state=123) 
    scaler =StandardScaler()
#     minmax=MinMaxScaler()
    
    train_x=origingroup1[i].loc[origingroup1[i]['seg']=='train',:]
    test_x=origingroup1[i].loc[origingroup1[i]['seg']=='test',:]
    train_y=origingroup1[i].loc[origingroup1[i]['seg']=='train',TA]
    test_y=origingroup1[i].loc[origingroup1[i]['seg']=='test',TA]
#     col1=['PRE_QTY','E21','E11','DIST','E21_af','E11_af','DIST_af']+list(pd.get_dummies(train_x['CLASS_NM']).columns)
#     le = LabelEncoder()

#     train_x=pd.concat([train_x,pd.get_dummies(train_x['CLASS_NM'])],axis=1)
#     test_x=pd.concat([test_x,pd.get_dummies(test_x['CLASS_NM'])],axis=1)
#     train_x, test_x, train_y, test_y = train_test_split(origingroup1[i], origingroup1[i].loc[:,'TA'], test_size = 0.3, random_state = 30)
 
    sc_tr_feat=scaler.fit_transform(pd.DataFrame(data=train_x,columns=col1))
    sc_te_feat=scaler.transform(pd.DataFrame(data=test_x ,columns=col1))
    
#     sc_tr_feat=minmax.fit_transform(pd.DataFrame(data=train_x,columns=col1))
#     sc_te_feat=minmax.fit_transform(pd.DataFrame(data=test_x ,columns=col1))
    model2=lr.fit(sc_tr_feat,train_y)
    lg_predict=pd.DataFrame(model2.predict(sc_te_feat),columns=['logi_pred'])
    lg_prob2=pd.DataFrame(model2.predict_proba(sc_te_feat),columns=['logi_prob0','logi_prob1'])  # 분류확률
    origin_nm=test_x.reset_index()[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF',TA]]
#     origin_nm=test_x.reset_index()[['BF_BIZ','CLASS_NM','GOODS_NM','PRE_QTY','PRE_TFIDF','E21','E11','DIST','E21_af','E11_af','DIST_af','TA']]
    
      #결과합
    logi_result= pd.concat([origin_nm, lg_predict,lg_prob2['logi_prob1']],axis=1)
    lgtotal_result=pd.concat([lgtotal_result,logi_result],axis=0)

print("time :", (time.time() - start)/60)  # 현재시각 - 시작시간 = 실행 시간

pd.set_option('display.max_row', 500) #데이터프레임 최대 출력 행수
pd.set_option('display.max_columns', 100)#데이터프레임 최대 출력 열수
lgtotal_result[lgtotal_result['BF_BIZ']=='GS25분당푸른점']

result_m=pd.merge(lgtotal_result, rftotal_result[['BF_BIZ','GOODS_NM','rf_pred','rf_prob1']], left_on=['BF_BIZ','GOODS_NM'], right_on=['BF_BIZ','GOODS_NM'], how='left')

xx=pd.merge(result_m,test[['BF_BIZ','GOODS_NM','REAL_QTY']], left_on=['BF_BIZ','GOODS_NM'], right_on=['BF_BIZ','GOODS_NM'], how='left')

#두 방법 결과 다른값
result_m.loc[result_m['logi_pred']!=result_m['rf_pred'],:].to_excel('D:/자료/업무/점포유형화/점포유형화고도화/팔림새유형/파일럿후로직수정/pre_qty_두로직차이결과.xlsx')

xx.to_excel('D:/자료/업무/점포유형화/점포유형화고도화/팔림새유형/파일럿후로직수정/temp.xlsx')


