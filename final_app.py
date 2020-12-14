# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:33:59 2020

@author: Winston Fernandes
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle5 as pkl
import warnings
import gc
import os
import seaborn as sns
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import zipfile 



#@st.cache(suppress_st_warning=True)
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    filenames.sort()
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

#if st.checkbox('Select a file in current directory'):
 #   folder_path = '.'
  #  if st.checkbox('Change directory'):
#folder_path = st.text_input('Enter folder path', './data')
   # st.write('You selected `%s`' % filename)





#@st.cache(suppress_st_warning=True)
def one_hot_encoding_dataframe(df):
    '''
    one hot encoding 
    '''
    original_columns = list(df.columns)
    cat_columns=[x for x in df.columns if df[x].dtype == 'object']
    df=pd.get_dummies(df,columns=cat_columns,dummy_na= False)
    new_added_columns=list(set(df.columns).difference(set(original_columns)))
    return df,new_added_columns


#@st.cache(suppress_st_warning=True)
def feature_engineering_on_app_train_test(test_point):
    
    '''
    final feature engineering applied on app_train and app_test
    
    '''
#     app_train=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
#     app_test=pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
#     #we merge the two data frames the preprocessing we on train must also be done on test
#     df=app_train.append(app_test).reset_index()
    
#     del app_train
#     del app_test
    
#     df=df[df['CODE_GENDER']!='XNA'] 

    df=test_point
    
#     print(df.head())
    
    df["CODE_GENDER"].replace({'XNA': np.nan}, inplace = True)
    
    df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    
    df,new_cat_columns=one_hot_encoding_dataframe(df)
    
    with open("all_columns_app_train_test.pkl", "rb") as f:
        all_columns = pkl.load(f)
    
#     print(df.head())
    
    
#     with open("all_columns_app_train_test.pkl", "wb") as f:
#         pkl.dump(all_columns, f)
    
#     df=df[df['SK_ID_CURR']==id_value]

#     #The XNA value doesn't mean any thing so it is removed from train data
#     df=df[df['CODE_GENDER']!='XNA'] 
        
#     #we remove this because 365243 is an outlier


#     #There is an outlier in the train data where AMT_INCOME_TOTAL of a person having highest income had difficulty in paying loan. 
#     df=df[df['AMT_INCOME_TOTAL']<(0.2*1e8)]
 
#     print(list(df.columns))
    
    df['DOCUMNNET_COUNT']=(df[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']]==1).sum(axis=1)
    
    
    df['AMT_REQ_CREDIT_BUREAU_HDWMQY']=(df[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']]).sum(axis=1)
    
    
    #Using domain knowledge and Pairplot
    
#     df['AMT_CREDIT_FLAG']=( df['AMT_CREDIT']<=300000).astype(int)
    
#     df['DAYS_BIRTH_FLAG']=(abs(df['DAYS_BIRTH']//365)<=19).astype(int)
    
    #pecentage of his life spent working
    df['DAYS_WORKING_PER']=df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    df['DAYS_UNEMPLOYED']=abs(df['DAYS_BIRTH'])-abs(df['DAYS_EMPLOYED'])
    
    df['GOODS_PRICE_INCOME_TOTAL_PER']=df['AMT_INCOME_TOTAL']/df['AMT_GOODS_PRICE']
    
    df['GOODS_PRICE_CREDIT_PER']=df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
    
    df['GOODS_PRICE_AMT_ANNUITY_PER']=df['AMT_ANNUITY']/df['AMT_GOODS_PRICE']
    
#     df['GOODS_PRICE_EMP']=abs(df['DAYS_EMPLOYED'])/df['AMT_GOODS_PRICE']
    
#     df['AMT_CREDIT_BIRTH']=df['AMT_CREDIT']/abs(df['DAYS_BIRTH']/365)
    
    #percentage income of person and the credit amount
    df['INCOME_CREDIT_PER'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    #percentage income of person and the credit amount
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS']+1)

    #Amount paid for previous loan application every month decided by the number of day employed
    df['ANNUITY_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED']/ df['AMT_ANNUITY']
    
    df['AMT_CREDIT_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED']/ df['AMT_CREDIT']
    
    #Amount paid for  loan application every month decided by the number of day lived
    df['ANNUITY_DAYS_BIRTH_PERC'] = df['DAYS_BIRTH']/ df['AMT_ANNUITY']

    #Anually paid amount to amount credited
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    df['PAYMENT_RATE_INV'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    df['PAY_TOWARDS_LOAN'] = df['AMT_INCOME_TOTAL']-df['AMT_ANNUITY']

    # df['AMT_INCOME_TOTAL_FLAG_LOAN_LESS_50'] =(df['AMT_ANNUITY']<=(0.50*df['AMT_INCOME_TOTAL'])).astype(int)
    
    df['MEAN_DEFAULT_SURR']=((df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']]).sum(axis=1))//4
    
    df['ADDRESS_MISSMATCH']=((df[['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY']]).sum(axis=1))
    
    df['MEAN_ENQUIRIES']=((df[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']]).mean(axis=1))
    
    df['CONTACT_REF']=((df[['FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']]).sum(axis=1))
    
    df['MAX_DAYS_SOMETHING_CHANGED']=((df[['DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
       'DAYS_REGISTRATION']]).max(axis=1))
    
    #Creating features from useful features
    df['EXT_SOURCE_MEAN']=(df[['EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3']]).mean(axis=1)
    
    df['EXT_SOURCE_MEDIAN']=(df[['EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3']]).median(axis=1)
    
    df['EXT_SOURCE_MIN']=(df[['EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3']]).min(axis=1)
    
    df['EXT_SOURCE_MAX']=(df[['EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3']]).max(axis=1)
    
    
    add_columns=list(set(all_columns)-set(df.columns))
    for col in add_columns:
        df[col]=0.0
    
    return df

#@st.cache(suppress_st_warning=True)
def missing_values():
    with open("missing_3_list_7.pkl", "rb") as f:
        missing_3_list = pkl.load(f)
    with open("missing_2_list (1).pkl", "rb") as f:
        missing_2_list = pkl.load(f)
    with open("missing_1_list_7.pkl", "rb") as f:
        missing_1_list = pkl.load(f)
    with open("ridge_clf_list (3).pkl", "rb") as f:
        ridge_clf_list = pkl.load(f)
    with open("ridge_train_column_list (1).pkl", "rb") as f:
        ridge_train_column_list = pkl.load(f)
        
    imputer1 = pkl.load(open('imputer1_7.sav', 'rb'))
    
    present_list = pkl.load(open('median_present_list (1).pkl', 'rb'))
    
    return missing_3_list,missing_2_list,missing_1_list,ridge_clf_list,imputer1,present_list,ridge_train_column_list


def fill_the_missing_values(df):
    
    missing_3_list,missing_2_list,missing_1_list,ridge_clf_list,imputer1,present_list,ridge_train_column_list=missing_values()
      
    df.replace([np.inf, -np.inf], np.nan,inplace=True)    
    
    df=df.drop(columns=missing_3_list).copy()    

    mean_imp_test_data  = imputer1.transform(df[missing_1_list])
    
    df.loc[:,missing_1_list]=mean_imp_test_data.copy()

    temp_list=[]
    for i in range(0,len(missing_2_list)):
        if df[missing_2_list[i]].isnull().bool():
            df["temp_"+str(missing_2_list[i])]=present_list[i]
        else:
            df["temp_"+str(missing_2_list[i])]=df[missing_2_list[i]]   
        temp_list.append("temp_"+missing_2_list[i]) 
    
    for i in range(0,len(missing_2_list)):
        train_columns=ridge_train_column_list[i]
        if df[missing_2_list[i]].isnull().bool():
            df[missing_2_list[i]] = ridge_clf_list[i].predict(df[train_columns]) 
    df.drop(columns=temp_list,inplace=True)
    return df
    
#@st.cache(suppress_st_warning=True)
def calculate_cibil_score(df):
#cibil_train=train_data[['3365_LATE_PAYMENT_FLAG_MEAN','CRED_FLAG_LESS_30_MEAN','ABS_YEAR_CREDIT_MAX','UNSEC_LOAN_COUNT_SUM','SEC_LOAN_COUNT_SUM','AMT_REQ_CREDIT_BUREAU_WEEK']].copy()
    cibil_test=df[['3365_LATE_PAYMENT_FLAG_MEAN','CRED_FLAG_LESS_30_MEAN','ABS_YEAR_CREDIT_MAX','UNSEC_LOAN_COUNT_SUM','SEC_LOAN_COUNT_SUM','AMT_REQ_CREDIT_BUREAU_WEEK']].copy()    
    scaler_cibil = pkl.load(open('scaler_cibil_7.sav', 'rb'))
      
    cibil_test_std = scaler_cibil.transform(cibil_test)

    
    cibil_test = pd.DataFrame(data = cibil_test_std,  
                      columns = ['3365_LATE_PAYMENT_FLAG_MEAN','CRED_FLAG_LESS_30_MEAN','ABS_YEAR_CREDIT_MAX','UNSEC_LOAN_COUNT_SUM','SEC_LOAN_COUNT_SUM','AMT_REQ_CREDIT_BUREAU_WEEK']) 


    num_test=(0.1*cibil_test['UNSEC_LOAN_COUNT_SUM'].copy()+0.1*cibil_test['SEC_LOAN_COUNT_SUM'].copy()+0.05*cibil_test['ABS_YEAR_CREDIT_MAX'].copy()+0.25*cibil_test['CRED_FLAG_LESS_30_MEAN'].copy())
    den_test=(0.30*cibil_test['3365_LATE_PAYMENT_FLAG_MEAN'].copy()+0.20*cibil_test['AMT_REQ_CREDIT_BUREAU_WEEK'].copy())+1


    df.loc[:,'CIBIL_SCORE']=(num_test.copy()/den_test.copy())

    df.loc[:,'CIBIL_SCORE']=df['CIBIL_SCORE'].fillna(0)

    return df

@st.cache(suppress_st_warning=True)
def load_past():
#    with open("df_test.pkl", "rb") as f:
#        df_past = pkl.load(f)
    zf = zipfile.ZipFile('df_test.zip', 'r')
    df_past = pkl.load(zf.open('df_test.pkl'))
    df_past['SK_ID_CURR']=df_past.index
    return df_past

@st.cache(suppress_st_warning=True)
def get_past_data(id):
    past=load_past()
    test_point_past_data=past[past['SK_ID_CURR']==id]
    test_point_past_data.drop('SK_ID_CURR',axis=1,inplace=True)
    return test_point_past_data


def main1(test_point):
    df=feature_engineering_on_app_train_test(test_point)
    df_past = get_past_data(int(test_point['SK_ID_CURR'].values))  
    df=df.join(df_past,how='left', on='SK_ID_CURR')
    del df_past
    with open("lgbm_clf_list_7.pkl", "rb") as f:
        lgbm_list = pkl.load(f)   
    with open("train_column7.pkl", "rb") as f:
        train_column = pkl.load(f)
    gc.collect()
    df=fill_the_missing_values(df)
    df=calculate_cibil_score(df)
    scaler = pkl.load(open('scaler_7.sav', 'rb'))
    X=df[train_column]
    X=scaler.transform(X)
    test_pred_proba=0
    for j in range(0,len(lgbm_list)):
        test_pred_proba+=lgbm_list[j].predict_proba(X,num_iteration=lgbm_list[j].best_iteration_)[:,1]/10
    st.write('Will client be a Defaulter ? (Yes=1/No=0): ',str(int(test_pred_proba>0.5)))
    st.write('Probablility of being a Defaulter: ',str(test_pred_proba))

#user_input = st.text_input("Enter the SK_ID_CURR",100001)

@st.cache(suppress_st_warning=True)
def return_head(filename):
    df=pd.read_csv(filename)
    return df.head(5).reset_index()




st.image('logo.svg')
st.title('Loan Prediction')
st.markdown('## Business Problem')
st.markdown('* Home Credit is a international non-bank financial institution.')
st.markdown('* The problem is of risk modelling.')
st.markdown('* Given the data of a client we have to predict if he/she is able to repay loan or  will have difficulty in paying back.')


filename = file_selector(folder_path='./data')
error_flag=0
df=pd.read_csv(filename)


diff=set(df['NAME_CONTRACT_TYPE'].values)-{'Cash loans', 'Revolving loans'}
if len(diff)>0:
    st.error('Contract Type must have value Cash loans or Revolving loans found '+str(diff))
    error_flag=1
diff=set(df['WEEKDAY_APPR_PROCESS_START'].values)-{'FRIDAY', 'MONDAY', 'SATURDAY', 'SUNDAY', 'THURSDAY', 'TUESDAY', 'WEDNESDAY'}
if len(diff)>0:
    st.error('WEEKDAY_APPR_PROCESS_START must have value SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY found '+str(diff))
    error_flag=1
if (max(df['EXT_SOURCE_3'])>1) or (max(df['EXT_SOURCE_1'])>1) or (max(df['EXT_SOURCE_2'])>1):
    st.error('External source 1,2,3 must not have values greater than 1')
    error_flag=1
if (min(df['CNT_CHILDREN'])<0):
    st.error('Count of childrens of the client should not be less than zero')
    error_flag=1
diff=set(df['FLAG_OWN_CAR'].values)-{'Y', 'N'}
if len(diff)>0:
    st.error('FLAG_OWN_CAR must have value Y or N found '+str(diff))
    error_flag=1
diff=set(df['FLAG_OWN_REALTY'].values)-{'Y', 'N'}
if len(diff)>0:
    st.error('FLAG_OWN_REALTY must have value Y or N found '+str(diff))
    error_flag=1
if (max(df['HOUR_APPR_PROCESS_START'])>24):
    st.error('The hour that the approval process started should not be greater than 24 hours') 
    error_flag=1
    

@st.cache(suppress_st_warning=True)      
def top5_data():
    with open("train_data_top_5.pkl", "rb") as f:
        top5_data = pkl.load(f)
    top5_data['DAYS_BIRTH']=-1*top5_data['DAYS_BIRTH']
    return top5_data
 
def display_top_5(top5_data):
    

    st.markdown('## Top 5 features that helped in prediction ')
    st.markdown('### EXT_SOURCE_1: ')
    st.markdown('Normalized score from external data source.')
    fig,axes=plt.subplots()
    axes.set_xlabel('EXT_SOURCE_1')
    axes.set_ylabel('Density')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==0,'EXT_SOURCE_1'],label='Will Repay')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==1,'EXT_SOURCE_1'],label='Will Default')
    st.pyplot(fig)
    
    st.markdown('#### Analysis')
    st.markdown('* External source 1 < 0.5 indicate high probability that client will not repay loan.')
    st.markdown('* External source 1 > 0.5 indicate high probability that client will default loan.')
    st.markdown('* There is a visible sepration between two classes.')
    
    st.markdown('#### Conclusion')
    st.markdown('* External source 1 is a useful feature.')

    
    
    st.markdown('### PAYMENT_RATE: ')
    st.markdown('Payment rate is ratio of Amount Annuity and Amount Credit.')
    st.markdown('Amount Annuity is the amount paid back in periodic intervals.')
    st.markdown('Amount Credit is the amount recieved by the institution.')
    fig,axes=plt.subplots()
    axes.set_xlabel('PAYMENT_RATE')
    axes.set_ylabel('Density')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==0,'PAYMENT_RATE'],label='Will Repay')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==1,'PAYMENT_RATE'],label='Will Default')
    st.pyplot(fig)
    st.markdown('#### Analysis')
    st.markdown('* Client having payment rate < 0.06 have higher chance of repaying loan ')
    st.markdown('* Client having payment rate > 0.06 and payment rate < 0.09  have higher chance of defaulting loan.')
    st.markdown('* Client having payment rate between 0.09 and 0.11 have higher chance of repaying loan.')
    st.markdown('* There is a visible sepration between two classes.')
    st.markdown('#### Conclusion')
    st.markdown('* Payment Rate is a useful feature.')

    
   
    st.markdown('### DAYS_BIRTH: ')
    st.markdown("Client's age in days at the time of application.")
    fig,axes=plt.subplots()
    axes.set_xlabel('DAYS_BIRTH')
    axes.set_ylabel('Density')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==0,'DAYS_BIRTH'],label='Will Repay')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==1,'DAYS_BIRTH'],label='Will Default')
    st.pyplot(fig)
    st.markdown('#### Analysis')
    st.markdown('* Client having age (in days) less than 15000 have high probability of being default. ')
    st.markdown('* Client having age (in days) greater than 15000 have high probability of repaying loan.')
    st.markdown('* There is a visible sepration between two classes.')
    st.markdown('#### Conclusion')
    st.markdown('Younger clients are more likely to default as compared to older.')
    
    st.markdown('### EXT_SOURCE_3: ')
    st.markdown('Normalized score from external data source.')
    fig,axes=plt.subplots()
    axes.set_xlabel('EXT_SOURCE_3')
    axes.set_ylabel('Density')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==0,'EXT_SOURCE_3'],label='Will Repay')
    sns.kdeplot(top5_data.loc[top5_data['TARGET']==1,'EXT_SOURCE_3'],label='Will Default')
    st.pyplot(fig)
    st.markdown('#### Analysis')
    st.markdown('* External source 3 < 0.4 indicate high probability that client will default loan.')
    st.markdown('* (External source 3 > 0.5 and External source 3 < 0.9) indicate high probability that client will repay loan.')
    st.markdown('* There is a visible sepration between two classes.')
    
    st.markdown('#### Conclusion')
    st.markdown('* External source 3 is a useful feature.')
    
    st.markdown('### AMT_ANNUITY: ')
    st.markdown('Annuities are basically loans that are paid back over a set period of time at a set interest rate with consistent payments each period.')
    fig,axes=plt.subplots()
    axes.set_xlabel('AMT_ANNUITY')
    axes.set_ylabel('Density')
    sns.kdeplot(top5_data.loc[(top5_data['TARGET'] == 0) & (top5_data['AMT_ANNUITY']<(100000)), 'AMT_ANNUITY'],label='Will Repay')
    sns.kdeplot(top5_data.loc[(top5_data['TARGET']==1) & (top5_data['AMT_ANNUITY']<(100000)),'AMT_ANNUITY'],label='Will Default')
    st.pyplot(fig)
    st.markdown('#### Analysis')
    st.markdown('* Amount less than 10000 there is more chance that client will repay.')
    st.markdown('* Amount between 20000 to 40000 shows a slight high probability that client will default loan.')
    st.markdown ('* Amount greater than 40000 but less than 80000 shows a slight high probability that loan will be repayed.')
    st.markdown('* There is a visible sepration between two classes.')
    
    st.markdown('#### Conclusion')
    st.markdown('* Amount Annuity is a useful feature.')

    

def add_reference():
    st.markdown('## Reference')
    st.markdown('* https://www.kaggle.com/c/home-credit-default-risk')
    st.markdown('* https://www.streamlit.io/')
    st.markdown('* https://www.appliedroots.com/')
    
def contact():
    st.markdown('## About me')
    st.markdown('* Linkedin: https://www.linkedin.com/in/winston-fernandes-a14a89145/')
    st.markdown('* Email 📧: winston23fernandes.wf@gmail.com')
    st.markdown('* Contact 📱: +91-7507050685')         
    st.markdown('* If you found this project informative hit the ⭐ on my github repo https://github.com/wins999/Home_Credit_Loan_Prediction')
    

if error_flag==0:
    df_head=return_head(filename)
    st.write("Test Client's data")
    st.dataframe(df_head)
    option = st.selectbox("Select the SK_ID_CURR",(df_head['SK_ID_CURR'].values))
    test_point=df_head[df_head['SK_ID_CURR']==int(option)]
    main1(test_point)
    top5_data=top5_data()
    display_top_5(top5_data)
    add_reference()
    contact()
    

    

   