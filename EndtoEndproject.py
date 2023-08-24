#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:38:53 2023

@author: wangshuyou
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid",font_scale=1.5)
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier


from sklearn.impute import SimpleImputer #補值
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

from imblearn.over_sampling import SMOTE

# import data
train_df = pd.read_csv('/Users/wangshuyou/Kaggle/spaceship-titanic/train.csv')
test_df = pd.read_csv('/Users/wangshuyou/Kaggle/spaceship-titanic/test.csv')

# Checking Dimensions of Data
print('Training dataset shape is:', train_df.shape)
print('Testing dataset shape is:', test_df.shape)

# Checking duplicates data
print(f'Duplicates in Train Dataset is: {train_df.duplicated().sum()}, ({100 * train_df.duplicated().sum()/len(train_df)})%')
print(f'Duplicates in Test Dataset is: {test_df.duplicated().sum()}, ({100 * test_df.duplicated().sum()/len(test_df)})%')

# Checking data type
print("Data Types of features of Training Data is:")
print(train_df.dtypes)
print('\n'+'-'*100)
print("Data Types of features of Testing Data is:")
print(test_df.dtypes)

# Percentage of missing values in dataset
def Missing_Counts( Data, NoMissing=True ) : 
    missing = Data.isnull().sum()  
    
    if NoMissing==False :
        missing = missing[ missing>0 ]
        
    missing.sort_values( ascending=False, inplace=True )  
    Missing_Count = pd.DataFrame( { 'Column Name':missing.index, 'Missing Count':missing.values } ) 
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['Missing Count'].apply( lambda x: '{:.2%}'.format(x/Data.shape[0] ))
    return  Missing_Count
print(Missing_Counts(train_df))
print(Missing_Counts(test_df))

# Checking cardinality of categorical features
# 確認有多少不同的物件->通常會刪除太多cardinality的特徵 and 新增新特徵
print("cardinality of categorical features in training datasets is:")
print(train_df.select_dtypes(include='object').nunique())
print('-'*70)
print("cardinality of categorical features in testing datasets is:")
print(test_df.select_dtypes(include='object').nunique())

#EDA
# transported分佈比例
plt.pie(train_df['Transported'].value_counts(),labels=train_df['Transported'].value_counts().keys(),autopct="%1.1f%%" ,textprops={"fontsize":20,"fontweight":"black"},colors=sns.color_palette("Set2")) # autopct顯示百分比
plt.title("Transported Feature Distribution")
plt.show()

# AGE 分佈特徵
sns.histplot(x = train_df['Age'],hue='Transported',data = train_df,kde = True,palette=('Set2'))
plt.show()

# expansion disturbution(histgram)
plt.figure(figsize=(14,10)) # 圖片長寬尺寸（英吋）
exp_cols = [ 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for index,column in enumerate(exp_cols):
    plt.subplot(3,2,index+1) # rows, columns, position
    sns.histplot(x = column, data = train_df, hue = 'Transported',bins =30,kde=True, palette=('Set2'))
    plt.title(f'{column} Disturbution')
    plt.ylim(0,100)
    plt.tight_layout() # 可考慮上下左右邊界
    
# cardinality disturbution(countplot)
cardi_cols = ['HomePlanet','CryoSleep','Destination','VIP']
plt.figure(figsize=(14,10)) # 圖片長寬尺寸（英吋）
for index, column in enumerate(cardi_cols):
    plt.subplot(4,1,index+1)
    sns.countplot(column, data = train_df, hue='Transported',palette='Set2')
    plt.title(f'{column} Disturbution')
    plt.tight_layout()

# Feature Engineering
# 1.PassengerID
def passengerid_new_features(df):
    # 細分為第幾組、第幾個成員
    df['Group'] = df['PassengerId'].apply(lambda x:x.split('_')[0])
    df['Member'] =df['PassengerId'].apply(lambda x:x.split('_')[1])
    # 家庭成員中順序的數量
    x = df.groupby('Group')['Member'].count().sort_values()
    y = set(x[x>1].index) #一人以上
    # 單身的族群
    df['Travelling_Solo'] = df['Group'].apply(lambda x:x not in y)
    df['Group_Size']=0
    for i in x.items():
        df.loc[df["Group"]==i[0],"Group_Size"]=i[1]
        
passengerid_new_features(train_df)
passengerid_new_features(test_df)
# 拿掉group、member的欄位
train_df.drop(columns = ['Group','Member'],inplace=True)
test_df.drop(columns = ['Group','Member'], inplace = True)
# visualizing
members_cols = ['Travelling_Solo','Group_Size']
plt.figure(figsize=(14,10))
for index,columns in enumerate(members_cols):
    plt.subplot(1, 2, index+1)
    sns.countplot(x = columns, data = train_df, hue = 'Transported', palette='Set2')
    plt.title(f'{columns} vs. Transported')
    plt.tight_layout()
# 2.Cabin 
def cabin_new_feature(df):
    # 先處理缺失值->補np.nan
    df['Cabin'].fillna('np.nan/np.nan/np.nan',inplace = True)
    df['cabin_Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['cabin_Number'] = df['Cabin'].apply(lambda x: x.split('/')[1])
    df['cabin_Side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
    # 把文字轉成缺失值格式
    cols = ['cabin_Deck','cabin_Number','cabin_Side']
    df[cols] = df[cols].replace('np.nan',np.nan)
    # 補值
    df['cabin_Deck'].fillna(df['cabin_Deck'].mode()[0],inplace = True)
    df['cabin_Number'].fillna(df['cabin_Number'].median(),inplace = True)
    df['cabin_Side'].fillna(df['cabin_Side'].mode()[0],inplace = True)
cabin_new_feature(train_df)
cabin_new_feature(test_df)
# visualzing
cabin_cols =['cabin_Deck','cabin_Side']
plt.figure(figsize=(14,10))
for index,column in enumerate(cabin_cols):
    plt.subplot(1,2,index+1)
    sns.countplot(x = column, data = train_df,hue = 'Transported',palette='Set2')
    plt.title(f'{column} vs. Transported')
    plt.tight_layout()

train_df['cabin_Number'] = train_df['cabin_Number'].astype(int)
test_df['cabin_Number'] = test_df['cabin_Number'].astype(int)
print("Total Unique values present in Cabin_Number feature is:",train_df["cabin_Number"].nunique())
print("The Mean of Cabin_Number Feature is: ",train_df["cabin_Number"].mean())
print("The Median of Cabin_Number Feature is:",train_df["cabin_Number"].median())
print("The Minimum value of Cabin_Number feature is:",train_df["cabin_Number"].min())
print("The Maximum value of Cabin_number Feature is:",train_df["cabin_Number"].max())

sns.histplot(x = train_df['cabin_Number'],data = train_df, hue = 'Transported', palette='Set2')
plt.xticks(list(range(0,1900,300)))
plt.vlines(300,ymin=0,ymax=550,color="black")
plt.vlines(600,ymin=0,ymax=550,color="black")
plt.vlines(900,ymin=0,ymax=550,color="black")
plt.vlines(1200,ymin=0,ymax=550,color="black")
plt.vlines(1500,ymin=0,ymax=550,color="black")
plt.show()

#cabin_number->cabin_region
def cabin_regions(df):
    df['Cabin_Region1'] = df['cabin_Number'] < 300
    df['Cabin_Region2'] = (df['cabin_Number'] >= 300)&(df['cabin_Number'] < 600)
    df['Cabin_Region3'] = (df['cabin_Number'] >= 600)&(df['cabin_Number'] < 900)
    df['Cabin_Region4'] = (df['cabin_Number'] >= 900)&(df['cabin_Number'] < 1200)
    df['Cabin_Region5'] = (df['cabin_Number'] >= 1200)&(df['cabin_Number'] < 1500)
    df['Cabin_Region6'] = (df['cabin_Number'] >= 1500)
cabin_regions(train_df)
cabin_regions(test_df)
train_df.drop(columns=['cabin_Number'],inplace = True)
test_df.drop(columns=['cabin_Number'],inplace = True)
# visualzing
regine_cols = ['Cabin_Region1','Cabin_Region2','Cabin_Region3','Cabin_Region4','Cabin_Region5','Cabin_Region6']
plt.figure(figsize=(14,10))
for index, column in enumerate(regine_cols):
    plt.subplot(3,2,index+1)
    sns.countplot(x = column, data = train_df, hue = 'Transported',palette='Set2')
    plt.title(f'{column} vs. Transported')
    plt.tight_layout()

# 3.Age
def age_group(df):
    age_group = []
    for i in df['Age']:
        if i <= 12:
            age_group.append('Age_0~12')
        elif (i > 12 and i <= 18):
            age_group.append('Age_13~18')
        elif (i > 18 and i <= 25):
            age_group.append('Age_19~25')       
        elif (i > 25 and i <= 32):
            age_group.append('Age_26~32')
        elif (i > 32 and i <= 50):
            age_group.append('Age_33~50')
        elif (i>50):
            age_group.append("age_50+")
        else:
            age_group.append(np.nan)
    df['Age_group'] = age_group
age_group(train_df)
age_group(test_df)
# visualzing

order = sorted(train_df['Age_group'].value_counts().keys().to_list())
#orders = ['Age_0~12','Age_13~18','Age_19~25','Age_26~32','Age_33~50',"age_50+"]
plt.figure(figsize=(14,10))
sns.countplot(x='Age_group', data = train_df, hue = 'Transported',palette='Set2',order = order)
plt.show()

# 4. 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
def new_exp_features(df):
    df['Total_Expanditure'] = df[exp_cols].sum(axis = 1)
    df['No Spending'] = (df['Total_Expanditure'] == 0)
new_exp_features(train_df)
new_exp_features(test_df)
# visualzing
plt.figure(figsize=(14,10))
sns.histplot(x = 'Total_Expanditure',data = train_df, hue = 'Transported', palette='Set2',bins = 200,kde = True)
plt.ylim(0,1000)
plt.xlim(0,10000)
plt.show()
# 使用 mean、median 來決定區分間距
mean = round(train_df['Total_Expanditure'].mean())
median = round(train_df['Total_Expanditure'].median())
print(mean)  # 1441
print(median) # 716
# 切四份
def expenditure_category(df):
    expense_category = []
    for i in df['Total_Expanditure']:
        if i == 0:
            expense_category.append('No Expense')
        elif i > 0 and i <= 716:
            expense_category.append('Low Expense')
        elif i > 716 and i <= 1441:
            expense_category.append('Median Expense')
        elif i > 1441:
            expense_category.append('High Expense')
    df['Expense_Category'] = expense_category
expenditure_category(train_df)
expenditure_category(test_df)
# visualzing:Expense_Category & No Spending
expanditure = ['Expense_Category','No Spending']
plt.figure(figsize=(18,8))
for index, column in enumerate(expanditure):
    plt.subplot(1,2,index+1)
    sns.countplot(x = column, data = train_df, hue = 'Transported', palette='Set2')
    plt.title(f'{column} vs. Transported')
    plt.tight_layout()

# Data prrprocessing
# Missing value
print(Missing_Counts(train_df))
# visualzing missing value
import missingno as msno # 缺失值視覺化
msno.bar(train_df, color='C1',fontsize=22)
plt.show()
# another way
sns.heatmap(train_df.isnull(),cmap = 'summer')
plt.show()

#補值
# 1. 欄位分類
cat_cols = train_df.select_dtypes(include = ['object','bool']).columns.to_list()
cat_cols.remove('Transported')
num_cols = train_df.select_dtypes(include = ['int','float64']).columns.to_list()

imputer1 = SimpleImputer(strategy='median')
imputer2 = SimpleImputer(strategy='most_frequent')

def fill_missingno(df):
    df[cat_cols] = imputer2.fit_transform(df[cat_cols])
    df[num_cols] = imputer1.fit_transform(df[num_cols])
fill_missingno(train_df)
fill_missingno(test_df)

# 確認重複值
print('Duplicate values in training data is: ',train_df.duplicated().sum())
print('Duplicate values in testing data is: ',test_df.duplicated().sum())
# Cardinality checking
print("cardinality of categorical features in training datasets is:")
print(train_df.select_dtypes(include='object').nunique())
print("cardinality of categorical features in testing datasets is:")
print(test_df.select_dtypes(include='object').nunique())
# 刪掉特定欄位
pass_df = test_df[['PassengerId']]
drop_cols = ['PassengerId','Cabin','Name']
train_df.drop(columns = drop_cols, inplace=True)
test_df.drop(columns = drop_cols, inplace=True)

# 右尾取log
log_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Total_Expanditure']
for i in log_cols:
    train_df[i] = np.log(1+train_df[i])
    test_df[i] = np.log(1+test_df[i])
# visualzing
plt.figure(figsize=(14,10))
for index,column in enumerate(log_cols):
    plt.subplot(3,2, index+1)
    sns.distplot(train_df[column],color='green')
    plt.ylim(0,0.5)
    plt.title(f'{column} disturbution')
    plt.tight_layout()
# objet 轉 bool
bool_cols = ['CryoSleep','VIP','Travelling_Solo','Cabin_Region1','Cabin_Region2','Cabin_Region3','Cabin_Region4','Cabin_Region5','Cabin_Region6','No Spending']
train_df[bool_cols] = train_df[bool_cols].astype('bool')
test_df[bool_cols] = test_df[bool_cols].astype('bool')

# encoding
nominal_cols = ['HomePlanet','Destination']
# label encoding
ordinal_cols = ["CryoSleep","VIP","Travelling_Solo","cabin_Deck","cabin_Side","Cabin_Region1","Cabin_Region2",
                    "Cabin_Region3","Cabin_Region4","Cabin_Region5","Cabin_Region6","Age_group","No Spending",
                    "Expense_Category"]
# label
label = LabelEncoder()
train_df[ordinal_cols] = train_df[ordinal_cols].apply(label.fit_transform)
test_df[ordinal_cols] = test_df[ordinal_cols].apply(label.fit_transform)
# one-hot
train_df = pd.get_dummies(train_df,columns=nominal_cols)
test_df = pd.get_dummies(test_df,columns=nominal_cols)

train_df['Transported'].replace({True:1,False:0},inplace=True)

X = train_df.drop(columns = 'Transported')
Y = train_df['Transported']

# Feature scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
test_df_scaled = sc.fit_transform(test_df)

# 比較scale前後模型成效
# no scale
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2, random_state=0)
# scaled
train_x1, test_x1, train_y1, test_y1 = train_test_split(X_scaled,Y,test_size=0.2,random_state=0)

# Model 
training_score = []
testing_score = []
# 1. scaled data :Logistic、SVM、KNN、 Naive-Bayes
def model_prediction(model):
    model.fit(train_x1, train_y1)
    x_train_pred1 = model.predict(train_x1)
    x_test_pred1 = model.predict(test_x1)
    a = accuracy_score(train_y1, x_train_pred1)*100
    b = accuracy_score(test_y1, x_test_pred1)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f'Accuracy rate of {model} model on training data is: ',a)
    print(f'Accuracy rate of {model} model on testing data is: ',b)
    print('\n----------------------------------------------------')
    #print(f'Precision rate of {model} model on training data is: ',precision_score(train_y1, x_train_pred1)*100)
    print(f'Precision rate of {model} model on testing data is: ',precision_score(test_y1, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    #print(f'Precision rate of {model} model on training data is: ',recall_score(train_y1, x_train_pred1)*100)
    print(f'Recall rate of {model} model on testing data is: ',recall_score(test_y1, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    print(f'F1 score of {model} model on testing data is: ',f1_score(test_y1, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    print(f'Confusion Matrix of {model} on testing is: ')
    print('\n----------------------------------------------------')
    cm = confusion_matrix(test_y1, x_test_pred1)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,cmap = 'summer',annot=True,fmt ='g') # annot:注釋, fmt:數字格式
    plt.show()
    
model_prediction(LogisticRegression())
model_prediction(SVC())
model_prediction(KNeighborsClassifier())
model_prediction(GaussianNB())

# 2. unscaled data :Decision tree、Random Forest、Ade boost、 Gradient-Boosting-Classifier
def model_prediction(model):
    model.fit(train_x, train_y)
    x_train_pred1 = model.predict(train_x)
    x_test_pred1 = model.predict(test_x)
    a = accuracy_score(train_y, x_train_pred1)*100
    b = accuracy_score(test_y, x_test_pred1)*100
    training_score.append(a)
    testing_score.append(b)
    
    print(f'Accuracy rate of {model} model on training data is: ',a)
    print(f'Accuracy rate of {model} model on testing data is: ',b)
    print('\n----------------------------------------------------')
    #print(f'Precision rate of {model} model on training data is: ',precision_score(train_y1, x_train_pred1)*100)
    print(f'Precision rate of {model} model on testing data is: ',precision_score(test_y, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    #print(f'Precision rate of {model} model on training data is: ',recall_score(train_y1, x_train_pred1)*100)
    print(f'Recall rate of {model} model on testing data is: ',recall_score(test_y, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    print(f'F1 score of {model} model on testing data is: ',f1_score(test_y, x_test_pred1)*100)
    print('\n----------------------------------------------------')
    print(f'Confusion Matrix of {model} on testing is: ')
    print('\n----------------------------------------------------')
    cm = confusion_matrix(test_y, x_test_pred1)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm,cmap = 'summer',annot=True,fmt ='g') # annot:注釋, fmt:數字格式
    plt.show()
    
model_prediction(DecisionTreeClassifier())
model_prediction(RandomForestClassifier())
model_prediction(AdaBoostClassifier())
model_prediction(GradientBoostingClassifier())
model_prediction(XGBClassifier())

# All models comparison
models = ["Logistic Regression","KNN","SVM","Naive Bayes","Decision Tree","Random Forest","Ada Boost",
          "Gradient Boost","XGBoost"]
df = pd.DataFrame({'Algorithms':models,
                   'Training score':training_score,
                   'Testing score':testing_score})
# visualzing score
df.plot(x = 'Algorithms',y = ['Training score','Testing score'], figsize = (16,6),kind = 'bar', title="Performance Visualization of Different Models",colormap="Set1")
plt.show()

# xgboost 調參
model1 = XGBClassifier()
parameters1 = {'n_estimators':[50,100,150],
               'random_state':[0,1,2],
               'learning_rate':[0.1,0.3,0.5,1.0]}

grid_search1 = GridSearchCV(model1, parameters1,n_jobs=-1)
grid_search1.fit(train_x,train_y)
grid_search1.best_score_
best_xgb = grid_search1.best_params_
# 使用最適參數
model1 = XGBClassifier(**best_xgb)
model1.fit(train_x,train_y)
x_test_pred1 = model1.predict(test_x)
accuracy_score(test_y, x_test_pred1) # 0.8085

# RF 調參
model2 = RandomForestClassifier()
# 1. n_estimators 數量，繪製學習曲線
score_lst = []
for i in range(100,550,10):
    rfm = RandomForestClassifier(n_estimators = i,random_state=0)
    score = cross_val_score(rfm,train_x,train_y).mean()
    score_lst.append(score)
max_score = max(score_lst)
print(f'最高分：{max_score}')    
print('子樹數量為：{}'.format(score_lst.index(max_score)*10+100))    

plt.plot(score_lst)
plt.show()
# 提升精度
score_lst = []
for i in range(375,385):
    rfm = RandomForestClassifier(n_estimators = i,random_state=0)
    score = cross_val_score(rfm,train_x,train_y).mean()
    score_lst.append(score)
max_score = max(score_lst)
print(f'最高分：{max_score}')    
print('子樹數量為：{}'.format(score_lst.index(max_score)+375))    

plt.plot(score_lst)
plt.show()

# 2. max_depth、min_samples_leaf、min_samples_split
parameters2 = {'max_depth':np.arange(1,20),'min_samples_split':np.arange(1,10),
               'min_samples_leaf':np.arange(1,10)}
rfd = RandomForestClassifier(n_estimators = 381, random_state=0)
grid_seache2 = GridSearchCV(rfd,parameters2,n_jobs=-1) 
grid_seache2.fit(train_x, train_y)
grid_seache2.best_score_
hyper_paras = grid_seache2.best_params_
# 14,6,2

model_rf = RandomForestClassifier(**hyper_paras,n_estimators = 381,random_state=0)
model_rf.fit(train_x, train_y)
hyper_rf_pred = model_rf.predict(test_x)
accuracy_score(test_y,hyper_rf_pred) # 0.8039

# Staking Model
staking_model = StackingClassifier(estimators=[('XGB',model1),('RF',model_rf)])
staking_model.fit(train_x, train_y)
x_train_pred_stake = staking_model.predict(train_x)
x_test_pred_stake = staking_model.predict(test_x)
print("Stacking Model accuracy on Training Data is:",accuracy_score(train_y, x_train_pred_stake))
print("Stacking Model accuracy on Testing Data is:",accuracy_score(test_y, x_test_pred_stake))

# Output of kaggle
ANS = staking_model.predict(test_df)
pass_df['Transported'] = ANS
pass_df['Transported'].replace({1:True,0:False},inplace=True) 

pass_df.shape

# save file
pass_df.to_csv("/Users/wangshuyou/Kaggle/spaceship-titanic/ETE_spaceship_prediction_project.csv",index=False)
