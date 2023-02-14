#!/usr/bin/env python
# coding: utf-8

# # Fundamentals of Data Science project

# # Razi Haider Bhatti (19I-1762), BS(DS)-N

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder 


# # Data cleaning and handling '?' values

# In[2]:


data = pd.read_csv('Blood_Pressure_data.csv')
data


# In[3]:


for i in data:
    print(data[i].value_counts())


# In[4]:


data = data.replace('?', np.nan)


# In[5]:


data['medical_specialty'].value_counts()


# In[6]:


df = data


# In[7]:


df['gender'] = df['gender'].replace(np.nan, 0) 


# In[8]:



df['weight'].value_counts()
i = 101765 - 98569
(98569/101765)*100
# next column
df['medical_specialty'].value_counts()
(49949/101765)*100


# In[9]:


df.count()


# In[10]:


df['payer_code'].value_counts(dropna = False)
df['payer_code'] = df['payer_code'].replace(np.nan, 'MC')


# In[11]:



d = df


# In[12]:


for columns in d:
    per = d[columns].isnull().sum()
    per = per / len(d)
    #print(per)
    if (per > 0.8):
        print(per)
        d = d.drop([columns], axis = 1)
d


# In[13]:


for i in d:
    print(d[i].value_counts(dropna = False))


# In[14]:



d['diag_1'].value_counts()


# In[15]:



d[['diag_1', 'diag_2', 'diag_3']] = d[['diag_1', 'diag_2', 'diag_3']].replace(np.nan, 'NaN')


count = 0
for i in d['diag_1']:
    if (d['diag_1'][count][0] == 'V' or d['diag_1'][count][0] == 'E'):
        d['diag_1'] = d['diag_1'].replace(d['diag_1'][count], 'NaN')
    count = count + 1

count1 = 0
for i in d['diag_2']:
    if (d['diag_2'][count1][0] == 'V' or d['diag_2'][count1][0] == 'E'):
        d['diag_2'] = d['diag_2'].replace(d['diag_2'][count1], 'NaN')
    count1 = count1 + 1

count2 = 0
for i in d['diag_3']:
    if (d['diag_3'][count2][0] == 'V' or d['diag_3'][count2][0] == 'E'):
        d['diag_3'] = d['diag_3'].replace(d['diag_3'][count2], 'NaN')
    count2 = count2 + 1


# In[16]:


d[['diag_1', 'diag_2', 'diag_3']] = d[['diag_1', 'diag_2', 'diag_3']].replace('NaN', np.nan)
d[['diag_1', 'diag_2', 'diag_3']] = d[['diag_1', 'diag_2', 'diag_3']].astype('float64')

m = d['diag_1'].mean()
m1 = d['diag_2'].mean()
m2 = d['diag_3'].mean()

print(m)
print(m1)
print(m2)


# In[17]:


d['cast'].value_counts(dropna = False)
d['cast'] = d['cast'].replace(np.nan, 'Caucasian')


# In[18]:


d[['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'patient_no', 'number_outpatient', 'number_inpatient', 'admission_typeid', 'discharge_disposition_id', 'admission_source_id']] = d[['time_in_hospital','num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'patient_no', 'number_outpatient', 'number_inpatient', 'admission_typeid', 'discharge_disposition_id', 'admission_source_id']].astype('float64')


n = d['medical_specialty'].mode()
d['medical_specialty'] = d['medical_specialty'].fillna('InternalMedicine')

# diags
d['diag_1'] = d['diag_1'].replace(np.nan, m).replace('NaN', m)
d['diag_2'] = d['diag_2'].replace(np.nan, m1).replace('NaN', m1)
d['diag_3'] = d['diag_3'].replace(np.nan, m2).replace('NaN', m2)


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


xl = d['patient_no']
plt.hist(xl)
plt.xlabel('patient number (should be unique)')
plt.ylabel('frequency')


# In[21]:


d.drop_duplicates(subset ="patient_no", keep = False, inplace = True)


# In[22]:


d


# In[23]:



d['gender'].value_counts()


# In[24]:


d['age group'] = d['age group'].replace('[0-10)', 5).replace('[10-20)', 15).replace('[20-30)', 25).replace('[30-40)', 35).replace('[40-50)', 45).replace('[50-60)', 55).replace('[60-70)', 65).replace('[70-80)', 75).replace('[80-90)', 85).replace('[90-100)', 95)
d


# In[25]:


for i in d:
    print(d[i].value_counts(dropna = False))


# In[26]:


d['Med'] = d['Med'].replace(np.nan, 'NO')
d['change'] = d['change'].replace(np.nan, 'NO')
d['number_diagnoses'] = d['number_diagnoses'].replace(np.nan, 'NO')
d['payer_code'] = d['payer_code'].replace(np.nan, 'MC')
d['max_glu_serum'] = d['max_glu_serum'].replace('>200', 1).replace('>300', 1).replace('None', 0).replace('Norm', 0)
d['A1Cresult'] = d['A1Cresult'].replace('>8', 1).replace('>7', 1).replace('None', 0).replace('Norm', 0)
d['rosiglitazone'] = d['rosiglitazone'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['pioglitazone'] = d['pioglitazone'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['glyburide'] = d['glyburide'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['glipizide'] = d['glipizide'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['glimepiride'] = d['glimepiride'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['metformin'] = d['metformin'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['insulin'] = d['insulin'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['repaglinide'] = d['repaglinide'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['nateglinide'] = d['nateglinide'].replace('No', 0).replace('Steady', 0).replace('Up', 2).replace('Down', 1)
d['Med'] = d['Med'].replace('No', 0).replace('Yes', 1)
d['change'] = d['change'].replace('Ch', 1).replace('No', 0)


# In[27]:


d


# In[28]:


for i in d:
    print(d[i].value_counts())


# In[29]:


d.count()
d['label'].value_counts()


# In[30]:


le = LabelEncoder()


# In[31]:


d


# # Encoding using fit_transform

# In[32]:


for i in d:
    d[i] = le.fit_transform(d[i])
d


# In[33]:


d.count()


# # Plotting some graphs

# In[34]:


import matplotlib.pyplot as plt


# In[35]:


x = d['age group'].head(20)
y = d['label'].head(20)
plt.xlabel('age group')
plt.hist(x)


# In[36]:


x = d['diag_1']
y = d['label']

plt.xlabel('diag_1')
plt.ylabel('label')

plt.hist(x)


# In[37]:


x = d['diag_2']
y = d['label']

plt.xlabel('diag_2')
plt.ylabel('label')

plt.hist(x)


# In[38]:


x = d['diag_3']
y = d['label']

plt.xlabel('diag_3')
plt.ylabel('label')

plt.hist(x)


# In[39]:


x = d['change']
y = d['label'].head(20)

plt.xlabel('change')
plt.ylabel('label')

plt.hist(x)


# In[40]:


x = d['number_outpatient']
y = d['label']

plt.xlabel('number_outpatient')
plt.ylabel('label')

plt.hist(x)


# In[41]:


x = d['admission_typeid']
y = d['label'].head(20)

plt.xlabel('admission_typeid')
plt.ylabel('label')

plt.hist(x)


# In[42]:


x = d['num_lab_procedures'].head(20)
y = d['label'].head(20)

plt.xlabel('num_lab_procedures')
plt.ylabel('label')

plt.scatter(x, y)


# In[43]:


x = d['num_procedures'].head(30)
y = d['label'].head(30)

plt.xlabel('num_procedures')
plt.ylabel('label')

plt.scatter(x, y)


# In[44]:


x = d['insulin'].head(20)
y = d['label'].head(20)

plt.xlabel('insulin')
plt.ylabel('label')

plt.scatter(x, y)


# In[45]:


d.count()


# In[46]:


for i in d:
    print(d[i].value_counts(dropna = False))


# # Machine Learning and Data splitting

# In[47]:


from sklearn.preprocessing import normalize as nm


# In[48]:


xD = d.drop(['label'], axis = 1)
xd = xD.drop(['medical_specialty', 'payer_code','metformin-rosiglitazone', 'metformin-pioglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 'citoglipton', 'examide', 'tolazamide', 'troglitazone', 'miglitol', 'acarbose', 'tolbutamide', 'acetohexamide', 'chlorpropamide', 'acarbose'], axis=1)
x = xd.to_numpy()
X = nm(x, norm='l2', axis=1, copy=True, return_norm = False)
y = d['label'].to_numpy()


# In[49]:


xd.count()


# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[51]:


train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.8, random_state = 27)


# # Normalizing data

# In[52]:


norm = MinMaxScaler().fit(train_x)
train_x_norm = norm.transform(train_x)
test_x_norm = norm.transform(test_x)

train_x_stand = train_x
test_x_stand = test_x


# In[53]:


scale = StandardScaler().fit(train_x_stand)
train_x_stand = scale.transform(train_x_stand)
test_x_stand = scale.transform(test_x_stand)


# # Using various algorithms for data selection (including graph)

# In[54]:


from sklearn.feature_selection import SelectFromModel as sfm
from sklearn.ensemble import RandomForestClassifier as rfc


# In[55]:


Y = d[['label']]
e = sfm(rfc(n_estimators=100), max_features=24)
e.fit(xd, Y)

e_s = e.get_support()
e_f = xd.loc[:,e_s].columns.tolist()
print(str(len(e_f)), 'selected features')
print(e_f)


# In[56]:


from sklearn.feature_selection import SelectKBest as sk
from sklearn.feature_selection import chi2 as ch 
from sklearn.ensemble import ExtraTreesClassifier


# In[57]:


Y = d[['label']]
et = ExtraTreesClassifier()
et.fit(xd, Y)
print(et.feature_importances_)
fi = pd.Series(et.feature_importances_, index = xd.columns)
fi.nlargest(20).plot(kind = 'barh')


# In[58]:


Y = d[['label']]
best = sk(score_func = ch, k = 10)
fit = best.fit(xd, Y)
dfsc = pd.DataFrame(fit.scores_)
dfcol = pd.DataFrame(xd.columns)
fsc = pd.concat([dfcol, dfsc], axis = 1)
fsc.columns = ['features', 'score']
#columns
print(fsc.nlargest(100, 'score'))


# In[59]:


d['label'].value_counts()


# In[60]:


xd


# In[61]:


train_x


# In[62]:


test_x


# In[63]:


train_y


# In[64]:


test_y


# In[65]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier as knc


# In[66]:


test_y


# # Using KNN classifier

# In[67]:


# KNN METHOD
model = knc(n_neighbors=7)
model.fit(train_x, train_y)
pred = model.predict(test_x)

a1 = accuracy_score(test_y, pred)*100
f1 =  f1_score(test_y, pred, average='micro')*100

print("Accuracy : ", a1, '%')
print("F1 score : ", f1, '%')


# In[68]:


pred


# # Using Logistic regression method

# In[69]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# In[70]:


# LOGISTIC REGRESSION METHOD
 
lr = LogisticRegression().fit(train_x_stand, train_y)
y_pred = lr.predict(test_x_stand)
score = lr.score(test_x_stand, test_y)

a9 = score*100
f9 =  f1_score(test_y, y_pred, average='micro')*100

print("Accuracy : ", a9, '%')
print("F1 score : ", f9, '%')


# In[71]:


y_pred


# # Using DecisionTreeClassifier

# In[101]:


from sklearn.tree import DecisionTreeClassifier


# In[102]:


dtc = DecisionTreeClassifier().fit(train_x_norm,train_y)
pred_Y = dtc.predict(test_x_norm)

a2 = accuracy_score(test_y, pred_Y)*100
f2 =  f1_score(test_y, pred_Y, average='micro')*100

print("Accuracy : ", a2, '%')
print("F1 score : ", f2, '%')


# # Using AdaBoostClassifier

# In[74]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[75]:


adc = AdaBoostClassifier().fit(train_x_norm, train_y)
prediction_y = adc.predict(test_x_norm)

a3 = accuracy_score(test_y, prediction_y)*100
f3 =  f1_score(test_y, prediction_y, average='micro')*100

print("Accuracy : ", a3, '%')
print("F1 score : ", f3, '%')


# # Using RandomForestClassifier

# In[76]:


rfc = RandomForestClassifier().fit(train_x_norm, train_y)
prediction = rfc.predict(test_x_norm)

a4 = accuracy_score(test_y, prediction)*100
f4 =  f1_score(test_y, prediction, average='micro')*100

print("Accuracy : ", a4, '%')
print("F1 score : ", f4, '%')


# # Using GaussianNB from naive_bayes

# In[77]:


from sklearn.naive_bayes import GaussianNB 


# In[78]:


gnb = GaussianNB().fit(train_x, train_y) 
pre_y = gnb.predict(test_x)

a5 = accuracy_score(test_y, pre_y)*100
f5 =  f1_score(test_y, pre_y, average='micro')*100

print("Accuracy : ", a5, '%')
print("F1 score : ", f5, '%')


# # Using GradientBoostingClassifier

# In[79]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


# In[80]:


gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(train_x_norm, train_y)
a6 = gbc.score(test_x_norm, test_y)
a6 = a6*100
predi_y = gbc.predict(test_x_norm)


f6 =  f1_score(test_y, predi_y, average='micro')*100

print("Accuracy : ", a6, '%')
print("F1 score : ", f6, '%')


# # Using HistGradientBoostingClassifier

# In[81]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2


# In[82]:


hgb = HistGradientBoostingClassifier(max_iter=100).fit(train_x_norm, train_y)
a7 = hgb.score(test_x_norm, test_y)
a7 = a7*100
predh_y = hgb.predict(test_x_norm)

f7 =  f1_score(test_y, predh_y, average='micro')*100

print("Accuracy : ", a7, '%')
print("F1 score : ", f7, '%')


# # LinearDiscriminantAnalysis classifier

# In[83]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[84]:


lda = LinearDiscriminantAnalysis().fit(train_x_norm, train_y)
P_y = lda.predict(test_x_norm)

a8 = accuracy_score(test_y, P_y)*100
f8 =  f1_score(test_y, P_y, average='micro')*100

print("Accuracy : ", a8, '%')
print("F1 score : ", f8, '%')


# # Classifiers and their respective accuracies and F scores using graphs

# In[85]:


cl_list = ['KNN','Logistic Regression', 'DecisionTreeClassifier', 'AdaBoostClassifier', 'RandomForestClassifier', 'GaussianNB', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'LinearDiscriminantAnalysis']
acc_list = [a1, a9, a2, a3, a4, a5, a6, a7, a8]
f_list = [f1, f9, f2, f3, f4, f5, f6, f7, f8]


# In[86]:


plt.barh(cl_list, acc_list, color = 'orange')
plt.xlabel('Accuracy (%)')


# In[87]:


plt.barh(cl_list, f_list, color = 'green')
plt.xlabel('F score (%)')


# In[88]:


plt.pie(acc_list, labels = cl_list, autopct='%1.1f%%', startangle=90)


# In[89]:


xd


# # Using HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, and LinearDiscriminantAnalysis (5 best models)

# In[90]:


l1 = []
for i in P_y:
    l1.append(i)
print(l1)


# In[91]:


l2 = []
for i in prediction_y:
    l2.append(i)
print(l2)


# In[92]:


l3 = []
for i in prediction:
    l3.append(i)
print(l3)


# In[93]:


l4 = []
for i in predi_y:
    l4.append(i)
print(l4)


# In[94]:


l5 = []
for i in predh_y:
    l5.append(i)
print(l5)


# In[95]:


print(len(l3))


# In[96]:


label = []
for i in range(0, len(l3)):
    c0 = 0
    c1 = 0
    c2 = 0
    if(l1[i] == 1):
        c1 = c1 + 1
    elif(l1[i] == 2):
        c2 = c2 + 1
    else:
        c0 = c0 + 1
        
    if(l2[i] == 1):
        c1 = c1 + 1
    elif(l2[i] == 2):
        c2 = c2 + 1    
    else:
        c0 = c0 + 1
        
    if(l3[i] == 1):
        c1 = c1 + 1
    elif(l3[i] == 2):
        c2 = c2 + 1
    else:
        c0 = c0 + 1
        
    if(l4[i] == 1):
        c1 = c1 + 1
    elif(l4[i] == 2):
        c2 = c2 + 1
    else:
        c0 = c0 + 1
        
    if(l5[i] == 1):
        c1 = c1 + 1
    elif(l5[i] == 2):
        c2 = c2 + 1
    else:
        c0 = c0 + 1
    
    if(c1 > c0 and c1 > c2):
        label.append(1)
    elif(c2 > c1 and c2 > c0):
        label.append(2)
    else:
        label.append(0)
        
print(label)


# In[97]:


test = []
for i in test_y:
    test.append(i)
print(test)


# In[98]:


len(label)


# In[99]:


count = 0
for i in range(0, len(label)):
    if (label[i] == test[i]):
        count = count + 1
print(count)


# In[100]:


print("ACCURACY BY MAJORITY : ", (count/len(label))*100, "%")

