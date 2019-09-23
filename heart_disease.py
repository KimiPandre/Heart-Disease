import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("heart.csv")

data.isnull().sum()

data.info()

col = {'age':'age', 'sex':'sex', 'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure', 
       'chol':'cholesterol', 'fbs':'fasting_blood_sugar', 'restecg':'rest_ecg', 
       'thalach':'max_heart_rate_achieved', 'exang':'exercise_induced_angina', 'oldpeak':'st_depression', 
       'slope':'st_slope', 'ca':'num_major_vessels', 'thal':'thalassemia', 'target':'target'}

data = data.rename(columns = col)

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()

sns.pairplot(data)
plt.show()

#AGE ANALYSIS

data.age.value_counts()

plt.figure(figsize=(40,40))
sns.barplot(x=data.age.value_counts().index,y=data.age.value_counts().values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.title('Age Analysis System')
plt.show()
#the best analysis can be divided into the elderly,middle-aged, young people by looking at the 
#age ranges.

minAge = min(data.age)
maxAge = max(data.age)
meanAge = data.age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)

young_ages = data[(data.age>=29) & (data.age<40)]
middle_ages = data[(data.age>=40) & (data.age<55)]
elderly_ages = data[(data.age>55)]
print('Young Ages :',len(young_ages))
print('Middle Ages :',len(middle_ages))
print('Elderly Ages :',len(elderly_ages))

sns.barplot(x=['young ages','middle ages','elderly ages'], y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()
#a new feature value can be removed from these age ranges will not affect this impact will see 
#in the future.

colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize = (5,5))
plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)], labels=['young ages','middle ages','elderly ages'],
         explode=explode, colors=colors, autopct='%1.1f%%')
plt.title('Age States',color = 'blue',fontsize = 15)
plt.show()

data['AgeRange']=0
youngAge_index=data[(data.age>=29)&(data.age<40)].index
middleAge_index=data[(data.age>=40)&(data.age<55)].index
elderlyAge_index=data[(data.age>55)].index

for index in elderlyAge_index:
    data.loc[index,'AgeRange']=2
    
for index in middleAge_index:
    data.loc[index,'AgeRange']=1

for index in youngAge_index:
    data.loc[index,'AgeRange']=0
    
sns.swarmplot(x="AgeRange", y="age",hue='sex',
              palette=["r", "c", "y"], data=data)
plt.show()

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(y="AgeRange", x="sex", data=data,
            label="Total", color="b")
plt.show()

sns.set_color_codes("pastel")
sns.barplot(y=, x="sex", data=data,
            label="Total", color="b")
plt.show()

sns.countplot(elderly_ages.sex)
plt.title("Elderly age people sex ratio")
plt.show()

elderly_ages.groupby(elderly_ages['sex'])['max_heart_rate_achieved'].agg('sum')

sns.barplot(x=elderly_ages.groupby(elderly_ages['sex'])['max_heart_rate_achieved'].agg('sum').index,
            y=elderly_ages.groupby(elderly_ages['sex'])['max_heart_rate_achieved'].agg('sum').values)
plt.title("Gender Group max heart rate achieved  Sum Time")
plt.show()

sns.violinplot(data.age, palette="Set3", bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=90)
plt.title("Age Rates")
plt.show()

plt.figure(figsize=(15,7))
sns.violinplot(x=data.age,y=data.target)
plt.xticks(rotation=90)
plt.legend()
plt.title("Age & Target System")
plt.show()

#SEX(GENDER ANALYSIS)

data['sex'].value_counts()

sns.countplot(data.sex)
plt.show()

sns.countplot(data.sex,hue=data.st_slope)
plt.title('Slope & Sex Rates Show')
plt.show()

total_genders_count = len(data.sex)
male_count = len(data[data['sex'] == 1])
female_count = len(data[data['sex'] == 0])
print('Total Genders :',total_genders_count)
print('Male Count    :',male_count)
print('Female Count  :',female_count)

#Percentage ratios
print("Male State: {:.2f}%".format((male_count / (total_genders_count)*100)))
print("Female State: {:.2f}%".format((female_count / (total_genders_count)*100)))

#sex and the heart health situation
male_andtarget_on = len(data[(data.sex == 1) & (data['target'] == 1)])
male_andtarget_off = len(data[(data.sex == 1) & (data['target'] == 0)])

sns.barplot(x = ['Male Target On','Male Target Off'], y = [male_andtarget_on, male_andtarget_off])
plt.xlabel('Male and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

female_andtarget_on = len(data[(data.sex == 0) & (data['target'] == 1)])
female_andtarget_off = len(data[(data.sex == 0) & (data['target'] == 0)])

sns.barplot(x = ['Female Target On','Female Target Off'] ,y = [female_andtarget_on, female_andtarget_off])
plt.xlabel('Female and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

sns.relplot(x="resting_blood_pressure", y="age", sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)
# the gender of people who are female is more common



#Chest Pain Type Analysis
'''A wide range of chest pain is present in cases of heart failure. These pains will be analyzed
 according to their problems and age ranges in the analysis system.'''
 
data['chest_pain_type'].value_counts()

sns.countplot(data.chest_pain_type)
plt.xlabel('Chest Type')
plt.ylabel('Count')
plt.title('Chest Type vs Count State')
plt.show()
#0 status at least
#1 condition slightly distressed
#2 condition medium problem
#3 condition too bad

cp_zero_target_zero = len(data[(data.chest_pain_type == 0) & (data.target == 0)])
cp_zero_target_one = len(data[(data.chest_pain_type == 0) & (data.target == 1)])
sns.barplot(x = ['cp_zero_target_zero','cp_zero_target_one'], y = [cp_zero_target_zero,cp_zero_target_one])
plt.show()

cp_zero_target_zero = len(data[(data.chest_pain_type == 1) & (data.target == 0)])
cp_zero_target_one = len(data[(data.chest_pain_type == 1) & (data.target == 1)])
sns.barplot(x = ['cp_zero_target_zero','cp_zero_target_one'], y = [cp_zero_target_zero,cp_zero_target_one])
plt.show()

cp_zero_target_zero = len(data[(data.chest_pain_type == 2) & (data.target == 0)])
cp_zero_target_one = len(data[(data.chest_pain_type == 2) & (data.target == 1)])
sns.barplot(x = ['cp_zero_target_zero','cp_zero_target_one'], y = [cp_zero_target_zero,cp_zero_target_one])
plt.show()

cp_zero_target_zero = len(data[(data.chest_pain_type == 3) & (data.target == 0)])
cp_zero_target_one = len(data[(data.chest_pain_type == 3) & (data.target == 1)])
sns.barplot(x = ['cp_zero_target_zero','cp_zero_target_one'], y = [cp_zero_target_zero,cp_zero_target_one])
plt.show()

#Age Range Analysis
target_0 = len(data[(data.target == 0) & (data.AgeRange == 0)])
target_1 = len(data[(data.target == 1) & (data.AgeRange == 0)])

colors = ['blue','green']
explode = [0,0.1]
plt.figure(figsize = (5,5))
plt.pie([target_0, target_1], explode=explode, labels=['Target 0 Age Range 0','Target 1 Age Range 0'],
        colors=colors, autopct='%1.1f%%')
plt.title('Target vs Age Range Young Age ',color = 'blue',fontsize = 15)
plt.show()

#Thalassemia Analysis
data.thalassemia.value_counts()

sns.barplot(x=data.thalassemia.value_counts().index, y = data.thalassemia.value_counts().values)
plt.xlabel('Thalach')
plt.ylabel('Count')
plt.title('Thalach Counts')
plt.xticks(rotation=45)
plt.show()

sns.swarmplot(x=data.age)
plt.title('Age Rates')
plt.show()

age_unique = sorted(data.age.unique())
age_values = data.groupby('age')['max_heart_rate_achieved'].count().values
mean_thalach = []
for i,age in enumerate(age_unique):
    mean_thalach.append(sum(data[data['age'] == age].max_heart_rate_achieved)/age_values[i])


plt.figure(figsize=(10,5))
sns.pointplot(x = age_unique, y = mean_thalach, color='red', alpha=0.8)
plt.xlabel('Age',fontsize = 15, color='blue')
plt.xticks(rotation=45)
plt.ylabel('Thalassemia', fontsize = 15, color='blue')
plt.title('Age vs Thalassemia',fontsize = 15,color='blue')
plt.grid()
plt.show()

#Our aim here is to obtain the average values of Thalach according to age ranges. Because we're 
#going to do chest pain.

age_range_thalach = data.groupby('AgeRange')['max_heart_rate_achieved'].mean()

sns.barplot(x = age_range_thalach.index, y = age_range_thalach.values)
plt.xlabel('Age Range Values')
plt.ylabel('Maximumm heart rate achieved By Age Range')
plt.title('illustration of the heart rate to the age range')
plt.show()
#As shown in this graph, this rate decreases as the heart rate 
#is faster and in old age areas.

cp_thalach = data.groupby('chest_pain_type')['max_heart_rate_achieved'].mean()

sns.barplot(x = cp_thalach.index, y = cp_thalach.values)
plt.xlabel('degree of chest pain')
plt.ylabel('Maximum heart rate achieved By Cp Values')
plt.title('Illustration of heart rate to degree of chest pain')
plt.show()
#As seen in this graph, it is seen that the heart rate is less 
#when the chest pain is low. But in cases where chest pain is 
#1, it is observed that the area is more. 2 and 3 were found to 
#be of the same degree.

#THALASSEMIA ANALYSIS
data['thalassemia'].value_counts()

sns.countplot(data['thalassemia'])
plt.show()

data[data['thalassemia'] == 0]

target_thal = data[(data['thalassemia']==1)].target.value_counts()

sns.barplot(x = target_thal.index, y = target_thal.values)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('number of people having heart disease according to thalassemia')
plt.show()

#Target 1
a=len(data[(data['target']==1)&(data['thalassemia']==0)])
b=len(data[(data['target']==1)&(data['thalassemia']==1)])
c=len(data[(data['target']==1)&(data['thalassemia']==2)])
d=len(data[(data['target']==1)&(data['thalassemia']==3)])
print('Target 1 Thal 0: ',a)
print('Target 1 Thal 1: ',b)
print('Target 1 Thal 2: ',c)
print('Target 1 Thal 3: ',d)
#so,Apparently, there is a rate at Thal 2.Now, draw graph
#Target 0
e=len(data[(data['Target']==0)&(data['Thal']==0)])
f=len(data[(data['Target']==0)&(data['Thal']==1)])
g=len(data[(data['Target']==0)&(data['Thal']==2)])
h=len(data[(data['Target']==0)&(data['Thal']==3)])
print('Target 0 Thal 0: ',e)
print('Target 0 Thal 1: ',f)
print('Target 0 Thal 2: ',g)
print('Target 0 Thal 3: ',h)

f,ax=plt.subplots(figsize=(7,7))
sns.barplot(y = ['T1 & Th0','T1 & Th1','T1 & Th2','T1 & Th3'], x = [1,6,130,28],
            color='green',alpha=0.5,label='Target1 thalassemia State')
sns.barplot(y = ['T1 & Th0','T1 & Th1','T1 & Th2','T1 & Th3'], x = [1,12,36,89],
            color='red',alpha=0.7,label='Target 0 Thal State')
ax.legend(loc='lower right',frameon=True)
ax.set(xlabel='Target State and Thal Counter',ylabel='Target State and Thal State',title='Target VS Thal')
plt.xticks(rotation=90)
plt.show()


#TARGET ANALYSIS

data.target.unique()
#only two values are shown.
#A value of 1 is the value of patient 0.

sns.countplot(data.target)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counter 1 & 0')
plt.show()

sns.countplot(data.target, hue = data.sex)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target & Sex Counter 1 & 0')
plt.show()

#determine the age ranges of patients with and without sickness and make analyzes about them
data['age'].unique()
agerange_target1 = []
agerange_target0 = []

for age in data.age.unique():
    agerange_target1.append(len(data[(data['age'] == age) & (data['target'] == 1)]))
    agerange_target0.append(len(data[(data['age'] == age) & (data['target'] == 0)]))
    
plt.scatter(x = data['age'].unique(), y = agerange_target1, color = 'red', label = 'Target1')
plt.scatter(x = data['age'].unique(), y = agerange_target0, color = 'blue', label = 'Target0')
plt.legend(loc = 'upper right', frameon = True)
plt.title('Target according to the AgeRange')
plt.xlabel('AgeRange')
plt.ylabel('Count')
plt.show()

sns.lineplot(x="sex", y="st_depression", hue="target", data=data)
plt.show()

#CHOLESTROL
g = sns.catplot(x="AgeRange", y="cholesterol", hue="sex", data=data, kind="bar")
plt.show()

ax = sns.barplot("sex", "cholesterol", data=data, linewidth=2.5)
plt.show()

male_young = data[(data['sex'] == 1) & (data['AgeRange'] == 0) & (data['target'] == 1)]
male_middle = data[(data['sex'] == 1) & (data['AgeRange'] == 1) & (data['target'] == 1)]
male_old = data[(data['sex'] == 1) & (data['AgeRange'] == 2) & (data['target'] == 1)]
print(" Young males having heart disease: ", len(male_young))
print(" Middle aged males having heart disease: ", len(male_middle))
print(" Old males having heart disease: ", len(male_old))

f,ax1=plt.subplots(figsize=(20,10))
sns.pointplot(x = np.arange(len(male_young)), y = male_young.resting_blood_pressure, color = 'black',
              alpha = 0.8, label = 'Young')
sns.pointplot(x = np.arange(len(male_middle)), y = male_middle.resting_blood_pressure, color = 'red',
              alpha = 0.8, label = 'Middle')
sns.pointplot(x = np.arange(len(male_old)), y = male_old.resting_blood_pressure, alpha = 0.8,
              color = 'lime', label = 'Old')
plt.xlabel('Range', fontsize = 15, color = 'Blue')
plt.xticks(rotation = 90)
plt.legend(loc = 'upper right', frameon = True)
plt.ylabel('Resting Blood Pressure', fontsize = 15, color = 'Blue')
plt.title('Age range values VS Blood Pressure')
plt.grid()
plt.show()




data_filter_mean = data[(data['target']==1)&(data['age']>50)].groupby('sex')[['resting_blood_pressure','cholesterol','max_heart_rate_achieved']].mean()
data_filter_mean.describe()


# MODEL, TRAINING and TESTING

d = data.corr()

X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)

y_pred = log.predict(X_test)

#CONFUSION METRICS
from sklearn import metrics
m = metrics.confusion_matrix(y_test,y_pred)
print(m) 
print("Accuracy: ", metrics.accuracy_score(y_test,y_pred)) 

x1 = [0,1]
ax = sns.heatmap(pd.DataFrame(m), annot = True, fmt='g', linecolor = 'g', cbar = True, 
                 square = True, xticklabels=x1, yticklabels=x1)
ax.set_title("Confusion metrics")
plt.show()

print("Precision score: ", metrics.precision_score(y_test,y_pred))
print("Recall score: ", metrics.recall_score(y_test,y_pred))
print("Fscore: ", metrics.f1_score(y_test,y_pred))

#ROC CURVE WITH AOC
#predict_proba:gives you the probabilities for the target (0 and 1 in your case) in array form. 
#AOC:Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
y_predict_prob = log.predict_proba(X_test)
#predict_proba gives you the probabilities for the target (0 and 1 in your case) in array form. 
y_predict_prob = y_predict_prob[:,-1] #for 1
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob)
aoc = metrics.roc_auc_score(y_test, y_predict_prob)
label = "Logistic(area = {})".format(aoc)
plt.plot(fpr, tpr)
plt.text(0.05, 0.2, label)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

# Naive Baes
from sklearn.naive_bayes import GaussianNB
gb = GaussianNB()
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

m = metrics.confusion_matrix(y_test,y_pred)
print(m) 
print("Accuracy: ", metrics.accuracy_score(y_test,y_pred)) 

ax = sns.heatmap(pd.DataFrame(m), annot = True, fmt='g', linecolor = 'g', cbar = True, 
                 square = True, xticklabels=x1, yticklabels=x1)
ax.set_title("Confusion metrics")
plt.show()

print("Precision score: ", metrics.precision_score(y_test,y_pred))
print("Recall score: ", metrics.recall_score(y_test,y_pred))
print("Fscore: ", metrics.f1_score(y_test,y_pred))

y_predict_prob = gb.predict_proba(X_test)
y_predict_prob = y_predict_prob[:,-1] #for 1
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob)
aoc = metrics.roc_auc_score(y_test, y_predict_prob)
label = "Gaussian(area = {})".format(aoc)
plt.plot(fpr, tpr)
plt.text(0.05, 0.2, label)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

#DECISION TREE
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeClassifier
df = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None,random_state=1)
df = df.fit(X_train, y_train)

y_pred = df.predict(X_test)

print("Accuracy score:", sm.accuracy_score(y_test, y_pred))

y_predict_prob = df.predict_proba(X_test)
y_predict_prob = y_predict_prob[:,-1] #for 1
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob)
aoc = metrics.roc_auc_score(y_test, y_predict_prob)
label = "Decision Tree(area = {})".format(aoc)
plt.plot(fpr, tpr)
plt.text(0.05, 0.2, label)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()

#ROC Curve (converting to binary classifier)

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc

y = label_binarize(y, classes=[0,1,2])
n_classes = 3

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

























