import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data =pd.read_csv("salarydata.csv")


# Replace "?" with NaN
data[["workclass","native-country","occupation"]]=data[["workclass","native-country","occupation"]].replace("?",np.NaN)
# fill the word "missing " in the place of categorical null values
for i in ['workclass','native-country','occupation']:
    data[i] = data[i].fillna('Missing')

data = data.drop(['education','education-num','marital-status','relationship', 'race','capital-gain',
       'capital-loss','hours-per-week','native-country'], axis=1)
#print(data)

data.drop_duplicates(inplace=True)

for col in data.columns:
    if data[col].dtypes == 'object':
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
#print(data)

x = data.drop("salary",axis =1)
y = data["salary"]

#split the dataset inti train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2,random_state=42)

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
#gbc_predict = gbc.predict(x_test)
#print(gbc_predict)


#write the model classifier using pickle
pickle.dump(gbc,open ('model.pkl','wb'))
