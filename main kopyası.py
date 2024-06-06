import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Veri setini yükleme
data = pd.read_csv("güncellenmişSon1.csv")
data=data.drop(["Unnamed: 0","second","src","dst"],axis=1)
label1data=data[data["label"]==1]
label0data=data[data["label"]==0]
selectedData=label1data.sample(n=1230,random_state=42)
data=pd.concat([selectedData,label0data])


# Hedef değişken
target = "label"

# Kategorik değişkeni sayısal hale getirme
data["wpan_dst_addr_mode"] = data["wpan_dst_addr_mode"].apply(lambda x: 1 if x == 'Long/64-bit' else 0)

# Özellikler ve hedef değişken
X = data.drop(target, axis=1)
y = data[target]

# Sayısal ve kategorik özellikleri belirleme
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Sayısal veri için işlemler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Kategorik veri için işlemler
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer ile her iki işlemi birleştirme
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# RandomForestClassifier ve DecisionTreeClassifier için hiperparametre ayarları
rf_params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}

dt_params = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}

# RandomForestClassifier ve DecisionTreeClassifier içeren Pipeline'ın oluşturulması
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])

dt_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier(random_state=42))])

# GridSearchCV ile hiperparametre araması yapma
rf_model = GridSearchCV(rf_model, rf_params, cv=3, scoring='accuracy')
dt_model = GridSearchCV(dt_model, dt_params, cv=3, scoring='accuracy')

# Hibrit model oluşturma
hybrid_model = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('decision_tree', dt_model)
], voting='hard')

# Veriyi train ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hibrit Modeli eğitme
hybrid_model.fit(X_train, y_train)

# Modeli kaydetme
joblib.dump(hybrid_model, 'hibrit_model.pkl')

# Hibrit Modelin doğruluğunu değerlendirme
hybrid_predictions = hybrid_model.predict(X_test)
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
print("Hibrit Model Doğruluğu:", hybrid_accuracy)

# Confusion matrix oluşturma
conf_matrix = confusion_matrix(y_test, hybrid_predictions)

# Confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=hybrid_model.classes_,
            yticklabels=hybrid_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
