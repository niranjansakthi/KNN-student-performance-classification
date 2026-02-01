# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance.csv")
columns_to_drop = ['student_id', 'grade']
df = df.drop(columns=columns_to_drop)
df = df.dropna()

df['performance_level'] = pd.cut(
    df['total_score'],
    bins=[0, 50, 75, 100],
    labels=['Low', 'Medium', 'High']
)

# Drop original continuous column
df = df.drop(columns=['total_score'])


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['performance_level'] = le.fit_transform(df['performance_level'])
# Low=0, Medium=1, High=2


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

f1_scores = []

for k in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    f1_scores.append(f1_score(Y_test, preds, average='macro'))


plt.figure()
plt.plot(range(1, 40), f1_scores)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Macro F1-score")
plt.title("K vs F1-score")
plt.show()


best_k = np.argmax(f1_scores) + 1
print("Best K:", best_k)


final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, Y_train)

Y_pred = final_model.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

accuracy = final_model.score(X_test, Y_test)
print("Accuracy:", accuracy)

sample_student = np.array([[15, 85, 3]])

sample_student_pred = sc.transform(sample_student)
prediction = final_model.predict(sample_student_pred)
labels = {0: "Low", 1: "Medium", 2: "High"}
print("Predicted students performance:", labels[prediction[0]])
