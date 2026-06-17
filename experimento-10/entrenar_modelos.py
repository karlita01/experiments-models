import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier
import joblib


# ===============================
# CARGAR DATASET
# ===============================

CSV_PATH = r"C:\Users\User\Desktop\EXP11\features_videos.csv"

df = pd.read_csv(CSV_PATH)

print("Dataset cargado correctamente")
print(df["clase"].value_counts())
print("Total de muestras:", len(df))


# ===============================
# PREPARAR X / y
# ===============================

# Quitamos columnas que no son features
X = df.drop(columns=["video", "clase"])
y = df["clase"]

# Convertir etiquetas texto a números
# normal -> 0, robo -> 1
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClases:")
for clase, valor in zip(encoder.classes_, encoder.transform(encoder.classes_)):
    print(clase, "=", valor)


# ===============================
# DIVISIÓN TRAIN / TEST
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain:", len(X_train))
print("Test:", len(X_test))


# ===============================
# MODELO 1: SVM
# ===============================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
])

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("\n===============================")
print("RESULTADOS SVM")
print("===============================")

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_svm,
    target_names=encoder.classes_
))


# ===============================
# MODELO 2: XGBOOST
# ===============================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n===============================")
print("RESULTADOS XGBOOST")
print("===============================")

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_xgb,
    target_names=encoder.classes_
))


# ===============================
# VALIDACIÓN CRUZADA
# ===============================

print("\n===============================")
print("VALIDACIÓN CRUZADA")
print("===============================")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_scores = cross_val_score(svm_model, X, y_encoded, cv=cv, scoring="accuracy")
xgb_scores = cross_val_score(xgb_model, X, y_encoded, cv=cv, scoring="accuracy")

print("SVM accuracy promedio:", svm_scores.mean())
print("SVM desviación:", svm_scores.std())

print("XGBoost accuracy promedio:", xgb_scores.mean())
print("XGBoost desviación:", xgb_scores.std())


# ===============================
# GUARDAR MODELOS
# ===============================

joblib.dump(svm_model, r"C:\Users\User\Desktop\EXP11\modelo_svm_robo.pkl")
joblib.dump(xgb_model, r"C:\Users\User\Desktop\EXP11\modelo_xgb_robo.pkl")
joblib.dump(encoder, r"C:\Users\User\Desktop\EXP11\label_encoder.pkl")

print("\nModelos guardados correctamente:")
print("modelo_svm_robo.pkl")
print("modelo_xgb_robo.pkl")
print("label_encoder.pkl")