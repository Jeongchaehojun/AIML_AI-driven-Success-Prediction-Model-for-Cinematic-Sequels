import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('imdb_movie_dataset_con.csv')

# 데이터 전처리
# 1. 필요한 열 선택 (Votes, Rank, Revenue만 사용)
data = data[['Votes', 'Rank', 'Revenue (Millions)', 'follow-up']]

# 2. 결측치 처리 (Revenue)
data.fillna(data.median(numeric_only=True), inplace=True)

# 3. 독립 변수(X)와 종속 변수(y) 분리
X = data.drop('follow-up', axis=1)
y = data['follow-up']

# 4. 데이터 분리 (훈련, 검증 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 성능 평가
# 1. 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 2. ROC 곡선 및 AUROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 3. PR 곡선 및 AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUPRC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# 4. F1 스코어
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# 5. 기타 성능 지표
print("\nClassification Report:\n", classification_report(y_test, y_pred))
