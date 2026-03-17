import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt

# Step 1: 데이터 불러오기 및 전처리
data = pd.read_csv('imdb_movie_dataset_con.csv')

# 1. 필요한 열 선택 (Votes, Rank, Revenue만 사용)
data = data[['Votes', 'Rank', 'Revenue (Millions)', 'follow-up']]

# 2. 결측치 처리 (Revenue)
data.fillna(data.median(numeric_only=True), inplace=True)

# 3. 독립 변수(X)와 종속 변수(y) 분리
X = data.drop('follow-up', axis=1)
y = data['follow-up']

# 4. 데이터 분리 (훈련, 검증 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 2: 클래스 불균형 처리 및 모델 학습
# 클래스 불균형 계산 및 scale_pos_weight 설정
class_0 = sum(y_train == 0)
class_1 = sum(y_train == 1)
scale_pos_weight = class_0 / class_1
print(f"Scale_Pos_Weight: {scale_pos_weight:.2f}")

# XGBoost 모델 정의
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

# 모델 학습
xgb_model.fit(X_train, y_train)

# Step 3: 모델 평가
# 예측 확률 (Probabilities)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# 예측 결과 (Threshold = 0.5 적용)
y_pred = (y_proba >= 0.5).astype(int)

# 성능 평가
# 1. 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 2. F1 스코어
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# 3. 기타 성능 지표
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 4: ROC 및 PR 곡선
# ROC 곡선 및 AUROC
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

# PR 곡선 및 AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUPRC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Step 5: 새로운 영화 데이터 예측

"""new_movie = pd.DataFrame({      
    'Votes': [300000],            # 영화 [쇼생크 탈출] - 원래 후속작 없음 IMDb 투표 수 
    'Rank': [1],                 # IMDb 순위 
    'Revenue (Millions)': [25]  # 수익 (백만 달러 단위) 
}) """  #새로운 영화 속편 제작 확률: 0.26  새로운 영화 속편 여부 예측 (1 = 제작, 0 = 미제작): 0

new_movie = pd.DataFrame({      
    'Votes': [210000],            # 영화 [대부1] - 후속작 있음 IMDb 투표 수 
    'Rank': [2],                 # IMDb 순위 
    'Revenue (Millions)': [250]  # 수익 (백만 달러 단위) 
}) 

# 예측 확률 계산
new_movie_proba = xgb_model.predict_proba(new_movie)[:, 1]

# 예측 결과 (Threshold = 0.5 적용)
new_movie_prediction = (new_movie_proba >= 0.5).astype(int)

print(f"새로운 영화 속편 제작 확률: {new_movie_proba[0]:.2f}")
print(f"새로운 영화 속편 여부 예측 (1 = 제작, 0 = 미제작): {new_movie_prediction[0]}")
