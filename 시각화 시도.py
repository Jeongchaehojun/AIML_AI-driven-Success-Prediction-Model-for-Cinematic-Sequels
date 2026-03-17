import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('imdb_movie_dataset_con.csv')

# 필요한 열 선택
data = data[['Votes', 'Rank', 'Revenue (Millions)', 'follow-up']]

# 결측치 처리
data.fillna(data.median(numeric_only=True), inplace=True)

# 1. 영화 수익과 투표 수의 관계
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Votes', y='Revenue (Millions)', hue='follow-up', data=data)
plt.title('Revenue vs Votes')
plt.xlabel('Votes (in thousands)')
plt.ylabel('Revenue (in millions)')
plt.legend(title='Follow-Up')
plt.show()

# 2. 속편 여부에 따른 영화 특성 비교
plt.figure(figsize=(10, 6))
sns.boxplot(x='follow-up', y='Revenue (Millions)', data=data)
plt.title('Revenue Distribution by Follow-Up')
plt.xlabel('Follow-Up (0 = No, 1 = Yes)')
plt.ylabel('Revenue (in millions)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='follow-up', y='Votes', data=data)
plt.title('Votes Distribution by Follow-Up')
plt.xlabel('Follow-Up (0 = No, 1 = Yes)')
plt.ylabel('Votes')
plt.show()

# 3. 속편 제작 영화의 수익 분포
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Revenue (Millions)', hue='follow-up', bins=30, kde=True, element='step')
plt.title('Revenue Distribution by Follow-Up')
plt.xlabel('Revenue (in millions)')
plt.show()

# 4. 수익 상위 10개의 영화의 속편 여부
top_10_revenue = data.nlargest(10, 'Revenue (Millions)')
plt.figure(figsize=(10, 6))
sns.barplot(x='Revenue (Millions)', y='Votes', hue='follow-up', data=top_10_revenue, dodge=False)
plt.title('Top 10 Revenue Movies and Follow-Up')
plt.xlabel('Revenue (in millions)')
plt.ylabel('Votes')
plt.show()

# 5. 전체 데이터의 상관관계 히트맵
correlation = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()
