from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 붓꽃(Iris) 데이터셋 불러오기
iris = load_iris()
print(iris)
# 2. 피처 이름 및 레이블(타겟) 이름 확인하기
print("=" * 50)
print("▶ 피처 이름 (Feature Names):")
print(iris.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print("\n▶ 레이블 이름 (Target Names):")
print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
print("=" * 50 + "\n")

# 3. 데이터를 한눈에 보기 편하게 Pandas DataFrame으로 변환
# iris.data에는 피처 데이터가, iris.feature_names에는 열 이름이 들어있습니다.
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# iris.target에는 0, 1, 2 형태의 숫자형 정답(레이블)이 들어있습니다.
df['target'] = iris.target

# 0, 1, 2 숫자를 실제 이름으로 바꿔서 보기 쉽게 추가합니다.
df['target_name'] = df['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

# 4. 데이터셋 샘플 출력
print("=== 붓꽃 데이터셋 샘플 (상위 5개) ===")
print(df.head())
print("\n")

print("=== 붓꽃 데이터셋 샘플 (하위 5개) ===")
print(df.tail())
print("\n")

print("=== 타겟(레이블)별 데이터 개수 확인 ===")
print(df['target_name'].value_counts())

# 5. 데이터 전처리: 학습용(Train)과 테스트용(Test) 데이터 분할
print("\n" + "=" * 50)
print("=== 5. 데이터 분할 (Train/Test Split: 8 대 2) ===")

# X: 피처(Feature, 예측을 위한 단서), y: 레이블(Target, 맞춰야 할 정답)
X = iris.data
y = iris.target

# test_size=0.2 : 20%를 테스트용, 나머지 80%를 학습용으로 나눔
# random_state=42 : 매번 실행할 때마다 동일하게 분할되도록 시드값 부여
# stratify=y : 붓꽃의 3가지 품종(0,1,2) 비율이 쏠리지 않도록 8:2로 골고루 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"▶ 원본 데이터 크기 \t: 피처 {X.shape}, 레이블 {y.shape}")
print(f"▶ 학습용(Train) 데이터 \t: 피처 {X_train.shape}, 레이블 {y_train.shape}")
print(f"▶ 테스트용(Test) 데이터 \t: 피처 {X_test.shape}, 레이블 {y_test.shape}")
print("=" * 50 + "\n")
