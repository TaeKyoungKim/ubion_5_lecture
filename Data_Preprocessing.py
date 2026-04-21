import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

class DataPreprocessor:
    """
    데이터 전처리의 핵심 방법론을 모아놓은 전문가용 클래스
    """
    def __init__(self, df):
        self.df = df.copy()

    def handle_missing_values(self, column, strategy='mean'):
        """
        1. 결측치 처리 (Imputation)
        - strategy: 'mean', 'median', 'most_frequent', 'constant'
        """
        imputer = SimpleImputer(strategy=strategy)
        self.df[column] = imputer.fit_transform(self.df[[column]])
        return self.df

    def remove_outliers_iqr(self, column):
        """
        2. 이상치 제거 (Outlier Removal)
        - IQR(Interquartile Range) 방식을 사용하여 극단치 제거
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self.df

    def scale_numeric_data(self, columns, method='standard'):
        """
        3. 데이터 스케일링 (Scaling)
        - standard: 표준화 (평균 0, 분산 1)
        - minmax: 정규화 (0~1 사이 값)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
            
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def encode_categorical(self, column, method='label'):
        """
        4. 범주형 데이터 인코딩 (Encoding)
        - label: 라벨 인코딩 (0, 1, 2...)
        - onehot: 원-핫 인코딩 (Dummy variables)
        """
        if method == 'label':
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
        elif method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=[column], drop_first=True)
        return self.df

    def create_bins(self, column, bins, labels):
        """
        5. 데이터 구간화 (Binning)
        - 수치형 데이터를 범주형(Low, Mid, High 등)으로 변환
        """
        self.df[f'{column}_binned'] = pd.cut(self.df[column], bins=bins, labels=labels)
        return self.df

# --- 실전 적용 예시 (타이타닉 데이터 활용) ---
if __name__ == "__main__":
    # 데이터 로드
    raw_df = pd.read_csv('data/titanic.csv')
    preprocessor = DataPreprocessor(raw_df)

    # (1) 결측치 처리: 나이(Age)는 중앙값으로, 승선항(Embarked)은 최빈값으로
    preprocessor.handle_missing_values('Age', strategy='median')
    preprocessor.handle_missing_values('Embarked', strategy='most_frequent')

    # (2) 이상치 제거: 요금(Fare)에서 너무 큰 값 제거
    preprocessor.remove_outliers_iqr('Fare')

    # (3) 인코딩: 성별(Sex)은 라벨 인코딩
    preprocessor.encode_categorical('Sex', method='label')

    # (4) 스케일링: 요금(Fare) 정규화
    preprocessor.scale_numeric_data(['Fare'], method='minmax')

    # 최종 결과 확인
    processed_df = preprocessor.df
    print("전처리 완료된 데이터 샘플:")
    print(processed_df.head())
    
    # 파일로 저장
    processed_df.to_csv('data/titanic_preprocessed.csv', index=False)