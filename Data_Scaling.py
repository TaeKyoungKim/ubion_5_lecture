import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataScaler:
    """
    데이터 스케일링 전문 클래스
    (표준화, 정규화, 로그 스케일링 포함)
    """
    def __init__(self, df):
        self.df = df.copy()

    def apply_standard_scaling(self, columns):
        """
        1. 표준화 (Standardization)
        - 평균을 0, 표준편차를 1로 변환
        - 이상치가 어느 정도 있을 때 유용하며, 많은 알고리즘(SVM, 선형 회귀 등)의 기본 가정임
        """
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def apply_minmax_scaling(self, columns):
        """
        2. 정규화 (Normalization / Min-Max Scaling)
        - 데이터를 0과 1 사이의 값으로 변환
        - 데이터의 최소/최대 범위를 알 때 유용하며, 딥러닝(이미지 처리 등)에서 자주 사용
        """
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def apply_log_scaling(self, columns):
        """
        3. 로그 스케일링 (Log Transformation)
        - 데이터의 분포가 한쪽으로 심하게 치우쳤을 때(왜도가 높을 때) 사용
        - 큰 수치를 작게 만들어 분포를 정규분포에 가깝게 만듦
        - 0 또는 음수가 있을 경우를 대비해 log1p(x + 1)를 사용함
        """
        for col in columns:
            self.df[col] = np.log1p(self.df[col])
        return self.df

# --- 실전 적용 예시 (타이타닉 데이터 활용) ---
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv('data/titanic.csv')
    
    # 결측치가 있으면 스케일링이 안 되므로 간단히 채우기
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    scaler_tool = DataScaler(df)

    # (1) Age(나이)에 대해 표준화 적용
    df_standard = scaler_tool.apply_standard_scaling(['Age'])
    
    # (2) Fare(요금)에 대해 로그 스케일링 적용 (요금은 왜도가 매우 높기 때문)
    df_log = scaler_tool.apply_log_scaling(['Fare'])
    
    # (3) 변환된 요금에 대해 다시 정규화(0~1) 적용
    df_final = scaler_tool.apply_minmax_scaling(['Fare'])

    print("스케일링 적용 후 데이터 샘플:")
    print(df_final[['Age', 'Fare']].head())