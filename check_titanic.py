import pandas as pd

# 1. 데이터 파일 경로 설정
file_path = "data/titanic.csv"

try:
    # CSV 파일 읽기
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("1. 데이터 기본 정보 (행과 열의 개수)")
    print("=" * 60)
    print(f"총 {df.shape[0]}개의 행(데이터)과 {df.shape[1]}개의 열(변수)로 이루어져 있습니다.\n")

    print("=" * 60)
    print("2. 데이터 구조 및 결측치, 데이터 타입 정보 (info)")
    print("=" * 60)
    # info()는 각 컬럼의 이름, 결측되지 않은(Non-Null) 데이터 수, 데이터 타입(dtype)을 보여줍니다.
    df.info()
    print("\n")

    print("=" * 60)
    print("3. 수치형/범주형 데이터 요약 통계량 (describe)")
    print("=" * 60)
    # 평균, 표준편차, 최소/최댓값 등을 보여주며, include='all'을 통해 범주형 데이터도 포함합니다.
    print(df.describe(include='all'))
    print("\n")

    print("=" * 60)
    print("4. 처음 5줄 데이터 미리보기 (head)")
    print("=" * 60)
    # 데이터가 실제로 어떻게 생겼는지 파악할 때 유용합니다.
    print(df.head())

except FileNotFoundError:
    print(f"❌ 오류: '{file_path}' 경로에서 파일을 찾을 수 없습니다.")
    print("작업 중인 폴더(e:\\apps\\ubion_5_lecture) 안에 data 폴더와 titanic.csv 파일이 있는지 확인해 주세요.")
