import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import os

# @st.cache_resource
def load_data(type:int=None):
    """
    데이터를 공통으로 로드하는 함수
    :return:
    """
    # 데이터 파일명
    data_file_name = 'insurance_policyholder_churn_synthetic.csv'
    try:
        base_path = Path(__file__).resolve().parent.parent  # src의 상위인 Project 폴더
    except NameError:
        base_path = Path(os.getcwd())  # 주피터 환경 대응

    data_path = base_path / "data" / data_file_name

    df = pd.read_csv(data_path)
    if type is None:    # 기본으로 컬럼만 삭제
        df = common_preprocess_churn_data(df)
    elif type == 1:     # 컬럼 많이 삭제 & 범주화 컬럼 변경
        df = common_preprocess_churn_data(df)
        df = common_preprocess_drop_dupl_cols(df)
        df = common_process_categorical_features(df)
    return df

def common_preprocess_churn_data(df):
    """
    공통적으로 처리할 공통 전처리 함수
    1. 학습에 사용하지 않는 변수 제거
    :param df:
    :return:
    """
    drop_cols = [
        'customer_id',              # 사용자 ID
        'region_name',              # 지역명
        'as_of_date',               # 기준일자
        'renewal_month',            # 갱신월
        'payment_frequency',        # 납입 주기
        'autopay_enabled',          # 자동 이체 여부
        'churn_type',               # 이탈 원인 분류
        # 'churn_flag',               # 이탈 여부 # 정답을 위해 살려 놓음.
        'churn_probability_true',   # 실제 이탈 확률
    ]

    # 컬럼 drop
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df_clean

def common_preprocess_drop_dupl_cols(df):
    # 내용이 중복되는 컬럼
    dupl_cols = [
        'age_band',                 # 나이 밴디지(age 컬럼과 중복) age 컬럼을 살릴지, age_band를 살릴지.
        'premium_last_year',        # 작년 보험료
        'missed_payment_flag',      # 'late_payment_count_12m'와 중복 4회 이상
        'coverage_amount',          # 보장 금액
        # 'current_premium',
        'num_approved_claims_12m',  # num_claims_12m 에 포함된 내용이고, 승인된 클레임 수
        # 'num_rejected_claims_12m',  # num_claims_12m 에 포함된 내용이고, 거절 클레임 수
                                    # 이 컬럼은 전체 중 거절 클레임 수 -> 비율 파생변수 만들 생각으로 살려둠.
        'num_pending_claims_12m',   # num_claims_12m 에 포함된 내용이고, 지연된 클레임 수
        #####
        'num_policies',             # lockin_analysis참고 multi_policy_flag 변수가 더 중요한 것으로 생각
        'complaint_resolution_days',# 'complaint_flag'와 중복
        'total_claim_amount_12m',   # 'total_payout_amount_12m'와 중복 # 둘 중 삭제 뭐할지 파생변수 만들지 생각 안 해봄
    ]

    # 컬럼 drop
    df_clean = df.drop(columns=[col for col in dupl_cols if col in df.columns])

    return df_clean

def common_process_categorical_features(df):
    # 범주형 컬럼
    cat_cols = ['age_band', 'marital_status', 'policy_type']    # age_band 포함

    # 라벨 인코더
    le = LabelEncoder()

    for col in cat_cols:
        # 데이터 프레임에 실제 컬럼이 있는지 확인 후 변환
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(df[col].dtype)

    return df