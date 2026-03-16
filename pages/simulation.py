import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# --- 페이지 설정 ---
st.set_page_config(page_title="이탈 예측 What-If 시뮬레이션", layout="wide")

st.title("📈 What-If 비즈니스 시뮬레이션")
st.markdown("""
**"만약 우리의 정책을 바꾼다면, 이탈률은 어떻게 변할까?"** 학습된 AI 모델을 사용하여 정책 변화에 따른 실제 예상 이탈률 변화를 분석합니다.
""")


# --- 데이터 및 모델 로드 함수 ---
@st.cache_resource
def load_assets():
    # 1. 데이터 로드
    df = pd.read_csv("./data/insurance_policyholder_churn_synthetic.csv")

    # 2. 모델 및 임계값 로드
    model = joblib.load("./model/churn_model_new.pkl")
    threshold = joblib.load("./model/threshold_new.pkl")

    return df, model, threshold


# --- 파생 변수 생성 함수 (EDA 코드에서 추출) ---
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    # 보험료 인상 충격 지표
    result['premium_increase_shock'] = np.clip(result['premium_change_pct'], 0, None) * (
                1 + result['num_price_increases_last_3y'])
    # 급격한 인상 여부 (12% 기준)
    result['premium_jump_flag'] = (result['premium_change_pct'] >= 0.12).astype(int)
    # 결제 위험 점수
    result['payment_risk_score'] = result['late_payment_count_12m'] + result['missed_payment_flag'] * 2
    # 서비스 불만 점수
    result['service_risk_score'] = result['complaint_flag'] + result['quote_requested_flag'] + result[
        'coverage_downgrade_flag']
    # 종합 위험 점수
    result['engagement_risk_score'] = result['payment_risk_score'] + result['service_risk_score']
    # 가입 기간 역수 (단기 고객 강조)
    result['tenure_inverse'] = 1 / (result['customer_tenure_months'] + 1)
    # 단기 및 단일 계약 여부
    result['single_policy_short_tenure'] = (
                (result['multi_policy_flag'] == 0) & (result['customer_tenure_months'] <= 24)).astype(int)
    # 상호작용 변수들
    result['premium_x_complaint'] = result['premium_jump_flag'] * result['complaint_flag']
    result['premium_x_quote'] = result['premium_jump_flag'] * result['quote_requested_flag']
    result['late_x_quote'] = (result['late_payment_count_12m'] >= 2).astype(int) * result['quote_requested_flag']
    result['downgrade_x_quote'] = result['coverage_downgrade_flag'] * result['quote_requested_flag']
    result['monthly_payment_flag'] = (result['payment_frequency'] == 'Monthly').astype(int)
    result['monthly_x_premium_jump'] = result['monthly_payment_flag'] * result['premium_jump_flag']
    result['auto_or_health'] = result['policy_type'].isin(['Auto', 'Health']).astype(int)
    return result


# --- 예측 수행 함수 ---
def run_prediction(df, model, threshold):
    # 학습에 사용되지 않는 컬럼 제거
    drop_cols = ['customer_id', 'as_of_date', 'churn_type', 'churn_probability_true', 'churn_flag']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 파생 변수 추가
    X_fe = add_engineered_features(X)

    # 확률 예측 및 이탈 여부 결정
    probs = model.predict_proba(X_fe)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs.mean(), (preds.mean() * 100), preds.sum()


# 자산 로드
try:
    df_origin, model, threshold = load_assets()
except Exception as e:
    st.error(f"모델 또는 데이터를 불러오는데 실패했습니다. 경로를 확인해주세요: {e}")
    st.stop()

# --- UI 설정 ---
st.markdown("### 🛠️ 시뮬레이션 시나리오 설정")

with st.container(border=True):
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.subheader("시나리오 1: 고객 불만 케어")
        complaint_reduction = st.slider(
            "민원 해결 비율 (%)",
            min_value=0, max_value=100, value=20, step=5,
            help="적극적인 응대로 민원을 제기한 고객의 불만을 해소했을 때의 효과입니다."
        )

    with col_input2:
        st.subheader("시나리오 2: 보험료 인상 관리")
        premium_limit = st.slider(
            "보험료 인상률 상한선 (%)",
            min_value=5, max_value=20, value=12, step=1,
            help="보험료 인상폭을 일정 수준 이하로 강제 조정했을 때의 효과입니다."
        )

    run_sim = st.button("🚀 AI 시뮬레이션 실행", type="primary", use_container_width=True)

# --- 시뮬레이션 실행 로직 ---
if run_sim:
    st.markdown("### 📊 시뮬레이션 결과 (AI 모델 예측)")

    # 1. AS-IS 계산
    as_is_prob, as_is_rate, as_is_count = run_prediction(df_origin, model, threshold)

    # 2. TO-BE 데이터 생성 (시뮬레이션 적용)
    sim_df = df_origin.copy()

    # 정책 1: 민원 해소 (랜덤하게 선택된 비율만큼 민원 flag를 0으로 변경)
    if complaint_reduction > 0:
        complaint_indices = sim_df[sim_df['complaint_flag'] == 1].index
        num_to_reduce = int(len(complaint_indices) * (complaint_reduction / 100))
        if num_to_reduce > 0:
            reduce_idx = np.random.choice(complaint_indices, num_to_reduce, replace=False)
            sim_df.loc[reduce_idx, 'complaint_flag'] = 0

    # 정책 2: 보험료 인상률 제한 (설정한 상한선보다 높은 경우 상한선으로 깎음)
    sim_df['premium_change_pct'] = sim_df['premium_change_pct'].clip(upper=premium_limit / 100)

    # 3. TO-BE 계산
    to_be_prob, to_be_rate, to_be_count = run_prediction(sim_df, model, threshold)

    # 결과 요약 Metric
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="현재 예상 이탈률", value=f"{as_is_rate:.2f}%")
    with col2:
        diff = to_be_rate - as_is_rate
        st.metric(
            label="시뮬레이션 후 이탈률",
            value=f"{to_be_rate:.2f}%",
            delta=f"{diff:.2f}%p",
            delta_color="inverse"
        )
    with col3:
        saved_count = int(as_is_count - to_be_count)
        st.metric(label="이탈 방어 성공 고객 수", value=f"{max(0, saved_count)}명")

    st.markdown("---")

    # 시각화
    st.subheader("💡 정책 도입 전/후 비교")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    # 한글 폰트 설정 (필요시)
    plt.rcParams['font.family'] = 'Malgun Gothic'

    categories = ['현재 상태 (AS-IS)', '정책 적용 후 (TO-BE)']
    rates = [as_is_rate, to_be_rate]
    sns.barplot(x=categories, y=rates, palette=['#FF9999', '#66B2FF'], ax=ax)

    for i, v in enumerate(rates):
        ax.text(i, v + 0.1, f"{v:.2f}%", ha='center', fontweight='bold', size=12)

    ax.set_ylabel("예상 이탈률 (%)")
    ax.set_ylim(0, max(rates) * 1.3)
    st.pyplot(fig)

    if diff < 0:
        st.success(
            f"**분석 결과**: 해당 정책 도입 시 이탈률이 기존 대비 약 **{abs(diff):.2f}%p** 감소하여, 총 **{saved_count}명**의 고객을 더 유지할 수 있을 것으로 예측됩니다.")
    else:
        st.warning("**분석 결과**: 현재 설정한 조건으로는 이탈률에 유의미한 감소가 나타나지 않았습니다. 정책 강도를 조절해보세요.")

else:
    st.info("💡 왼쪽 슬라이더를 통해 시나리오를 설정하고 버튼을 클릭하세요. 실제 학습된 AI 모델이 결과를 계산합니다.")