import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 데이터 로드 및 전처리 구조
# ==========================================
# 분석에 사용되는 변수 정의:
# - 독립변수 (X): studytime (주당 공부시간), absences (결석 횟수), G1 (1학기 성적)
# - 종속변수 (y): G3 (최종 성적)
# (실제 데이터프레임 X, y를 인자로 받아 함수가 실행됩니다)


# ==========================================
# 2. OLS 정규방정식 수동 구현 및 단일 예측
# ==========================================
def run_manual_ols(X, y):
    X_values = X.values
    y_values = y.values

    # 절편(Intercept) 결합을 위한 상수항(1) 열 추가
    X_matrix = np.c_[np.ones((len(X_values), 1)), X_values]

    # 정규방정식 수동 연산: beta = (X^T * X)^(-1) * X^T * y
    XT_X = X_matrix.T.dot(X_matrix)
    beta = np.linalg.inv(XT_X).dot(X_matrix.T).dot(y_values)

    # 잔차(Residual) 및 오차 표준편차 산출 (시험 당일 불확실성 노이즈용)
    all_predictions = X_matrix.dot(beta)
    residuals = y_values - all_predictions
    error_std = np.std(residuals)

    return beta, error_std, X_matrix


# ==========================================
# 3. 솔레스키 분해(Cholesky Decomposition) 수동 구현
# ==========================================
def manual_cholesky(A):
    """라이브러리를 사용하지 않고 공분산 행렬을 하삼각행렬 L로 수동 분해"""
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(max(0, A[i][i] - sum_k))
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]
    return L


# ==========================================
# 4. 몬테카를로 시뮬레이션 루프 (10,000회)
# ==========================================
def run_monte_carlo_simulation(X, y, beta, error_std, sim_count=10000):
    # 고위험군 학생 시나리오 설정
    scenario_studytime = 1.5
    scenario_absences = 30
    scenario_G1 = 11

    # 변수 간 공분산 반영을 위한 솔레스키 분해 적용
    cov_matrix = X.cov().values
    L_matrix = manual_cholesky(cov_matrix)

    # 표준정규난수 생성 후 공분산 구조 주입 (다변량 정규분포 시뮬레이션)
    np.random.seed(42)
    Z = np.random.normal(0, 1, (3, sim_count))
    scenario_mean = np.array(
        [scenario_studytime, scenario_absences, scenario_G1]
    ).reshape(3, 1)

    virtual_students = scenario_mean + L_matrix.dot(Z)
    virtual_students = virtual_students.T

    # 가상 학생들의 선형 예측 점수 계산
    virt_X_matrix = np.c_[np.ones((sim_count, 1)), virtual_students]
    predicted_G3 = virt_X_matrix.dot(beta)

    # 시험 당일의 환경적 불확실성(오차 노이즈) 결합 및 점수 범위 클리핑(0~20점)
    final_noise = np.random.normal(0, error_std, sim_count)
    simulated_scores = predicted_G3 + final_noise
    simulated_scores = np.clip(simulated_scores, 0, 20)

    return simulated_scores


# ==========================================
# 5. 과락 확률 계산 및 대수의 법칙 검증
# ==========================================
def calculate_failure_risk(simulated_scores, fail_threshold=8.0):
    sim_count = len(simulated_scores)

    # 과락 기준(8점 미만) 판정
    failures = (simulated_scores < fail_threshold).astype(int)
    cumulative_failures = np.cumsum(failures)
    n_trials = np.arange(1, sim_count + 1)

    # 누적 확률 계산 (대수의 법칙 수렴 과정 추적)
    convergence_probs = (cumulative_failures / n_trials) * 100
    
    return convergence_probs