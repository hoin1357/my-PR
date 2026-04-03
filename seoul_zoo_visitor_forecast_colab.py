"""
서울대공원 일별 방문객 예측용 구글 코랩 실행 코드

사용 방법
1. 이 파일과 `seoul_zoo_daily_visitors_full.csv`를 코랩 작업 폴더에 올립니다.
2. 코랩에서 이 파일 내용을 그대로 실행합니다.
3. 코드가 자동으로 학습 데이터와 정답 데이터를 나누고, 모델을 학습한 뒤,
   실제 방문객 수와 예측 방문객 수를 그래프로 보여 줍니다.
"""

import subprocess
import sys
from pathlib import Path

# 코랩에는 대부분의 라이브러리가 이미 있지만,
# `holidays`는 없는 경우가 있으므로 먼저 설치합니다.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "holidays"])

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. 데이터 파일 이름 설정
# ------------------------------------------------------------
# 코랩 폴더 안에 있는 정제된 전체 데이터 CSV 파일 이름입니다.
# 파일 이름을 바꾸지 않았다면 그대로 두시면 됩니다.
DATA_FILE = "seoul_zoo_daily_visitors_full.csv"


# ------------------------------------------------------------
# 2. 데이터 읽기
# ------------------------------------------------------------
# year, month, day 열을 하나의 실제 날짜(date)로 합칩니다.
data = pd.read_csv(DATA_FILE)
data["date"] = pd.to_datetime(data[["year", "month", "day"]])
data = data.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------
# 3. 최신 3개월을 정답 데이터셋으로 분리할 기준 날짜 만들기
# ------------------------------------------------------------
# 마지막 3개 월을 정답 데이터로 쓰기 위해,
# 데이터에 들어 있는 월 목록 중 가장 마지막 3개 월을 찾습니다.
unique_months = data["date"].dt.to_period("M").drop_duplicates().sort_values()
answer_start = unique_months.iloc[-3].to_timestamp()


# ------------------------------------------------------------
# 4. 기본 결측치 처리
# ------------------------------------------------------------
# visitor_count가 비어 있으면 먼저 같은 연-월 평균으로 채웁니다.
# 그래도 채울 수 없으면 학습 데이터 평균으로 채웁니다.
train_mask_for_target = data["date"] < answer_start
train_mean_for_target = data.loc[train_mask_for_target, "visitor_count"].dropna().mean()
monthly_mean_for_target = data.groupby(data["date"].dt.to_period("M"))["visitor_count"].transform("mean")
data["visitor_count"] = data["visitor_count"].fillna(monthly_mean_for_target)
data["visitor_count"] = data["visitor_count"].fillna(train_mean_for_target)
data["visitor_count"] = data["visitor_count"].round().astype(int)


# ------------------------------------------------------------
# 5. 날짜 기반 특징 만들기
# ------------------------------------------------------------
# 모델이 날짜의 계절성과 요일 패턴을 이해할 수 있게
# 사람이 자주 보는 달력 정보들을 숫자로 바꿉니다.
data["day_of_week"] = data["date"].dt.dayofweek
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
data["quarter"] = data["date"].dt.quarter
data["day_of_year"] = data["date"].dt.dayofyear
data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
data["days_since_start"] = (data["date"] - data["date"].min()).dt.days

# month_sin, month_cos는 "12월 다음이 1월"이라는
# 원형 계절 구조를 모델이 잘 배우게 돕는 값입니다.
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
data["day_of_year_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
data["day_of_year_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)

# 한국 공휴일 여부도 방문객 수에 영향을 줄 수 있으므로 특징으로 추가합니다.
year_range = range(int(data["year"].min()), int(data["year"].max()) + 1)
korean_holidays = holidays.country_holidays("KR", years=year_range)
data["is_holiday"] = data["date"].dt.date.map(lambda current_date: int(current_date in korean_holidays))


# ------------------------------------------------------------
# 6. 과거 방문객 수를 이용한 시계열 특징 만들기
# ------------------------------------------------------------
# lag_1은 "어제 방문객 수", lag_7은 "일주일 전 방문객 수" 같은 의미입니다.
# rolling_mean_7은 "직전 7일 평균 방문객 수"입니다.
# 모두 shift(1)을 써서 미래 정보를 보지 않게 만듭니다.
for lag in [1, 7, 14, 28]:
    data[f"lag_{lag}"] = data["visitor_count"].shift(lag)

# 1년 전 비슷한 시점의 방문객 수도 강한 계절성 힌트가 되므로 함께 넣습니다.
for lag in [364, 365, 371]:
    data[f"lag_{lag}"] = data["visitor_count"].shift(lag)

shifted_target = data["visitor_count"].shift(1)
for window in [7, 14, 28]:
    data[f"rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
    data[f"rolling_std_{window}"] = shifted_target.rolling(window=window).std()


# ------------------------------------------------------------
# 7. 학습용 입력(X)과 정답(y) 만들기
# ------------------------------------------------------------
# 아래 열들이 모델이 보는 입력 특징입니다.
feature_columns = [
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "quarter",
    "day_of_year",
    "week_of_year",
    "days_since_start",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_364",
    "lag_365",
    "lag_371",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
]

train_mask = data["date"] < answer_start
answer_mask = data["date"] >= answer_start

X_train = data.loc[train_mask, feature_columns]
y_train = data.loc[train_mask, "visitor_count"]
X_answer = data.loc[answer_mask, feature_columns]
y_answer = data.loc[answer_mask, "visitor_count"]


# ------------------------------------------------------------
# 8. 특징 결측치 평균 대체
# ------------------------------------------------------------
# lag, rolling 특징은 초반 날짜에서 값이 비게 됩니다.
# 이때는 학습 데이터 평균으로만 채워서 데이터 누수를 막습니다.
imputer = SimpleImputer(strategy="mean")
X_train_filled = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns, index=X_train.index)
X_answer_filled = pd.DataFrame(imputer.transform(X_answer), columns=feature_columns, index=X_answer.index)


# ------------------------------------------------------------
# 9. 모델 만들기
# ------------------------------------------------------------
# HistGradientBoostingRegressor를 쓰는 이유:
# 1) 방문객 수처럼 비선형 패턴이 있는 표 데이터에 강합니다.
# 2) 요일, 계절성, 최근 추세가 섞인 문제에서 성능이 좋습니다.
# 3) 데이터 스케일링이 꼭 필요하지 않아 사용이 쉽습니다.
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_depth=6,
    max_iter=500,
    min_samples_leaf=10,
    l2_regularization=0.1,
    random_state=42,
)


# ------------------------------------------------------------
# 10. 모델 학습
# ------------------------------------------------------------
model.fit(X_train_filled, y_train)


# ------------------------------------------------------------
# 11. 정답 데이터셋 예측
# ------------------------------------------------------------
predictions = model.predict(X_answer_filled)
predictions_rounded = np.round(predictions).astype(int)


# ------------------------------------------------------------
# 12. 평가 지표 계산
# ------------------------------------------------------------
# MAE: 평균적으로 몇 명 정도 틀렸는지
# RMSE: 큰 오차에 더 민감한 평균 오차
# MAPE: 실제 값 대비 몇 퍼센트 정도 틀렸는지
# R²: 전체 패턴을 얼마나 잘 설명하는지
mae = mean_absolute_error(y_answer, predictions_rounded)
rmse = np.sqrt(mean_squared_error(y_answer, predictions_rounded))
mape = np.mean(np.where(y_answer != 0, np.abs((y_answer - predictions_rounded) / y_answer) * 100, 0))
r2 = r2_score(y_answer, predictions_rounded)

print(f"평가 시작 월: {answer_start.strftime('%Y-%m-%d')}")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 13. 실제값과 예측값을 한 표로 정리
# ------------------------------------------------------------
results = data.loc[answer_mask, ["date", "visitor_count"]].copy()
results = results.rename(columns={"visitor_count": "actual"})
results["predicted"] = predictions_rounded
results["absolute_error"] = (results["actual"] - results["predicted"]).abs()
results["error_rate"] = np.where(
    results["actual"] != 0,
    (results["absolute_error"] / results["actual"] * 100).round(2),
    0.0,
)

print("\n정답 데이터셋 예측 결과 일부 미리보기")
print(results.head(20).to_string(index=False))


# ------------------------------------------------------------
# 14. 시각화
# ------------------------------------------------------------
# 첫 번째 그래프: 날짜별 실제값과 예측값 비교
# 두 번째 그래프: 날짜별 절대 오차
# 세 번째 그래프: 실제값-예측값 산점도
fig, axes = plt.subplots(3, 1, figsize=(16, 15))

axes[0].plot(results["date"], results["actual"], label="Actual", linewidth=2)
axes[0].plot(results["date"], results["predicted"], label="Predicted", linewidth=2)
axes[0].set_title("Seoul Zoo Daily Visitors: Actual vs Predicted")
axes[0].set_ylabel("Visitors")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(results["date"], results["absolute_error"], color="tomato", linewidth=2)
axes[1].set_title("Absolute Error by Date")
axes[1].set_ylabel("Absolute Error")
axes[1].grid(alpha=0.3)

axes[2].scatter(results["actual"], results["predicted"], alpha=0.7)
min_value = min(results["actual"].min(), results["predicted"].min())
max_value = max(results["actual"].max(), results["predicted"].max())
axes[2].plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
axes[2].set_title("Actual vs Predicted Scatter")
axes[2].set_xlabel("Actual Visitors")
axes[2].set_ylabel("Predicted Visitors")
axes[2].grid(alpha=0.3)

plt.tight_layout()
try:
    # 코랩 노트북 안에서는 그래프를 바로 화면에 보여 줍니다.
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        get_ipython = None

    if get_ipython is not None and get_ipython() is not None:
        plt.show()
    else:
        output_image_path = Path("seoul_zoo_colab_preview.png")
        plt.savefig(output_image_path, dpi=200, bbox_inches="tight")
        print(f"\n그래프 이미지를 저장했습니다: {output_image_path}")
finally:
    plt.close(fig)


# ------------------------------------------------------------
# 15. 특정 날짜 예측값 조회 예시
# ------------------------------------------------------------
# 아래처럼 원하는 날짜를 넣어서 결과 표에서 찾아볼 수 있습니다.
query_date = "2025-05-05"
query_result = results.loc[results["date"] == pd.Timestamp(query_date)]

print(f"\n조회 날짜: {query_date}")
if query_result.empty:
    print("해당 날짜는 정답 데이터셋 구간에 없습니다.")
else:
    print(query_result.to_string(index=False))
