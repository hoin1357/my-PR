from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent

PREFERRED_SOURCE_FILES = [
    "서울대공원 시간대별 입장객 현황(2015년~2018년).xlsx",
    "서울대공원 입장객 정보_2018년.xlsx",
    "서울대공원 입장객 정보_2019년.xlsx",
    "서울대공원 입장객 정보_2020년.xlsx",
    "서울대공원 입장객 정보_2021년.xlsx",
    "서울대공원 입장객 정보_2022년.xlsx",
    "서울대공원 입장객 정보_2023년.xlsx",
    "서울대공원 입장객 정보_2024년 12월.xlsx",
    "서울대공원 입장객 정보_2025년 1월~7월.xlsx",
]

FULL_CSV_PATH = BASE_DIR / "seoul_zoo_daily_visitors_full.csv"
TRAIN_CSV_PATH = BASE_DIR / "seoul_zoo_daily_visitors_train.csv"
ANSWER_CSV_PATH = BASE_DIR / "seoul_zoo_daily_visitors_answer.csv"
FULL_XLSX_PATH = BASE_DIR / "seoul_zoo_daily_visitors.xlsx"
PREDICTION_CSV_PATH = BASE_DIR / "seoul_zoo_answer_predictions.csv"
PLOT_PATH = BASE_DIR / "seoul_zoo_prediction_visualization.png"
SUMMARY_PATH = BASE_DIR / "seoul_zoo_model_summary.json"


@dataclass
class DatasetBundle:
    full: pd.DataFrame
    train: pd.DataFrame
    answer: pd.DataFrame
    answer_start: pd.Timestamp
    visitor_imputation_summary: dict[str, int]


def load_source_workbooks(base_dir: Path) -> list[Path]:
    available_files = {path.name: path for path in base_dir.glob("*.xlsx")}
    missing_files = [name for name in PREFERRED_SOURCE_FILES if name not in available_files]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"필요한 원본 엑셀 파일이 없습니다: {missing_text}")
    return [available_files[name] for name in PREFERRED_SOURCE_FILES]


def parse_excel_like_date(value: object) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, (pd.Timestamp,)):
        return pd.Timestamp(value).normalize()

    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return pd.NaT
        # 엑셀의 날짜 시리얼 값은 보통 40000대이므로, 너무 작은 값은 날짜로 보지 않습니다.
        if float(value) < 1000:
            return pd.NaT
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")).normalize()

    text = str(value).strip()
    if not text or text in {"월계", "합계"}:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.normalize()


def parse_numeric(value: object) -> float:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def extract_2015_to_2017(path: Path) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    records: list[dict[str, object]] = []

    for worksheet in workbook.worksheets:
        title = worksheet.title.strip()
        year_text = "".join(ch for ch in title if ch.isdigit())
        if len(year_text) < 4:
            continue

        year = int(year_text[:4])
        if year < 2015 or year > 2017:
            continue

        current_month: int | None = None
        for month_value, day_value, total_value, *_ in worksheet.iter_rows(min_row=4, values_only=True):
            if isinstance(month_value, (int, float, np.integer, np.floating)) and not pd.isna(month_value):
                current_month = int(month_value)

            if current_month is None:
                continue

            if not isinstance(day_value, (int, float, np.integer, np.floating)):
                continue

            try:
                current_day = int(day_value)
                current_total = parse_numeric(total_value)
                current_date = pd.Timestamp(year=year, month=current_month, day=current_day)
            except (TypeError, ValueError):
                continue

            records.append(
                {
                    "date": current_date,
                    "visitor_count": current_total,
                    "source_file": path.name,
                    "source_sheet": title,
                }
            )

    return records


def find_daily_header(worksheet) -> tuple[int, int, int]:
    header_rows: list[list[str]] = []

    for row_index, row in enumerate(worksheet.iter_rows(min_row=1, max_row=20, values_only=True), start=1):
        normalized_values = [str(value).strip() if value is not None else "" for value in row]
        compact_values = [value.replace(" ", "") for value in normalized_values]
        header_rows.append(compact_values)

        if "날짜" not in compact_values:
            continue

        date_col = compact_values.index("날짜")
        rows_to_search = [compact_values]
        if len(header_rows) >= 2:
            rows_to_search.append(header_rows[-2])

        for candidate_row in rows_to_search:
            if "총계" in candidate_row:
                return row_index, date_col, candidate_row.index("총계")
            if "일합계" in candidate_row:
                return row_index, date_col, candidate_row.index("일합계")

    raise ValueError(f"헤더를 찾지 못했습니다: {worksheet.title}")


def extract_daily_workbook(path: Path) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    records: list[dict[str, object]] = []

    for worksheet in workbook.worksheets:
        header_row, date_col, visitor_col = find_daily_header(worksheet)

        for row in worksheet.iter_rows(min_row=header_row + 1, values_only=True):
            date_value = row[date_col] if len(row) > date_col else None
            visitor_value = row[visitor_col] if len(row) > visitor_col else None

            if isinstance(date_value, str) and date_value.strip() == "월계":
                continue

            current_date = parse_excel_like_date(date_value)
            if pd.isna(current_date):
                continue

            records.append(
                {
                    "date": current_date,
                    "visitor_count": parse_numeric(visitor_value),
                    "source_file": path.name,
                    "source_sheet": worksheet.title.strip(),
                }
            )

    return records


def build_raw_dataset(source_files: Iterable[Path]) -> pd.DataFrame:
    all_records: list[dict[str, object]] = []

    for path in source_files:
        if "시간대별" in path.name:
            all_records.extend(extract_2015_to_2017(path))
        else:
            all_records.extend(extract_daily_workbook(path))

    dataset = pd.DataFrame(all_records)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values("date").drop_duplicates(subset="date", keep="first").reset_index(drop=True)
    return dataset


def get_answer_start_date(dataset: pd.DataFrame) -> pd.Timestamp:
    unique_months = dataset["date"].dt.to_period("M").drop_duplicates().sort_values()
    if len(unique_months) < 3:
        raise ValueError("최신 3개월 평가를 하기에 데이터 개월 수가 부족합니다.")
    return unique_months.iloc[-3].to_timestamp()


def fill_missing_visitor_count(dataset: pd.DataFrame, answer_start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    filled = dataset.copy()
    train_mask = filled["date"] < answer_start

    raw_null_mask = filled["visitor_count"].isna()
    monthly_means = filled.groupby(filled["date"].dt.to_period("M"))["visitor_count"].transform("mean")
    train_mean = filled.loc[train_mask, "visitor_count"].dropna().mean()

    monthly_fill_mask = raw_null_mask & monthly_means.notna()
    filled.loc[monthly_fill_mask, "visitor_count"] = monthly_means.loc[monthly_fill_mask]

    global_fill_mask = filled["visitor_count"].isna()
    filled.loc[global_fill_mask, "visitor_count"] = train_mean

    filled["visitor_count"] = filled["visitor_count"].round().astype(int)

    summary = {
        "visitor_count_null_before_fill": int(raw_null_mask.sum()),
        "visitor_count_filled_with_month_mean": int(monthly_fill_mask.sum()),
        "visitor_count_filled_with_train_mean": int(global_fill_mask.sum()),
    }
    return filled, summary


def export_ready_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    exported = dataset.copy()
    exported["year"] = exported["date"].dt.year
    exported["month"] = exported["date"].dt.month
    exported["day"] = exported["date"].dt.day
    return exported[["year", "month", "day", "visitor_count"]].reset_index(drop=True)


def split_dataset(dataset: pd.DataFrame, answer_start: pd.Timestamp, visitor_summary: dict[str, int]) -> DatasetBundle:
    export_df = export_ready_dataset(dataset)
    answer_mask = dataset["date"] >= answer_start
    train_df = export_df.loc[~answer_mask].reset_index(drop=True)
    answer_df = export_df.loc[answer_mask].reset_index(drop=True)
    return DatasetBundle(
        full=export_df,
        train=train_df,
        answer=answer_df,
        answer_start=answer_start,
        visitor_imputation_summary=visitor_summary,
    )


def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    featured = dataset.copy().sort_values("date").reset_index(drop=True)

    featured["year"] = featured["date"].dt.year
    featured["month"] = featured["date"].dt.month
    featured["day"] = featured["date"].dt.day
    featured["day_of_week"] = featured["date"].dt.dayofweek
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    featured["quarter"] = featured["date"].dt.quarter
    featured["day_of_year"] = featured["date"].dt.dayofyear
    featured["week_of_year"] = featured["date"].dt.isocalendar().week.astype(int)
    featured["days_since_start"] = (featured["date"] - featured["date"].min()).dt.days
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    featured["day_of_year_sin"] = np.sin(2 * np.pi * featured["day_of_year"] / 365.25)
    featured["day_of_year_cos"] = np.cos(2 * np.pi * featured["day_of_year"] / 365.25)

    year_range = range(int(featured["year"].min()), int(featured["year"].max()) + 1)
    korean_holidays = holidays.country_holidays("KR", years=year_range)
    featured["is_holiday"] = featured["date"].dt.date.map(lambda current_date: int(current_date in korean_holidays))

    for lag in [1, 7, 14, 28]:
        featured[f"lag_{lag}"] = featured["visitor_count"].shift(lag)

    # 1년 전 비슷한 시점의 방문객 수를 같이 주면 강한 계절성을 더 잘 배울 수 있습니다.
    for lag in [364, 365, 371]:
        featured[f"lag_{lag}"] = featured["visitor_count"].shift(lag)

    shifted_values = featured["visitor_count"].shift(1)
    for window in [7, 14, 28]:
        featured[f"rolling_mean_{window}"] = shifted_values.rolling(window=window).mean()
        featured[f"rolling_std_{window}"] = shifted_values.rolling(window=window).std()

    return featured


def build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42,
    )


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    absolute_percentage_error = np.where(actual != 0, np.abs((actual - predicted) / actual) * 100, 0)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(absolute_percentage_error)),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(dataset: pd.DataFrame, answer_start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, float], dict[str, int]]:
    featured = add_features(dataset)
    answer_mask = featured["date"] >= answer_start

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

    x_train = featured.loc[~answer_mask, feature_columns]
    y_train = featured.loc[~answer_mask, "visitor_count"]
    x_answer = featured.loc[answer_mask, feature_columns]
    y_answer = featured.loc[answer_mask, "visitor_count"]

    imputer = SimpleImputer(strategy="mean")
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=feature_columns, index=x_train.index)
    x_answer_imputed = pd.DataFrame(imputer.transform(x_answer), columns=feature_columns, index=x_answer.index)

    feature_null_summary = {
        "train_feature_null_before_fill": int(x_train.isna().sum().sum()),
        "answer_feature_null_before_fill": int(x_answer.isna().sum().sum()),
        "train_feature_null_after_fill": int(x_train_imputed.isna().sum().sum()),
        "answer_feature_null_after_fill": int(x_answer_imputed.isna().sum().sum()),
    }

    model = build_model()
    model.fit(x_train_imputed, y_train)
    predictions = model.predict(x_answer_imputed)

    metrics = calculate_metrics(y_answer, predictions)
    result = featured.loc[answer_mask, ["date", "visitor_count"]].copy()
    result["predicted"] = np.round(predictions).astype(int)
    result["absolute_error"] = (result["visitor_count"] - result["predicted"]).abs()
    result["error_rate"] = np.where(
        result["visitor_count"] != 0,
        (result["absolute_error"] / result["visitor_count"] * 100).round(2),
        0.0,
    )
    result = result.rename(columns={"visitor_count": "actual"}).reset_index(drop=True)
    return result, metrics, feature_null_summary


def save_exports(bundle: DatasetBundle) -> None:
    bundle.full.to_csv(FULL_CSV_PATH, index=False, encoding="utf-8-sig")
    bundle.train.to_csv(TRAIN_CSV_PATH, index=False, encoding="utf-8-sig")
    bundle.answer.to_csv(ANSWER_CSV_PATH, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(FULL_XLSX_PATH, engine="openpyxl") as writer:
        bundle.full.to_excel(writer, sheet_name="full", index=False)
        bundle.train.to_excel(writer, sheet_name="train", index=False)
        bundle.answer.to_excel(writer, sheet_name="answer", index=False)


def save_prediction_outputs(prediction_df: pd.DataFrame, metrics: dict[str, float]) -> None:
    prediction_df.to_csv(PREDICTION_CSV_PATH, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].plot(prediction_df["date"], prediction_df["actual"], label="Actual", linewidth=2)
    axes[0].plot(prediction_df["date"], prediction_df["predicted"], label="Predicted", linewidth=2)
    axes[0].set_title("Seoul Zoo Daily Visitors: Actual vs Predicted")
    axes[0].set_ylabel("Visitors")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(prediction_df["date"], prediction_df["absolute_error"], color="tomato", linewidth=2)
    axes[1].set_title("Absolute Error by Date")
    axes[1].set_ylabel("Absolute Error")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(prediction_df["actual"], prediction_df["predicted"], alpha=0.7)
    min_value = min(prediction_df["actual"].min(), prediction_df["predicted"].min())
    max_value = max(prediction_df["actual"].max(), prediction_df["predicted"].max())
    axes[2].plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    axes[2].set_title(
        "Actual vs Predicted Scatter"
        f" | MAE={metrics['mae']:.1f}, RMSE={metrics['rmse']:.1f}, MAPE={metrics['mape']:.2f}%"
    )
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Predicted")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_summary(
    raw_dataset: pd.DataFrame,
    bundle: DatasetBundle,
    prediction_df: pd.DataFrame,
    metrics: dict[str, float],
    feature_null_summary: dict[str, int],
) -> dict[str, object]:
    sample_checks = {
        "2015-01-01": 1645,
        "2018-01-01": 2316,
        "2021-01-01": 633,
        "2025-01-01": 2396,
    }

    sample_results: list[dict[str, object]] = []
    for date_text, expected_value in sample_checks.items():
        actual_series = raw_dataset.loc[raw_dataset["date"] == pd.Timestamp(date_text), "visitor_count"]
        actual_value = None if actual_series.empty else int(round(actual_series.iloc[0]))
        sample_results.append(
            {
                "date": date_text,
                "expected": expected_value,
                "actual": actual_value,
                "matched": actual_value == expected_value,
            }
        )

    summary = {
        "date_range": {
            "min_date": raw_dataset["date"].min().strftime("%Y-%m-%d"),
            "max_date": raw_dataset["date"].max().strftime("%Y-%m-%d"),
            "answer_start": bundle.answer_start.strftime("%Y-%m-%d"),
        },
        "row_counts": {
            "full_rows": int(len(bundle.full)),
            "train_rows": int(len(bundle.train)),
            "answer_rows": int(len(bundle.answer)),
        },
        "visitor_imputation_summary": bundle.visitor_imputation_summary,
        "feature_null_summary": feature_null_summary,
        "metrics": metrics,
        "sample_checks": sample_results,
        "prediction_preview": prediction_df.head(10).assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d")).to_dict(
            orient="records"
        ),
    }
    return summary


def main() -> None:
    source_files = load_source_workbooks(BASE_DIR)
    raw_dataset = build_raw_dataset(source_files)

    answer_start = get_answer_start_date(raw_dataset)
    filled_dataset, visitor_summary = fill_missing_visitor_count(raw_dataset, answer_start)
    bundle = split_dataset(filled_dataset, answer_start, visitor_summary)
    save_exports(bundle)

    prediction_df, metrics, feature_null_summary = train_and_evaluate(filled_dataset, answer_start)
    save_prediction_outputs(prediction_df, metrics)

    summary = build_summary(filled_dataset, bundle, prediction_df, metrics, feature_null_summary)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("정제 데이터와 예측 결과 파일을 생성했습니다.")
    print(f"- 전체 데이터 CSV: {FULL_CSV_PATH.name}")
    print(f"- 학습 데이터 CSV: {TRAIN_CSV_PATH.name}")
    print(f"- 정답 데이터 CSV: {ANSWER_CSV_PATH.name}")
    print(f"- 정제 데이터 엑셀: {FULL_XLSX_PATH.name}")
    print(f"- 예측 결과 CSV: {PREDICTION_CSV_PATH.name}")
    print(f"- 시각화 이미지: {PLOT_PATH.name}")
    print(f"- 요약 파일: {SUMMARY_PATH.name}")
    print(
        "평가 지표:",
        f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%, R2={metrics['r2']:.4f}",
    )


if __name__ == "__main__":
    main()
