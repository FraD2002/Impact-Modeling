from pathlib import Path

import numpy as np
import pandas as pd

from project_io import read_required_csvs, write_csv_frames


def filter_high_volatility_days(volatility_df, percentile_threshold=95):
    filtered_days = []
    daily_vol = []
    for i in range(0, len(volatility_df), 195):
        range_volatility = volatility_df.iloc[i:i + 195]
        daily_volatility = range_volatility.mean()
        mean_volatility = daily_volatility.mean()
        daily_vol.append(mean_volatility)

    volatility_threshold = np.percentile(daily_vol, percentile_threshold)
    for i, volatility in enumerate(daily_vol):
        if volatility <= volatility_threshold:
            filtered_days.append(i)
    return filtered_days


def daily_volatility(volatility_df):
    daily_vol = []
    for i in range(0, len(volatility_df), 195):
        range_volatility = volatility_df.iloc[i:i + 195]
        daily_volatility = range_volatility.mean()
        mean_volatility = daily_volatility
        daily_vol.append(mean_volatility)

    return np.array(daily_vol).T


def compute_volatility(midQuoteReturnsArrayDf):
    window_size = int(10 * 6.5 * 30)
    stds = []
    midQuoteReturnsArrayDf.fillna(0, inplace=True)

    for index, row in midQuoteReturnsArrayDf.iterrows():
        rolling_std = row.rolling(window=window_size).std()
        scaled_std = rolling_std * np.sqrt(195)
        stds.append(scaled_std)

    volatility_df = pd.concat(stds, axis=1)
    volatility_df.columns = midQuoteReturnsArrayDf.index
    volatility_df.fillna(0, inplace=True)

    return volatility_df


def main():
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "Data"
    input_dir = project_root / "Input"
    input_dir.mkdir(exist_ok=True)

    required_files = [
        "midQuoteReturnsArrayDf.csv",
        "totalDailyValueDf.csv",
        "imbalanceDf.csv",
        "vwap330Df.csv",
        "vwapCloseDf.csv",
        "arrivalPriceDf.csv",
        "terminalPriceDf.csv",
    ]
    frames = read_required_csvs(data_dir, required_files, index_col=0)
    midQuoteReturnsArrayDf = frames["midQuoteReturnsArrayDf"]
    totalDailyValueDf = frames["totalDailyValueDf"]
    imbalanceDf = frames["imbalanceDf"]
    vwap330Df = frames["vwap330Df"]
    vwapCloseDf = frames["vwapCloseDf"]
    arrivalPriceDf = frames["arrivalPriceDf"]
    terminalPriceDf = frames["terminalPriceDf"]
    volatility_df = compute_volatility(midQuoteReturnsArrayDf)
    filtered_indices = filter_high_volatility_days(volatility_df)
    daily_vol = pd.DataFrame(daily_volatility(volatility_df))

    midQuoteReturnsArrayDf = midQuoteReturnsArrayDf.iloc[:, filtered_indices]
    totalDailyValueDf = totalDailyValueDf.iloc[:, filtered_indices]
    imbalanceDf = imbalanceDf.iloc[:, filtered_indices]
    vwap330Df = vwap330Df.iloc[:, filtered_indices]
    vwapCloseDf = vwapCloseDf.iloc[:, filtered_indices]
    arrivalPriceDf = arrivalPriceDf.iloc[:, filtered_indices]
    terminalPriceDf = terminalPriceDf.iloc[:, filtered_indices]
    daily_vol = daily_vol.iloc[:, filtered_indices]

    write_csv_frames(
        input_dir,
        {
            "midQuoteReturnsArrayDf": midQuoteReturnsArrayDf,
            "totalDailyValueDf": totalDailyValueDf,
            "imbalanceDf": imbalanceDf,
            "vwap330Df": vwap330Df,
            "vwapCloseDf": vwapCloseDf,
            "arrivalPriceDf": arrivalPriceDf,
            "terminalPriceDf": terminalPriceDf,
            "dailyVolDf": daily_vol,
        },
        index_label="Stock",
    )


if __name__ == "__main__":
    main()
