import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from project_io import ensure_required_directories, read_nonempty_lines, write_csv_frames
from impactUtils.VWAP import VWAP
from Preprocessing.ArrivalPrice import getArrivalPrice
from Preprocessing.DailyValue import getDailyValue
from Preprocessing.Imbalance import getImbalance
from Preprocessing.MidQuoteReturns import getMidQuoteReturns
from Preprocessing.TerminalPrice import getTerminalPrice
from taq.MyDirectories import MyDirectories
from taq.TAQQuotesReader import TAQQuotesReader
from taq.TAQTradesReader import TAQTradesReader


def main():
    startTS = 18 * 60 * 60 * 1000 / 2
    impactTS = startTS + (6 * 60 * 60 * 1000)
    endTS = startTS + (13 * 60 * 60 * 1000 / 2)
    numBuckets = int(math.ceil((endTS - startTS) / 120000))

    trade_directory = MyDirectories.getTradesDir()
    quote_directory = MyDirectories.getQuotesDir()

    ensure_required_directories([quote_directory, trade_directory])

    sp500_stocks = read_nonempty_lines("SP500.txt")

    quote_days = [
        quote_day
        for quote_day in os.listdir(quote_directory)
        if os.path.isdir(os.path.join(quote_directory, quote_day))
    ]
    if not quote_days:
        raise ValueError(f"No quote day folders found in {quote_directory}")

    num_stocks = len(sp500_stocks)
    num_days = len(quote_days)

    midQuoteReturnsArray = []
    totalDailyValue = []
    imbalance = []
    vwap330 = []
    vwapClose = []
    arrivalPrice = []
    terminalPrice = []

    for quote_day in quote_days:
        quote_day_path = os.path.join(quote_directory, quote_day)
        mqr_stock = []
        tdv_stock = []
        imb_stock = []
        vwap330_stock = []
        vwapClose_stock = []
        arrival_stock = []
        terminal_stock = []

        for stock_symbol in sp500_stocks:
            quote_path = os.path.join(quote_day_path, f"{stock_symbol}_quotes.binRQ")
            trade_path = os.path.join(trade_directory, quote_day, f"{stock_symbol}_trades.binRT")
            if not (os.path.exists(quote_path) and os.path.exists(trade_path)):
                mqr_stock.append(np.full(numBuckets, np.nan))
                tdv_stock.append(np.nan)
                imb_stock.append(np.nan)
                vwap330_stock.append(np.nan)
                vwapClose_stock.append(np.nan)
                arrival_stock.append(np.nan)
                terminal_stock.append(np.nan)
                continue

            trades = TAQTradesReader(trade_path)
            quotes = TAQQuotesReader(quote_path)

            mqr_stock.append(getMidQuoteReturns(quotes, startTS, endTS, numBuckets))
            tdv_stock.append(getDailyValue(trades))
            imb_stock.append(getImbalance(trades, startTS, impactTS))
            vwap330_stock.append(VWAP(trades, startTS, impactTS).getVWAP())
            vwapClose_stock.append(VWAP(trades, startTS, endTS).getVWAP())
            arrival_stock.append(getArrivalPrice(trades, startTS, endTS, numBuckets))
            terminal_stock.append(getTerminalPrice(trades, startTS, endTS, numBuckets))

        midQuoteReturnsArray.append(mqr_stock)
        totalDailyValue.append(tdv_stock)
        imbalance.append(imb_stock)
        vwap330.append(vwap330_stock)
        vwapClose.append(vwapClose_stock)
        arrivalPrice.append(arrival_stock)
        terminalPrice.append(terminal_stock)

    concatenated_array = np.concatenate(np.array(midQuoteReturnsArray, dtype=float), axis=1)
    midQuoteReturnsArray = np.reshape(concatenated_array, (concatenated_array.shape[1], -1)).T
    totalDailyValue = np.array(totalDailyValue, dtype=float).T
    imbalance = np.array(imbalance, dtype=float).T
    vwap330 = np.array(vwap330, dtype=float).T
    vwapClose = np.array(vwapClose, dtype=float).T
    arrivalPrice = np.array(arrivalPrice, dtype=float).T
    terminalPrice = np.array(terminalPrice, dtype=float).T

    print("midQuoteReturnsArray:", midQuoteReturnsArray.shape)
    print("Total Daily Value:", totalDailyValue.shape)
    print("Imbalance:", imbalance.shape)
    print("VWAP Impact:", vwap330.shape)
    print("VWAP Close:", vwapClose.shape)
    print("Arrival Price:", arrivalPrice.shape)
    print("Terminal Price:", terminalPrice.shape)

    data_dir = Path("Data")
    data_dir.mkdir(exist_ok=True)

    midQuoteReturnsArrayDf = pd.DataFrame(midQuoteReturnsArray, index=sp500_stocks)
    totalDailyValueDf = pd.DataFrame(totalDailyValue, index=sp500_stocks)
    imbalanceDf = pd.DataFrame(imbalance, index=sp500_stocks)
    vwap330Df = pd.DataFrame(vwap330, index=sp500_stocks)
    vwapCloseDf = pd.DataFrame(vwapClose, index=sp500_stocks)
    arrivalPriceDf = pd.DataFrame(arrivalPrice, index=sp500_stocks)
    terminalPriceDf = pd.DataFrame(terminalPrice, index=sp500_stocks)

    write_csv_frames(
        data_dir,
        {
            "midQuoteReturnsArrayDf": midQuoteReturnsArrayDf,
            "totalDailyValueDf": totalDailyValueDf,
            "imbalanceDf": imbalanceDf,
            "vwap330Df": vwap330Df,
            "vwapCloseDf": vwapCloseDf,
            "arrivalPriceDf": arrivalPriceDf,
            "terminalPriceDf": terminalPriceDf,
        },
        index_label="Stock",
    )


if __name__ == "__main__":
    main()
