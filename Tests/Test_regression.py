from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import unittest


def load_x_sigma():
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "Input"
    required = [
        input_dir / "totalDailyValueDf.csv",
        input_dir / "imbalanceDf.csv",
        input_dir / "dailyVolDf.csv",
    ]
    if not all(path.exists() for path in required):
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, 10000)
        sigma = np.clip(rng.lognormal(mean=-0.1, sigma=0.3, size=10000), 1e-6, None)
        return x, sigma

    totalDailyValue = pd.read_csv(input_dir / "totalDailyValueDf.csv")
    imbalance = pd.read_csv(input_dir / "imbalanceDf.csv")
    dailyVol = pd.read_csv(input_dir / "dailyVolDf.csv")

    totalDailyValue.drop("Stock", axis=1, inplace=True)
    imbalance.drop("Stock", axis=1, inplace=True)
    dailyVol.drop("Stock", axis=1, inplace=True)

    avgDailyValue = totalDailyValue.T.rolling(window=10, min_periods=10).mean().T.fillna(0.0)

    imbalance = imbalance.iloc[:, 10:]
    dailyVol = dailyVol.iloc[:, 10:]
    avgDailyValue = avgDailyValue.iloc[:, 10:]

    x = np.array((imbalance / (6 * avgDailyValue / 6.5)).stack())
    sigma = np.array(dailyVol.stack())
    return x, sigma


x, sigma = load_x_sigma()
m, std = np.mean(x), np.std(x)


def temp_impact(x,eta,beta):
    temporary_impact = eta * sigma * np.sign(x) * np.power(np.abs(x) , beta)
    return temporary_impact


class Test_Curvefit(unittest.TestCase):

    def testRegression(self):
        #we preset some number for eta and beta
        eta = 0.3
        beta = 1.2
        
        #we check if the recovered parameters are consistent
        eta_test_list = []
        beta_test_list = []
        for i in range(600):
            #simulate a dataset
            x_test = np.random.normal(m,std,size = len(sigma))

            #add some noise
            noise_level = std/3

            #simulate y-values according to the preset eta and beta
            y_test = eta * sigma * np.sign(x_test) * np.power(np.abs(x_test) , beta) + np.random.normal(0,noise_level,size = len(x_test))

            #fit the generated dataset
            popt, pcov = curve_fit(temp_impact, x_test, y_test)
            eta_test_list.append(popt[0])
            beta_test_list.append(popt[1])
        
        eta_test = np.mean(eta_test_list)
        beta_test = np.mean(beta_test_list)

        self.assertAlmostEqual(eta,eta_test, delta=0.01)
        self.assertAlmostEqual(beta,beta_test, delta=0.01)

        print('true eta: %s, recovered eta: %s'%(eta,eta_test))
        print('true beta: %s, recovered beta: %s'%(beta,beta_test))

if __name__ == '__main__':
    unittest.main() 
