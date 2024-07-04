from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import talib as ta
import matplotlib.pyplot as plt
import math

with open('95fund.txt', 'r') as f:
    kw_list = f.read().split("\n")

integer_list = [int(x) for x in kw_list]

def get_stock_data_from_file(code):   # csvファイルからデータを取得する関数（第1回講義で説明済）
    data_f = pd.read_csv("trade_data/{}.T.csv".format(code), index_col = 0)
    data_f.index = pd.to_datetime(data_f.index)
    return data_f

def F_MA(close):  # 移動平均線の傾きを算出
    w_width = 3
    slopes = []
    for i in range(len(close)):
        h_data = [ x + 1 for x in range(w_width)]  # h_data = [1, 2, 3, ..., w_width]
        if i < w_width:
            v_data = [ 10 for _ in range(w_width)]  # 最初の部分はダミーデータ
        else:
            v_data = close[i - w_width: i]
        slope, _ = np.polyfit(h_data, v_data, 1) # 回帰直線の傾きを計算
        slopes.append(slope)

    r_series = np.array(slopes)
    return r_series

def BB(close, n, nu, nd):
    b_upper, _, b_lower = ta.BBANDS(close, timeperiod=n, nbdevup=nu, nbdevdn=nd)
    return b_upper, b_lower

def MACD(close, n1, n2, n3):
    macd, macdsignal, _ = ta.MACD(close, fastperiod=n1, slowperiod=n2, signalperiod=n3)
    return macd, macdsignal

class bb(Strategy): # 独自の売買戦略
    n = 60   # 移動平均線日数
    n1 = 5   # 短期日数
    n2 = 20  # 長期日数
    n3 = 9   # シグナル日数
    nb=25
    fm75=2#1to10
    upper_sigma = 2  # ボリンジャーバンド+2σ
    lower_sigma = 2
    slinit = 95
    tpinit =105
    def init(self):
        close_prices = self.data["Close"].astype(float)
        self.sma60 = self.I(SMA, self.data["Close"], self.n)
        self.f_ma75 = self.I(F_MA, self.sma60)
        self.macd, self.macdsignal = self.I(MACD, close_prices, 10, self.n2, self.n3)  # MACD
        self.b_upper, self.b_lower = self.I(BB, close_prices, self.nb, self.upper_sigma, self.lower_sigma)  # BB
        self.fm75=self.fm75
        self.slinit=self.slinit
        self.tpinit=self.tpinit

  
    def next(self):
        if crossover(self.macd, self.macdsignal):     # -2σ > 終値 で買う
            price = self.data["Close"]
            self.buy(
                                sl=price * (0.01*self.slinit),
                                tp=price * (0.01*self.tpinit)
                            )
   
s_code=6920
# df = get_stock_data_from_file(s_code)
df = get_stock_data_from_file(s_code)



data = df[dt.datetime(2013,1,1):dt.datetime(2022,12,31)]

bt = Backtest(data, bb, trade_on_close=True)
result=bt.optimize(slinit = range(90, 100, 1), tpinit = range(101, 111, 1), maximize = "Return [%]")
# result = bt.run()  # バックテストを実行

rt = result["Return [%]"]
print(rt)
print(result["# Trades"])
print(result)  # 実行結果のデータを表示
bt.plot()

###############################################
# trd =0
# sum_trd = 0
# codes = integer_list
# sum_r = 0
# sum_tp=0
# sum_sl=0
# win_r = 0
# br_code = []
# for s_code in codes:
#   df = get_stock_data_from_file(s_code)

#   data = df[dt.datetime(2013,1,1):dt.datetime(2022,12,31)]  #10年間分のデータを使用

#   bt = Backtest(data, bb, trade_on_close=True)
#   result=bt.optimize(slinit = range(90, 100, 1), tpinit = range(101, 111, 1), maximize = "Return [%]")

# #   result = bt.run()  # バックテストを実行
#   rt = result["Return [%]"]
#   win=result["Win Rate [%]"]
#   trd= result["# Trades"]
#   if rt == 0:
#       br_code.append(s_code)
#   print(f'Return ({s_code}): {rt:.5f}')
# #   print(result["# Trades"])

#   sum_r = sum_r + rt
#   sum_trd=sum_trd+trd
#   sum_tp=sum_tp+result._strategy.tpinit
#   sum_sl=sum_sl+result._strategy.slinit
#   if math.isnan(win):
#     win_r = win_r + 0
#   else:
#     win_r = win_r + win
# ave_r = sum_r / len(codes)
# ave_tp=sum_tp / len(codes)
# ave_sl=sum_sl / len(codes)
# ave_win = win_r / len(codes)
# ave_trd = sum_trd / len(codes)
# print(f'Average of returns: {ave_r:.5f}')
# print(f'Average of tp: {ave_tp:.5f}',f'Average of sl: {ave_sl:.5f}')
# print(f'Average of win: {ave_win}')
# print(f'Average of trade:{ave_trd}')
# plt.figure()
# plt.bar(np.array([x for x in range(1,len(res)+1)]),np.array(res), align="center")
# plt.savefig("macd_fma_saiteki.png")
# plt.figure()
# plt.bar(np.array([x for x in range(1,len(win_arr)+1)]),np.array(win_arr), align="center")
# plt.savefig("macd_fma_saiteki_win.png")
# plt.show()
       ####################################     
# res =[]
# trd =[]
# win_arr = []
# codes = integer_list
# sum_r = 0
# win_r = 0
# br_code = []
# for s_code in codes:
#   df = get_stock_data_from_file(s_code)

#   data = df[dt.datetime(2013,1,1):dt.datetime(2022,12,31)]  #10年間分のデータを使用

#   bt = Backtest(data, bb, trade_on_close=True)
#   result = bt.run()

# #   result = bt.run()  # バックテストを実行
#   rt = result["Return [%]"]
#   win=result["Win Rate [%]"]
#   if rt ==0:
#       br_code.append(s_code)
#   print(f'Return ({s_code}): {rt:.5f}')
#   print(result["# Trades"])
#   print(win)
#   res.append(rt)
#   trd.append(result["# Trades"])
#   win_arr.append(win)
#   sum_r = sum_r + rt
#   if math.isnan(win):
#     win_r = win_r + 0
#   else:
#     win_r = win_r + win
      


# ave_r = sum_r / len(codes)
# ave_win = win_r / len(codes)
# print(f'Average of returns: {ave_r:.5f}')
# print(f'Average of win: {ave_win}')
# print(br_code)
# plt.figure() 
# plt.bar(np.array([x for x in range(1,len(res)+1)]),np.array(res), align="center")
# plt.savefig("macdonly_not_saiteki.png")
# plt.figure() 
# plt.bar(np.array([x for x in range(1,len(win_arr)+1)]),np.array(win_arr), align="center")
# plt.savefig("macdonly_not_saiteki_win.png")
# plt.show()