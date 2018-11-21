# Copyright 2018 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from moonshot import Moonshot
from moonshot.commission import PerShareCommission

class USStockCommission(PerShareCommission):
    IB_COMMISSION_PER_SHARE = 0.005

class VMOTTrend(Moonshot):
    """
    Hedging strategy that sells the market based on 2 trend rules:
    
    1. Sell 50% if market price is below 12-month moving average
    2. Sell 50% if market 12-month return is below 0
    
    This strategy constitutes the "Trend" portion of the Alpha Architect
    Value/Momentum/Trend (VMOT) ETF.
    """

    CODE = "vmot-trend"
    DB = "spy-1d"
    REBALANCE_INTERVAL = "W"
    COMMISSION_CLASS = USStockCommission

    def prices_to_signals(self, prices):

        closes = prices.loc["Close"]
        
        one_year_returns = (closes - closes.shift(252))/closes.shift(252)
        market_below_zero = one_year_returns < 0        
        
        mavgs = closes.rolling(window=252).mean()
        market_below_mavg = closes < mavgs
        
        hedge_signals = market_below_zero.astype(int) + market_below_mavg.astype(int)
        hedge_signals = -hedge_signals
        
        return hedge_signals
    
    def signals_to_target_weights(self, signals, prices):
        # Resample using the rebalancing interval.
        # Keep only the last signal of the period, then fill it forward
        signals = signals.resample(self.REBALANCE_INTERVAL).last()
        signals = signals.reindex(prices.loc["Close"].index, method="ffill")

        # Divide signal counts by 2 to get the target weights
        weights = signals / 2
        return weights

    def target_weights_to_positions(self, weights, prices):
        # Enter the position the day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions, prices):
        # Enter on the close
        closes = prices.loc["Close"]
        # The return is the security's percent change over the period,
        # multiplied by the position.
        gross_returns = closes.pct_change() * positions.shift()
        return gross_returns

    def order_stubs_to_orders(self, orders, prices):
        orders["Exchange"] = "SMART"
        orders["OrderType"] = "MOC"
        orders["Tif"] = "DAY"
        return orders