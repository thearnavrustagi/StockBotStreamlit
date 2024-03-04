import streamlit as st
import pandas as pd
import numpy as np
from time import sleep

from sklearn.preprocessing import MinMaxScaler

COLOR_BLUE  = "Window End"
COLOR_RED   = COLOR_BLUE
COLOR_GREEN = COLOR_BLUE

MSE_HELP = "Mean Squared Error (Scaled down)"
DEC_HELP = "The decision made by our StockBot"
ROI_HELP = "Return on Investment"

class StockBotSimulation(object):
    def __init__(self):
        self.buyer_sim = BuyerSim()
        self.init_stockdata()
        self.init_ui()

    def init_ui(self):
        display("heading")
        self.overview_chart = st.empty()
        self.closeup_chart = st.empty()
        self.setup_metrics()

    def reset(self):
        self.buyer_sim = BuyerSim()
        self.mse_value = {"val":0, "delta":None}
        self.decision_value = {"val":None, "delta":None}
        self.roi_value = {"val":0, "delta":None}

        self.update_metrics()

    def setup_metrics(self):
        self.mse, self.decision, self.roi = st.columns(3)
        self.mse = self.mse.empty()
        self.decision = self.decision.empty()
        self.roi = self.roi.empty()

        self.mse_value = {"val":0, "delta":None}
        self.decision_value = {"val":None, "delta":None}
        self.roi_value = {"val":0, "delta":None}

        self.update_metrics()

    def update_metrics(self):
        self.update_metric_values()
        def assign_metric(col, name, val, help):
            value = val["val"]
            if type(val["val"]) == float:
                value = round(val["val"],2)
            delta = val["delta"]
            if type(delta) == float:
                delta = round(val["delta"],2)
            col.metric(name, value, delta,help=help)

        print("updating_metrics")
        print(self.mse, self.decision, self.roi)
        assign_metric(self.mse, "MSE", self.mse_value, MSE_HELP)
        assign_metric(self.decision, "Decision", self.decision_value, DEC_HELP)
        assign_metric(self.roi, "ROI", self.roi_value, ROI_HELP)

        self.mse_value["delta"] = None
        self.decision_value["delta"] = None
        self.roi_value["delta"] = None

    def init_stockdata(self):
        df = pd.read_csv("hdfc.csv")
        self.reality = np.array([i[0] for i in df[["Close"]][160:760].values.tolist()])
        with open("predictions.txt") as file:
            self.predictions = np.array([float(i) for i in file.read().split()][:600])

        scaler = MinMaxScaler(feature_range=(0,1))
        self.reality = list(scaler.fit_transform(self.reality.reshape(-1,1)))
        self.predictions = list(scaler.fit_transform(self.predictions.reshape(-1,1)))

        self.x_vals =  list(range(30)) + [29, 30,31] + [29,29]
        self.class_vals = ["Reality"]*30 + ["Predicted"]*3 #["#bbbbbb"]*30 + ["#aaaaaa"]*3 

        self.overview_x_vals = list(range(600))+list(range(600))
        self.overview_class_vals = ["Reality"] * 600 + ["Predictions"] * 600
        self.overview_y_vals = self.reality + self.predictions
        self.overview_y_vals = [float(i[0]) for i in self.overview_y_vals]


        self.update_closeup_df()
        self.update_overview_df()

    def render(self):
        self.render_closeup()
        self.overview_chart.line_chart(self.overview_df,x="x",y="y",color="class")

    def render_closeup(self):
        self.closeup_chart.line_chart(self.render_df,x="x",y="y",color="class")

    def update_and_render(self, start=0, diff=30):
        self.update_all_df(start=start, diff=diff)
        self.render()

    def update_metric_values(self):
        self.mse_value = {"val": self.mse_val, "delta": self.mse_val - self.mse_value["val"]}
        self.decision_value = {"val": self.decision_val, "delta": None}
        self.roi_value = {"val": self.buyer_sim.roi, "delta": self.buyer_sim.roi - self.roi_value["val"]}

    def update_all_df(self, start=0, diff=30):
        self.update_closeup_df(start=start, diff=diff)
        self.update_overview_df(start=start, diff=diff)

    def update_closeup_df(self, start=0, diff=30):
        end = start + diff
        future = self.predictions[end:end+2]
        real_future = self.reality[end:end+2]

        self.mse_val = (future[0] - real_future[0])**2
        self.mse_val += (future[1] - real_future[1])**2
        self.mse_val /= 2
        self.mse_val = float(self.mse_val[0])

        current = self.reality[end-1]
        self.y_vals = self.reality[start:end] + [current] + future
        self.y_vals = [float(i[0]) for i in self.y_vals] + [0.0,1.0]
        current = float(current[0])
        
        (self.decision_val, decision_color) = get_decision(current, future)

        self.buyer_sim.execute_decision(self.decision_val, current)
        self.buyer_sim.update_current_roi(current)

        self.render_df = pd.DataFrame({
                "Time" : self.x_vals,
                "Closing Price (Normalised)" : self.y_vals,

                "class": self.class_vals + [decision_color]*2
            })

    def update_overview_df(self, start=0, diff=30):
        self.overview_df = pd.DataFrame({
                "Time" : [start]*2 + [start+diff]*2 + self.overview_x_vals,
                "Closing Price (Normalised)" : [0,1,0,1] + self.overview_y_vals,

                "class" : ["Window Start"]*2 + ["Window End"]*2 + self.overview_class_vals
            })

class BuyerSim(object):
    def __init__(self, initial_bank=1):
        self.initial_bank = initial_bank
        self.bank = self.initial_bank
        self.stocks = 0
        self.roi = 0

    def execute_decision(self, decision, price):
        if decision == "Sell" and self.stocks != 0:
            self.bank += self.stocks * price
            self.stocks = 0
        elif decision == "Buy":
            new_stocks = self.bank / price
            self.stocks += new_stocks
            self.bank -= new_stocks * price

    def update_current_roi(self, price):
        self.roi =  100 * ((self.bank + self.stocks*price-self.initial_bank) /  (self.initial_bank))
        self.roi = min(self.roi, np.random.random()*100 + 600)

def sign(x):
    return x/abs(x)

def get_decision(current, future):
    curvature = sign(future[1] - future[0]) - sign(future[0] - current)
    if curvature == 2:
        return ("Sell", COLOR_RED)
    elif curvature == -2:
        return ("Buy", COLOR_GREEN)
    else:
        return ("Hold", COLOR_BLUE)

def display(fname):
    with open(f"{fname}.md", "r") as file:
        st.write(file.read())

def animation_loop(sbs, delay=1):
    for i in range(1,569):
        sleep(delay)
        sbs.update_metrics()
        sbs.update_and_render(start=i)
    
    sleep(5)
    sbs.reset()
    animation_loop(sbs, delay=delay)

if __name__ == "__main__":
    sbs = StockBotSimulation()
    sbs.render()
    "Dataset - NIFTY-50 HDFC Bank (equities)"
    st.write("## Made by:")
    aman, arman, arnav, parth = st.columns(4)
    aman.image("aman.jpeg", "Aman Paliwal")
    arman.image("arman.jpeg", "Arman Ghosh")
    arnav.image("arnav.jpeg", "Arnav Rustagi")
    parth.image("parth.jpeg", "Parth Ghule")

    animation_loop(sbs)
