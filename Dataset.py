from torch_geometric.data import Dataset
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import networkx as nx
import numpy as np
import planarity
from tqdm import tqdm

import networkx as nx
import torch

from utils import get_network_PMFG, make_corr, nx_to_pyg, normalize_data


# Train을 위한 것. Frequency는 조금 더 많은 데이터를 학습하기 위함임-> Train용, Validation은 반드시 often_freq를 rebal_term과 같게 해야함.
class GraphCustomDataset(Dataset):
    def __init__(self, df_list, start_date, end_date, seq_length, rebal_term, ref_term, often=True, often_freq = 5):
        self.df_open, self.df_high, self.df_low, self.df_close, self.df_vol, self.df_markets = df_list
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.rebal_term = rebal_term
        self.ref_term = ref_term
        self.rebal_idx_list = []
        self.G_list = []
        self.label_list = []
        self._indices = []
        self.often = often # validation 뽑을 때는 이를 False로 놓고 뽑아야 함.
        self.often_freq = often_freq
        self.make_rebal_idx_list()
        self.stocks_list = []

    def make_rebal_idx_list(self):
        if not isinstance(self.start_date, pd.Timestamp):
            self.start_date = pd.to_datetime(self.start_date)
        if not isinstance(self.end_date, pd.Timestamp):
            self.end_date = pd.to_datetime(self.end_date)
        
        start_idx = self.df_open.index.get_loc(self.start_date)
        end_idx = self.df_open.index.get_loc(self.end_date)
        self.rebal_idx_list = list(range(start_idx, end_idx, self.often_freq)) #미리 리밸런싱 날짜를 저장해둔다.

    def select_stocks(self, rebal_idx):    
        #check if rebal_date is in a datetime format
        selected_stocks = self.df_markets.iloc[rebal_idx].sort_values(ascending=False)[:50].index.tolist()
        
        valid_stocks = []
        for stock in selected_stocks: #seq_term 일수만큼 그래프가 생성 + 이전 리밸런싱 기간동안 데이터가 존재, 각 seq에서 그래프 만들때 참조하는 일수만큼 모두 데이터 존재
            if (self.df_open[stock].iloc[rebal_idx-self.rebal_term-self.seq_length-self.ref_term:rebal_idx+1].notnull().all() and #20일치 데이터가 모두 존재하는 경우만 -> open
                self.df_high[stock].iloc[rebal_idx-self.rebal_term-self.seq_length-self.ref_term:rebal_idx+1].notnull().all() and #20일치 데이터가 모두 존재하는 경우만 -> high
                self.df_low[stock].iloc[rebal_idx-self.rebal_term-self.seq_length-self.ref_term:rebal_idx+1].notnull().all() and #20일치 데이터가 모두 존재하는 경우만 -> low
                self.df_close[stock].iloc[rebal_idx-self.rebal_term-self.seq_length-self.ref_term:rebal_idx+1].notnull().all() and #20일치 데이터가 모두 존재하는 경우만 -> close
                self.df_vol[stock].iloc[rebal_idx-self.rebal_term-self.seq_length-self.ref_term:rebal_idx+1].notnull().all()): #20일치 데이터가 모두 존재하는 경우만 -> volume
                valid_stocks.append(stock) # 모두 만족하는 경우에 대해서만 valid_stocks에 추가
        selected_stocks_final = valid_stocks[:30]  # 상위 30개 주식을 최종 선택
        return selected_stocks_final

    def make_labels(self, selected_stocks_final, rebal_idx):
        future_prices = self.df_close[selected_stocks_final].iloc[rebal_idx+self.rebal_term].values
        current_prices = self.df_close[selected_stocks_final].iloc[rebal_idx].values
        labels = [1 if future > current else 0 for future, current in zip(future_prices, current_prices)]
        return labels

    def make_graph(self, selected_stocks_final, ref_idx):
        input_df = self.df_close[selected_stocks_final].iloc[ref_idx-self.ref_term:ref_idx+1]
        corr_matrix = make_corr(input_df)
        G = get_network_PMFG(corr_matrix)
        return G


    def prepare_data(self):
        idx = 0  # Initialize idx outside the loop
        for rebal_idx in self.rebal_idx_list:
            selected_stocks_final = self.select_stocks(rebal_idx)
            labels = self.make_labels(selected_stocks_final, rebal_idx)
            graph_sequences = []
            for past_days in reversed(range(self.ref_term)): #sequence를 담아야 하므로 역순으로 간다. iloc이므로 정수인덱싱. 따라서 
                G = self.make_graph(selected_stocks_final, rebal_idx-past_days)
                for node in G.nodes:
                    open_price = self.df_open[selected_stocks_final[node]].iloc[rebal_idx-past_days]
                    high_price = self.df_high[selected_stocks_final[node]].iloc[rebal_idx-past_days]
                    low_price = self.df_low[selected_stocks_final[node]].iloc[rebal_idx-past_days]
                    close_price = self.df_close[selected_stocks_final[node]].iloc[rebal_idx-past_days]
                    volume = self.df_vol[selected_stocks_final[node]].iloc[rebal_idx-past_days]

                    G.nodes[node]['open'] = open_price
                    G.nodes[node]['high'] = high_price
                    G.nodes[node]['low'] = low_price
                    G.nodes[node]['close'] = close_price
                    G.nodes[node]['volume'] = volume

                G = nx_to_pyg(G)
                graph_sequences.append(G)

            self.G_list.append(graph_sequences)
            self.label_list.append(labels)
            self._indices.append(idx)
            self.stocks_list.append(selected_stocks_final)
            idx += 1  # Increment idx
    
    def transform(self, data):
        # Transform the data
        return data
        
    def len(self):
        return len(self.rebal_idx_list)
    
    def get(self, idx):
        return self.G_list[idx], self.label_list[idx]