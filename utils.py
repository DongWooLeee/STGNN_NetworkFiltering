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

# load the data


def load_data():
    # 영업일에만 존재하는 데이터
    df_open = pd.read_csv('open.csv',index_col=0, thousands=',')
    df_open.index = pd.to_datetime(df_open.index)
    df_high = pd.read_csv('high.csv',index_col=0, thousands=',')
    df_high.index = pd.to_datetime(df_high.index)
    df_low = pd.read_csv('low.csv',index_col=0, thousands=',')
    df_low.index = pd.to_datetime(df_low.index)
    df_close = pd.read_csv('close.csv',index_col=0, thousands=',')
    df_close.index = pd.to_datetime(df_close.index)
    df_vol = pd.read_csv('volume.csv',index_col=0, thousands=',')
    df_vol.index = pd.to_datetime(df_vol.index)
    # 아래 시가총액 데이터는 매일 존재함
    df_markets = pd.read_csv('market_equity_kospi.csv', thousands = ',', index_col=0) #이는 매일 존재
    
    df_index = pd.read_csv('index.csv',index_col=0, thousands=',')
    df_index.index = pd.to_datetime(df_index.index)
    
    df_index = df_index.loc[:,['KOSPI']]
    
    
    
    
    latest_index_index = max(df_index.index)
    latest_close_index = max(df_close.index)
    

    if latest_index_index > latest_close_index:
        df_index = df_index.loc[:latest_close_index]
    else:
        df_close = df_close.loc[:latest_index_index]

    
   
    return df_open, df_high, df_low, df_close, df_vol, df_index, df_markets

#utils.py


def select_stocks(df_list, train_start_idx, test_start_idx, candidate_num, final_num):
    '''
    리밸런싱 처음 시작하는 날 기준으로 상위 50개 종목을 선택하는 함수. 정적인 자산풀 운용을 위해 미리 뽑아 놓는다.
    '''
    df_open, df_high, df_low, df_close, df_vol, _, df_market = df_list
    
    # 리밸런싱을 시작하는 날짜 기준으로 상위 candidate_num개의 종목을 선택
    selected_stocks = df_market.iloc[test_start_idx].sort_values(ascending=False)[:candidate_num].index.tolist()
    
    valid_stocks = []
    
    for stock in selected_stocks:
        close_prices = df_close[stock].iloc[train_start_idx//2:test_start_idx+1]
        vol_data = df_vol[stock].iloc[train_start_idx//2:test_start_idx+1]
        
        if (df_open[stock].iloc[train_start_idx//2:test_start_idx+1].notnull().all() and
            df_high[stock].iloc[train_start_idx//2:test_start_idx+1].notnull().all() and
            df_low[stock].iloc[train_start_idx//2:test_start_idx+1].notnull().all() and
            close_prices.notnull().all() and
            vol_data.notnull().all() and vol_data.gt(0).all()):  # 거래량이 0보다 큰 경우만 확인
            
            
            valid_stocks.append(stock)
    
    selected_stocks_final = valid_stocks[:final_num]  # 상위 final_num개만큼 주식을 최종 선택
    
    return selected_stocks_final








def get_network_PMFG(corr_matrix):
    n = len(corr_matrix)
    # 상관관계의 절대값을 취하지 않고, 실제 값을 사용합니다.
    rholist = [[corr_matrix[i][j], i, j] for i in range(n) for j in range(i+1, n) if corr_matrix[i][j] != 0]
    # 상관관계 값을 기준으로 정렬합니다. 이 때 절대값이 아닌 실제 값으로 정렬합니다.
    rholist.sort(key=lambda x: x[0], reverse=True)

    G = nx.Graph()
    for rho, i, j in tqdm(rholist):
        G.add_edge(i, j, weight=np.abs(rho))
        # 그래프의 planarity를 체크하고, 최대 엣지 수를 초과하지 않는지 확인합니다.
        if not planarity.is_planar(G) or G.number_of_edges() > 3 * (n - 2) - 1:
            G.remove_edge(i, j)
    return G




def mst_graph(corr_matrix):
    G = nx.Graph()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            # 상관관계가 음수인 경우 절댓값을 취함
            abs_corr = abs(corr_matrix[i][j])
            origin_corr = corr_matrix[i][j]
            # 거리 계산
            distance = np.sqrt(2 * (1 - origin_corr))
            # 거리를 기반으로 간선 추가, 상관관계 절댓값을 가중치로 사용
            G.add_edge(i, j, weight=distance, edge_weight=abs_corr)
    # 거리를 기준으로 최소 신장 트리 생성
    T = nx.minimum_spanning_tree(G, weight='edge_weight')
    return T


def make_corr(dataframe):
    corr_df = dataframe.corr()
    corr_matrix = corr_df.values
    return corr_matrix


def nx_to_pyg(G): #should get a networkx graph as input
    # NetworkX 그래프를 PyTorch Geometric 데이터로 변환
    x = torch.tensor([[G.nodes[node]['open'],G.nodes[node]['open_last'], 
                       G.nodes[node]['high'], G.nodes[node]['high_last'], 
                       G.nodes[node]['low'], G.nodes[node]['low_last'],
                       G.nodes[node]['close'], G.nodes[node]['close_last'],
                       G.nodes[node]['volume'], G.nodes[node]['volume_last']] for node in G.nodes], dtype=torch.float)
        
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    y = torch.tensor([G.nodes[node]['label'] for node in G.nodes], dtype=torch.long) #GCN 시에는 모든 시점에 대해서 label이 존재해야 함. LSTM 시에서는 마지막 시점에 대해서만 label이 존재해야 함.
    # if edge_weights is nan, replace it with 0
    edge_weights = np.nan_to_num(edge_weights)
    data = Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_weights, dtype=torch.float), y=y) #일단 y 넣어줌. lstm에서는 마지막 sequence의 label만을 사용한다. 나머지는 사용하지 않음
    return data




def make_graph_dataset(dataset):
    # train_dataset의 모든 graph sequence에 담긴 그래프를 모두 가져온다.
    graphs = []
    for i in range(len(dataset)):
        graph = dataset.get(i)[-1] # GCN 훈련시킬 그래프들.
        graphs.append(graph)
    return graphs

def nan_value_checker(dataset):
    for l in range(len(dataset)):
        graphs = dataset.get(l)
   
    for idx in range(len(graphs)):
        if torch.isnan(graphs[idx].x).any():
            print(idx+1,'x'*(idx+1),l)
        if torch.isnan(graphs[idx].edge_index).any():
            print(idx,'edge_index'*idx)
        if torch.isnan(graphs[idx].edge_attr).any():
            print(idx,'edge_attr'*idx)
        if torch.isnan(graphs[idx].y).any():
            print(idx,'y'*idx)



def standard_scale_feature(dataframe, start_date, end_date):
    # 각 주식의 특성을 최솟값 -1, 최댓값 1로 스케일링하는 함수
    scaled_df = dataframe.copy()
    for stock in scaled_df.columns:
        # nan 값을 제외하고 최솟값과 최댓값 계산
        
        valid_values = scaled_df.loc[start_date:end_date,stock].dropna()    
        mean_val = valid_values.mean()
        std_val = valid_values.std()
        
        # 평균과 표준편차를 이용하여 스케일링
        scaled_values = (scaled_df[stock] - mean_val) / std_val
        scaled_df[stock] = scaled_values
    
        
    return scaled_df


from sklearn.metrics import accuracy_score, f1_score


"""     def test_lstm(best_lstm_model, test_dataset):
        best_lstm_model.eval()  # 모델을 평가 모드로 설정
        test_preds, test_labels = [], []
        results = []
        
        for l in range(len(test_dataset)):
            graph_sequence = test_dataset.get(l)
            labels = graph_sequence[-1].y.cpu().numpy()  # 실제 라벨
            
            with torch.no_grad():
                output = best_lstm_model(graph_sequence)
                softmax_output = torch.softmax(output, dim=1)
                preds = torch.argmax(softmax_output, dim=1).cpu().numpy()  # 예측 라벨

                # 결과를 결과 리스트에 추가
                real_date_idx = test_dataset.rebal_idx_list[l]
                real_date = test_dataset.df_open.index[real_date_idx]
                for idx, pred in enumerate(preds):
                    results.append({'date': real_date, 'index': idx, 'probability': softmax_output[idx][1].item(), 'prediction': pred})

            test_preds.extend(preds)
            test_labels.extend(labels)

        # 데이터프레임 생성 및 CSV로 저장
        result_df = pd.DataFrame(results)
        
        accuracy = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        
        return result_df, accuracy, f1  """


def test_lstm(best_lstm_model, test_dataset):
    best_lstm_model.eval()  # 모델을 평가 모드로 설정
    test_preds, test_labels = [], []
    results = []
    
    
    for l in range(len(test_dataset)):
        graph_sequence = test_dataset.get(l)
        labels = graph_sequence[-1].y.cpu().numpy()  # 실제 라벨
        
        with torch.no_grad():
            output = best_lstm_model(graph_sequence)
            softmax_output = torch.softmax(output, dim=1)
            preds = torch.argmax(softmax_output, dim=1).cpu().numpy()  # 예측 라벨

            one_preds = []
            zero_preds = []
            for idx, pred in enumerate(preds):
                if pred == 1:
                    one_preds.append((idx, softmax_output[idx][1].item(), pred))
                else:
                    zero_preds.append((idx, softmax_output[idx][0].item(), pred))

            # 0으로 예측된 데이터 중에서 확률이 높은 상위 N개 선택
            top_zero_preds = sorted(zero_preds, key=lambda x: x[1], reverse=True)[:len(one_preds)]

            # 선택된 데이터를 결과 리스트에 추가
            selected_data = one_preds + top_zero_preds
            
            #map the l with the rebal_idx_list
            
            real_date_idx = test_dataset.rebal_idx_list[l]
            
            real_date = test_dataset.df_open.index[real_date_idx]
        
            
            for data in selected_data:
                results.append({'date': real_date, 'index': data[0], 'probability': data[1], 'prediction': data[2]})

        test_preds.extend(preds)
        test_labels.extend(labels)

    # 데이터프레임 생성 및 CSV로 저장
    result_df = pd.DataFrame(results)
    
    
    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    
    return result_df, accuracy, f1 