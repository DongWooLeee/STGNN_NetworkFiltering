import torch
import torch.nn as nn
import torch
from model import LSTMModel

def train_lstm(train_dataset, valid_dataset, best_gcn, num_epochs, hidden_size, num_classes, seed, seq_length, rebal_term, ref_term, often_freq, PMFG,  patience=10):
    lstm_model = LSTMModel(gcn_model=best_gcn, lstm_hidden_dim=hidden_size, num_classes=num_classes, num_nodes=30)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-5, weight_decay=5e-6)

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(num_epochs):
        lstm_model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        for l in range(len(train_dataset)):
            graph_sequence = train_dataset.get(l)
            labels = graph_sequence[-1].y.cpu().numpy()  # 실제 라벨
            optimizer.zero_grad()
            output = lstm_model(graph_sequence)
            loss = criterion(output, graph_sequence[-1].y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_preds = torch.argmax(output, dim=1).cpu().numpy()  # 예측 라벨
            all_train_preds.extend(train_preds)
            all_train_labels.extend(labels)


        # 검증 루프 및 성능 평가
        lstm_model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        for l in range(len(valid_dataset)):
            graph_sequence = valid_dataset.get(l)
            labels = graph_sequence[-1].y.cpu().numpy()  # 실제 라벨
            with torch.no_grad():
                output = lstm_model(graph_sequence)
                val_loss += criterion(output, graph_sequence[-1].y).item()
                val_preds = torch.argmax(output, dim=1).cpu().numpy()  # 예측 라벨
                all_val_preds.extend(val_preds)
                all_val_labels.extend(labels)
        
        val_loss /= len(valid_dataset)

        print(f'Epoch: {epoch} | Train Loss: {total_loss/len(train_dataset)} | Validation Loss: {val_loss} ')

        # 최상의 모델 가중치 업데이트
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = lstm_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping 체크
        if patience_counter >= patience:
            print(f"Early stopping triggered. Stopping training at epoch {epoch}")
            break

    # 최상의 검증 손실을 갖는 모델의 가중치 저장 및 로드
    if best_model_weights is not None:
        torch.save(best_model_weights, f'./checkpoints/best_lstm_model_{seed}_{seq_length}_{rebal_term}_{ref_term}_{often_freq}_{PMFG}.pth')
        print("Best LSTM model weights saved successfully.")
        
    best_lstm = LSTMModel(gcn_model=best_gcn, lstm_hidden_dim=hidden_size, num_classes=num_classes)
    best_lstm.load_state_dict(torch.load(f'./checkpoints/best_lstm_model_{seed}_{seq_length}_{rebal_term}_{ref_term}_{often_freq}_{PMFG}.pth'))
    best_lstm.eval()
    return best_lstm, all_train_preds, all_train_labels, all_val_preds, all_val_labels