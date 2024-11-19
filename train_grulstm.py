from Model import gru_lstm
from train import hypertune

grulstm = gru_lstm.BuildGRULSTM
ss_grulstm_model, ss_grulstm_history, ss_grulstm_pred, ss_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "SingleShot")
seq2seq_grulstm_model, seq2seq_grulstm_history, seq2seq_grulstm_pred, seq2seq_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "Seq2Seq")
ar_grulstm_model, ar_grulstm_history, ar_grulstm_pred, ar_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "Autoregressive")