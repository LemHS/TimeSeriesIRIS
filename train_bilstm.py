from Model import bilstm
from train import hypertune

bi_lstm = bilstm.BuildBiLSTM
ss_bilstm_model, ss_bilstm_history, ss_bilstm_pred, ss_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "SingleShot")
seq2seq_bilstm_model, seq2seq_bilstm_history, seq2seq_bilstm_pred, seq2seq_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "Seq2Seq")
ar_bilstm_model, ar_bilstm_history, ar_bilstm_pred, ar_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "Autoregressive")