from Model import vanilla_lstm
from train import hypertune

vilstm = vanilla_lstm.BuildVanillaLSTM
ss_vilstm_model, ss_vilstm_history, ss_vilstm_pred, ss_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "SingleShot")
seq2seq_vilstm_model, seq2seq_vilstm_history, seq2seq_vilstm_pred, seq2seq_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "Seq2Seq")
ar_vilstm_model, ar_vilstm_history, ar_vilstm_pred, ar_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "Autoregressive")