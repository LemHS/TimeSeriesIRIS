from Model import bilstm, cnn_lstm, gru_lstm, vanilla_lstm
from train import train_model, evaluate_model, hypertune, plot_models

vilstm = vanilla_lstm.BuildVanillaLSTM
ss_vilstm_model, ss_vilstm_history, ss_vilstm_pred, ss_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "SingleShot")
seq2seq_vilstm_model, seq2seq_vilstm_history, seq2seq_vilstm_pred, seq2seq_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "Seq2Seq")
ar_vilstm_model, ar_vilstm_history, ar_vilstm_pred, ar_vilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", vilstm, "Autoregressive")

grulstm = gru_lstm.BuildGRULSTM
ss_grulstm_model, ss_grulstm_history, ss_grulstm_pred, ss_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "SingleShot")
seq2seq_grulstm_model, seq2seq_grulstm_history, seq2seq_grulstm_pred, seq2seq_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "Seq2Seq")
ar_grulstm_model, ar_grulstm_history, ar_grulstm_pred, ar_grulstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", grulstm, "Autoregressive")

cnnlstm = cnn_lstm.BuildCNNLSTM
ss_cnnlstm_model, ss_cnnlstm_history, ss_cnnlstm_pred, ss_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "SingleShot")
seq2seq_cnnlstm_model, seq2seq_cnnlstm_history, seq2seq_cnnlstm_pred, seq2seq_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "Seq2Seq")
ar_cnnlstm_model, ar_cnnlstm_history, ar_cnnlstm_pred, ar_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "Autoregressive")

bi_lstm = bilstm.BuildBiLSTM
ss_bilstm_model, ss_bilstm_history, ss_bilstm_pred, ss_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "SingleShot")
seq2seq_bilstm_model, seq2seq_bilstm_history, seq2seq_bilstm_pred, seq2seq_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "Seq2Seq")
ar_bilstm_model, ar_bilstm_history, ar_bilstm_pred, ar_bilstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", bi_lstm, "Autoregressive")