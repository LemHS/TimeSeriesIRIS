from Model import cnn_lstm
from train import hypertune

cnnlstm = cnn_lstm.BuildCNNLSTM
ss_cnnlstm_model, ss_cnnlstm_history, ss_cnnlstm_pred, ss_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "SingleShot")
seq2seq_cnnlstm_model, seq2seq_cnnlstm_history, seq2seq_cnnlstm_pred, seq2seq_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "Seq2Seq")
ar_cnnlstm_model, ar_cnnlstm_history, ar_cnnlstm_pred, ar_cnnlstm_actual, test_date = hypertune("MYOR.JK", 0.1, 0.1, "2024-11-1", cnnlstm, "Autoregressive")