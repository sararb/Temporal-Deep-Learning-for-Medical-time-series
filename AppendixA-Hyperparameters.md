
# Best Hyperparameters results

## LSTM model 


 |  |lstm-time_concat_soft |lstm-no_time |lstm-time_concat_mlp |lstm-time_mask |
 |--- | --- | --- | --- | --- | 
 | model_type | lstm | lstm | lstm | lstm | 
 | min_measurement | 3 | 3 | 3 | 3 | 
 | cycle_len | 15 | 15 | 30 | 20 | 
 | batch_size | 4 | 12 | 20 | 32 | 
 | class_weight | 0.55 | 0.55 | 0.45 | 0.4 | 
 | optimizer | radam | adam | adam | adam | 
 | max_lr | 0.0008929 | 0.001375 | 0.000599 | 0.0007358 | 
 | weight_decay | 6.62E-05 | 7.58E-05 | 1.24E-06 | 7.30E-05 | 
 | patient_model-hidden_dims | 28 | 4 | 24 | 24 | 
 | patient_model-output_dim | 16 | 56 | 40 | 56 | 
 | patient_model-activation | gelu | relu | gelu | tanh | 
 | time_model-hidden_dims | None | None | 24 | None | 
 | time_model-projection_size | 24 | None | None | 28 | 
 | time_model-output_dim | 16 | 1 | 40 | None | 
 | event_model-continuous_hidden_dims | 20 | 32 | 28 | 8 | 
 | event_model-continuous_output_dim | 56 | 16 | 48 | 56 | 
 | event_model-tf_activation | relu | relu | gelu | gelu | 
 | temporal_model-hidden_size | 64 | 56 | 16 | 64 | 
 | temporal_model-num_layers | 3 | 2 | 1 | 1 | 
 | temporal_model-dropout | 0.5 | 0.1 | 0.1 | 0.5 | 
 | temporal_model-hidden_act | relu | tanh | relu | gelu | 
 | temporal_model-hidden_dropout_prob | 0.1 | 0.2 | 0.3 | 0.3 | 
 | classifier_model-hidden_dim | 64 | 32 | 64 | 24 | 
 | classifier_model-output_dropout | 0.5 | 0.1 | 0.4 | 0.3 | 


## C-LSTM model 

 |  |clstm-forget_output |clstm-output |clstm-forget |
 |--- | --- | --- | --- | 
 | model_type | clstm | clstm | clstm | 
 | min_measurement | 3 | 3 | 3 | 
 | cycle_len | 40 | 20 | 15 | 
 | batch_size | 8 | 16 | 32 | 
 | class_weight | 0.55 | 0.55 | 0.6 | 
 | optimizer | adam | adam | sgd | 
 | max_lr | 0.006557 | 0.003244 | 0.009452 | 
 | weight_decay | 3.45E-06 | 1.60E-05 | 2.99E-05 | 
 | patient_model-hidden_dims | 24 | 28 | 4 | 
 | patient_model-output_dim | 16 | 32 | 24 | 
 | patient_model-activation | relu | relu | gelu | 
 | time_model-output_dim | 1 | 1 | 1 | 
 | event_model-continuous_hidden_dims | 24 | 28 | 8 | 
 | event_model-continuous_output_dim | 56 | 64 | 16 | 
 | event_model-tf_activation | tanh | gelu | tanh | 
 | temporal_model-hidden_size | 8 | 48 | 56 | 
 | temporal_model-timedecay_size | 5 | 1 | 2 | 
 | temporal_model-dropout | 0.4 | 0.4 | 0.4 | 
 | temporal_model-hidden_act | tanh | tanh | tanh | 
 | classifier_model-hidden_dim | 32 | 24 | 40 | 
 | classifier_model-output_dropout | 0.3 | 0.1 | 0.2 | 



## Attention (Transformer-based) model 

 |  |attention-time_encode |attention-time_concat_soft |attention-no_time |attention-time_concat_mlp |attention-time_mask |
 |--- | --- | --- | --- | --- | --- | 
 | name | time_encode | time_concat_soft | no_time | time_concat_mlp | time_mask | 
 | model_type | attention | attention | attention | attention | attention | 
 | min_measurement | 3 | 3 | 3 | 3 | 3 | 
 | cycle_len | 15 | 15 | 30 | 35 | 5 | 
 | batch_size | 16 | 12 | 32 | 4 | 4 | 
 | class_weight | 0.7 | 0.4 | 0.5 | 0.35 | 0.65 | 
 | optimizer | adam | radam | sgd | adam | radam | 
 | max_lr | 0.001972 | 0.001261 | 0.000539 | 0.0001125 | 0.0008968 | 
 | weight_decay | 1.70E-06 | 7.75E-06 | 1.57E-06 | 3.23E-05 | 1.53E-06 | 
 | patient_model-hidden_dims | 16 | 32 | 24 | 16 | 12 | 
 | patient_model-output_dim | None | 56 | 40 | 32 | 32 | 
 | patient_model-activation | relu | gelu | relu | relu | tanh | 
 | time_model-hidden_dims | None | None | None | 16 | None | 
 | time_model-projection_size | None | 12 | None | None | 4 | 
 | time_model-output_dim | None | 56 | None | 32 | None | 
 | event_model-continuous_hidden_dims | 20 | 16 | 16 | 12 | 20 | 
 | event_model-continuous_output_dim | 24 | 40 | 64 | 24 | 64 | 
 | event_model-tf_activation | gelu | relu | gelu | tanh | tanh | 
 | temporal_model-num_layers | 3 | 4 | 4 | 5 | 1 | 
 | temporal_model-attn_dropout_prob | 0.4 | 0 | 0.2 | 0.2 | 0.2 | 
 | temporal_model-feed_forward_hidden | 8 | 56 | 56 | 40 | 16 | 
 | temporal_model-hidden_act | tanh | gelu | gelu | relu | relu | 
 | temporal_model-hidden_dropout_prob | 0.2 | 0.1 | 0 | 0.3 | 0.1 | 
 | classifier_model-hidden_dim | 16 | 24 | 24 | 16 | 24 | 
 | classifier_model-output_dropout | 0.4 | 0.5 | 0.3 | 0.1 | 0.3 | 
