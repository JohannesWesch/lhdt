def _set_args(encoder_flag, max_input, max_output, L_E, L_D, grad_checkpoint=False):
    global encoder_only, max_encoder_position_embeddings, max_decoder_position_embeddings, num_layers, num_decoder_layers, gradient_checkpointing
    encoder_only = encoder_flag
    max_encoder_position_embeddings = max_input
    max_decoder_position_embeddings = max_output
    num_layers = L_E
    num_decoder_layers = L_D
    gradient_checkpointing = grad_checkpoint

# HDT-E paper configuration: 12 layers, 768 hidden, 12 heads = 109M params
d_model = 768
d_kv = 64
d_ff = 3072
num_layers = 12
num_decoder_layers = None
num_heads = 12
relative_attention_num_buckets = 32
relative_attention_max_distance = 128
dropout_rate = 0.1
layer_norm_epsilon = 1e-6
initializer_factor = 1.0
feed_forward_proj = "relu"
is_encoder_decoder = False
use_cache = True
use_bias = False
pad_token_id = 0
eos_token_id = 1
# Paper uses 8192 context length for long documents
max_encoder_position_embeddings = 8192
max_decoder_position_embeddings = 256
position_embedding_type = "absolute"
decode_anchor_tokens = False
is_hierarchical = True
encoder_only = False
gradient_checkpointing = False  # Enable to save memory at the cost of speed