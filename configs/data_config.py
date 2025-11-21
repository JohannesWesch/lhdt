def _set_args(tok, max_input_length, mlm_p):
    global tok_name, model_max_length, mlm_probability
    tok_name = tok
    model_max_length = max_input_length
    mlm_probability = mlm_p

num_proc = 6  # Optimal for multi-GPU setup
max_entries_in_raw_dataset = 1e10
preprocess_batch_size = 2048
# Paper uses google-bert/bert-base-uncased for HDT-E, but T5 tokenizer works better
# T5 has 32,128 vocab which is closer to the 32,768 model capacity
tok_name = "google-bert/bert-base-uncased"
model_max_length = 8192  # Paper uses 8192 context for long documents
vocab_size = 32768  # Fixed vocabulary size for the model
mlm_probability = 0.15  # Standard MLM masking probability
use_streaming = True  # Enable streaming to avoid downloading 110+ GB of data
# Paper datasets: unarXive, Wikipedia, HUPD
ds_info = [{"path": "howey/unarXive", "name": "default", "split": "train"}, {"path": "howey/wiki_en", "name": "default", "split": "train"}, {"path": "howey/hupd", "name": "default", "split": "train"}]