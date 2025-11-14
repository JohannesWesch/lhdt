def _set_args(run_name):
    global name
    name = run_name

offline = False  # Save locally, sync later with: wandb sync path/to/run
id = None  # pass correct id to resume experiment!
anonymous = None  # enable anonymous logging
project = "pretrain_HDT"
log_model: False  # upload lightning ckpts
prefix = ""  # a string to put at the beginning of metric keys
entity = None  # Use default wandb account (jo-wesch-karlsruhe-institute-of-technology)
group = ""
tags = []
job_type = ""
name = "HDT_8192"