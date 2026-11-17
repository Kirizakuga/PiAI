out_dir = "out-karus"         # where checkpoints get saved
dataset = "karus"             # reads data/karus/train.bin and val.bin

# model size (~12M params)
n_layer = 6                   # 6 Transformer blocks stacked
n_head = 6                    # 6 attention heads per block
n_embd = 384                  # each token is a vector of 384 numbers
block_size = 256              # the model sees 256 tokens of context at once
dropout = 0.1                 # randomly drops 10% of signals, which fights overfitting

# training
batch_size = 32               # 32 sequences per step
gradient_accumulation_steps = 1
max_iters = 5000              # number of training steps
learning_rate = 1e-3          # small models can use a higher LR
lr_decay_iters = 5000         # LR slowly decreases until the end
min_lr = 1e-4
warmup_iters = 100            # start with a small LR, then ramp up
beta2 = 0.99

# checking progress
eval_interval = 250           # check val loss every 250 steps
eval_iters = 100
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves

# Windows + your GPU
device = "cuda"
dtype = "bfloat16"            # RTX 40-series supports this; faster and uses less memory
compile = False               # torch.compile doesn't work on Windows
wandb_log = False