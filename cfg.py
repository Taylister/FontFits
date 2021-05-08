# loss weight
theta_per = 1.
theta_style = 1000.
theta_gan = 1e-2
epsilon = 1e-8

# learning parameter
learning_rate = 1e-4 
beta1 = 0.9
beta2 = 0.999 
momentum = 0.9
rho = 0.05

# enviromnent paramter
pred_gpu = True
pred_mode = "multiple"

## There are two mode in pred_mode.
### One is correspond. Corresponding to file name(i_s, mask, i_t), The network generates o_t and o_sk(these file name are same as input).
### Another is multiple. 
#### Bellow file path and directory path used for multiple mode
pred_i_t_dir = 'multi_gen/source_text'
pred_style_path = 'dataset/cover_inpaint/test/006066567X.jpg'
pred_style_mask_path = 'dataset/cover_mask/test/006066567X_00.jpg'

# training parameter
## period
ex_max_iter = 500000
write_log_interval = 100
save_ckpt_interval = 100000
gen_example_interval = 5000
## data
batch_size = 8
data_shape = [256, None]

# path(directory) setting
## Dataset path(directory)
data_dir = 'dataset'
example_data_dir = 'dataset'

train_data_dir = 'train'
validation_data_dir = 'validation'
test_data_dir = 'test'

### Network Input
style = 'cover_inpaint'
style_mask = 'cover_mask'
input_text = 'input_text'

### Network True output
c_mask = 'processed_mask'
c_skeleton = 'skeletonize'
extracted_title = 'extracted_title'

## Output path(directory)
train_result_dir = 'result'
pred_result_dir = 'eval'
checkpoint_savedir = 'logs'

## checkpoint path
### check point path to resume train
train_ckpt_path = "result/20210218193522/logs/train_step-400000.model"

### check point path to predict
model_weight = "result/20210126003839/logs/train_step-380000.model"


