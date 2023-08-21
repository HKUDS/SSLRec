optimizer:
  name: adam
  lr: 1.0e-3
  weight_decay: 0

train:
  epoch: 200
  batch_size: 4096
  save_model: false
  loss: pairwise
  test_step: 1
  #pretrain_path: ./checkpoint/xxxx.pth
  reproducible: true
  seed: 421
  trainer: AdaGCLTrainer

test:
  metrics: [recall, ndcg]
  k: [10, 20, 40]
  batch_size: 256

data:
  type: general_cf # choose in {general_cf, multi_behavior, sequential, social}
  name: yelp

model:
  name: adagcl # case-insensitive
  layer_num: 2
  reg_weight: 1.0e-5
  cl_weight: 1.0e-1
  ib_weight: 1.0e-2
  temperature: 0.5
  embedding_size: 32
  gamma: -0.45
  zeta: 1.05
  init_temperature: 2.0
  temperature_decay: 0.98
  lambda0: 1.0e-4
  
