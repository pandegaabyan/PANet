# Train 1way 1shot
python train.py with gpu_id=0 mode='train' dataset='VOC' label_sets=0 n_steps=50 print_interval=10 model.align=False task.n_ways=1 task.n_shots=1

# Test 1way 1shot
python test.py with gpu_id=0 mode='test' snapshot='./runs/PANet_VOC_sets_0_1way_1shot_[train]/3/snapshots/50.pth'

# Test 1way 5shot
python test.py with gpu_id=0 mode='test' snapshot='./runs/PANet_VOC_sets_0_1way_5shot_[train]/1/snapshots/30000.pth'