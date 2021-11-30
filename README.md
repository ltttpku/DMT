This repo is created for the project of "Fundamental of Digital Media Tech".

### Usage
 + pip install -r requirement.txt
 + python train.py --config configs/cell.txt
 + You could modify any parameter in cell.txt.

### TODO List
 + visualization!
 + need val code as well as test-time code
 + data augumentation (done)
 + batch_size too small(OOM shocks me), may need DDP
 + may need to modify the augment.py
 + patch (to achieve complex task)
