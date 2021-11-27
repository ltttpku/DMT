This repo is created for the project of "Fundamental of Digital Media Tech".

### Usage
 + pip install -r requirement.txt
 + python train.py --config configs/cell.txt
 + You could modify any parameter in cell.txt.

### TODO List
 + fix mask groundtruth ! 
 + need val code as well as test-time code
 + data augumentation
 + batch_size too small(OOM shocks me), may need DDP
 + 