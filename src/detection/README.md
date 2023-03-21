## Train and validate yolo on Idun

- Log in:
```
# Login using idun-login1 node:
$ ssh <username>@idun-login1.hpc.ntnu.no
```

- Access cluster
```
# Login using idun-login1 node:
$ cd /cluster/work/username
```

- Clone git repo
```
$ git clone https://github.com/andreajessen/MODSIM.git
```

- Cd cloned repo
```
$ cd MODSIM
```

- Input correct username in src/detection/yolov8_train.slurm
    - /cluster/work/username/MODSIM/src/detection/dataset.yaml

- Upload dataset on Idun
    - Input correct dataset path in src/detection/dataset.yaml

- Change parameters/epochs in src/detection/yolov8_train.slurm

- Run job
```
$ chmod u+x src/detection/yolov8_train.slurm
$ sbatch src/detection/yolov8_train.slurm
```

- The run is saved to /runs/detect/

## Test yolo on Idun
- Input the model path you want to test in src/detection/yolov8_test.slurm
    - model=/cluster/work/username/MODSIM/runs/detect/train_name/weights/best.pt
- Run job
```
chmod u+x src/detection/yolov8_test.slurm
sbatch src/detection/yolov8_test.slurm
```
