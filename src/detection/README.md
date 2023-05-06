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
- Or in terminal run (change model to train weights paths and source to test image path)
```
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_1cls_100e_640imgsz_terminal/weights/best.pt source= '/cluster/home/solveijm/hurtigruten/test.txt' imgsz=1920 name=test_1cls_100e_640imgsz_terminal save=True save_conf=True save_txt=True

```


## Test yolo on syntetich data

```
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_light_condition1.txt' imgsz=1920 name=test_light_condition1 save=True save_conf=True save_txt=True
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_light_condition2.txt' imgsz=1920 name=test_light_condition2 save=True save_conf=True save_txt=True
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_light_condition3.txt' imgsz=1920 name=test_light_condition3 save=True save_conf=True save_txt=True

yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_cloud_light1.txt' imgsz=1920 name=test_cloud_light1 save=True save_conf=True save_txt=True
!yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_cloud_light2.txt' imgsz=1920 name=test_cloud_light2 save=True save_conf=True save_txt=True
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_cloud_light3.txt' imgsz=1920 name=test_cloud_light3 save=True save_conf=True save_txt=True

yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_rain_light1.txt' imgsz=1920 name=test_rain_light1 save=True save_conf=True save_txt=True
yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_rain_light2.txt' imgsz=1920 name=test_rain_light2 save=True save_conf=True save_txt=True

yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_foggy.txt' imgsz=1920 name=test_foggy save=True save_conf=True save_txt=True

yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_stormy.txt' imgsz=1920 name=test_stormy save=True save_conf=True save_txt=True

yolo task=detect mode=predict model=/cluster/work/solveijm/MODSIM/runs/detect/train_100e_and_syntetich3/weights/best.pt source='/cluster/home/solveijm/DNV_data/test_stormy_rain.txt' imgsz=1920 name=test_stormy_rain save=True save_conf=True save_txt=True
```

