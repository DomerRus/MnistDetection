
# Mnist detection YOLO v5

<img src="https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png" width="500">


# REST API

The REST API to the example app is described below.

## Search number on image (return image)

### Request

`POST /detect/img/`

    curl -i -H 'Accept: application/json' http://localhost:8080/yolo-service/detect/img/

### Response

    HTTP/1.1 200 OK
    Date: Sun, 2 Jun 2022 12:36:30 GMT
    Status: 200 OK
    Connection: close
    Content-Type: image/jpeg
    Content-Length: 2

    [image]


## Search number on image (return bbox coordinates)

### Request

`POST /detect/`

    curl -i -H 'Accept: application/json' http://localhost:8080/yolo-service/detect/

### Response

    HTTP/1.1 200 OK
    Date: Sun, 2 Jun 2022 12:36:30 GMT
    Status: 200 OK
    Connection: close
    Content-Type: application/json
    Content-Length: 2

    [{
        "bounds": [
            107.24603271484375,
            328.1503601074219,
            132.0777587890625,
            114.7342529296875
        ],
        "className": "5",
        "probability": 0.8773182034492493
    },...]


# Train

### mnist.yaml

```
train: ../../dataset/set/images/train
val: ../../dataset/set/images/val
test: ../../dataset/set/images/test

nc: 10

names: ["0", "1", "2","3", "4", "5", "6", "7", "8","9"]
```



### GPU Training requirements

```
py -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
```
https://developer.nvidia.com/cuda-toolkit-archive
```
### Train model

use `--device [0,1, ... or cpu]` for choose GPU 

```
python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch -1 --epochs 5 --data mnist.yaml --workers 24 --name <your project name>
```
### export model
```
python export.py --weights ./runs/train/<your project name>/weights/best.pt --include torchscript
```
