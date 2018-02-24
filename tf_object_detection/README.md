# farewell song 2.0
Tensorflow object detection with faster rcnn nas
### Get Tensorflow object detection models
``` bash
# from knest/tf_object_detection
git clone https://github.com/tensorflow/models
```

### Setup object detection dependencies on your machine
``` bash
# from knest/tf_object_detection/models/research
sudo python3 setup.py install
```

### Get the pre-trained model
``` bash
# navigate in your browser and download + unzip faster_rcnn_nas and move the folder into knest/tf_object_detection/
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
```

### Configure the directories to link with the object detection API
``` bash
# assuming you have all the required files you can run my handy dandy script
# data/, bird_images/, faster_rcnn_nas_coco_2017_11_08/, training/, faster_rcnn_nas_coco.config
# you might have to merge with the data folder already in models/research/object_detection
source organize_model.sh
```

### Compile Protobuf
``` bash
# from models/research/
protoc object_detection/protos/*.proto --python_out=.
```

### Add Libraries to PYTHONPATH
This needs to be ran for every new terminal instance
``` bash
# from models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### Installation tests
You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:
```bash
python object_detection/builders/model_builder_test.py
```

### Train the neural network (faster_rcnn_nas)
```bash
# from knest/tf_object_detection/models/research/object_detection
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_nas_coco.config
```

### Inference - exporting the inference graph
```bash
# from knest/tf_object_detection/models/research/object_detection
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_nas_coco.config \
    --trained_checkpoint_prefix training/INPUT_LATEST_CHECKPOINT_MODEL \
    --output_directory bird_graph
```

### Run the Inference
refactor object_detection_tutorial.ipynb into a .py file
```bash
# from knest/tf_object_detection/models/research/object_detection
python3 FILE_NAME.py
```

