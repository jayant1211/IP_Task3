import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.



# model
MODEL_NAME = 'class_traffic'  # change to whatever folder has the new graph

# Path to frozen detection graph. This is the actual model that is used for the object detection.
pb_path = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
label_path = os.path.join('training', 'object-detection.pbtxt')  # our labels are in training/object-detection.pbkt

NUM_CLASSES = 1  # we only are using one class at the moment (mask at the time of edit)

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map

# In[7]:
label_map = label_map_util.load_labelmap(label_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height,channel) = image.shape
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# using only 2 images:
# image1.jpg
# image2.jpg
test_dir = 'test'
img_path = [os.path.join(test_dir, '{}.png'.format(i)) for i in range(6)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

mapping = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        cap = cv2.VideoCapture('2.avi')
        cmodel = tf.keras.models.load_model("classification.h5")
        i = 0
        for image_path in img_path:
            image_o = cv2.imread(image_path)
            image = cv2.cvtColor(image_o,cv2.COLOR_RGB2BGR)
            img_height, img_width, img_channel = image.shape
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = image
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            image_np,box=vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            #box=vis_util.custom_code()
            #ymin, xmin, ymax, xmax = (box)

            no_of_signs=len(box)

            if len(box)==1:
                if box[0]==(0,0,0,0):
                    no_of_signs=0

            #print(no_of_signs)
            #continue
            for i in range(no_of_signs):
                ymin, xmin, ymax, xmax = box[i]
                x_up = int(xmin * img_width)
                y_up = int(ymin * img_height)
                x_down = int(xmax * img_width)
                y_down = int(ymax * img_height)
                print(x_up, y_up, x_down, y_down)
                #img2 = cv2.rectangle(image, (x_up, y_up), (x_down, y_down), (0, 0, 0), 2)
                img2 = image
                cv2.imshow('result', cv2.cvtColor(cv2.resize(img2, (800, 600)),cv2.COLOR_BGR2RGB))

                if x_up != 0:
                    cropped = image_np[y_up:y_down, x_up:x_down]
                    cropped = cv2.cvtColor(cv2.resize(cropped, (28, 28)),cv2.COLOR_BGR2RGB)
                    cv2.imshow('cropped', cv2.resize(cropped, (80, 60)))
                    data = []
                    data.append(np.array(cropped))
                    sign = np.array(data)
                    score = cmodel.predict(sign)
                    Y_pred = cmodel.predict_classes(sign)
                    int_array = Y_pred.astype(int)
                    a = int(Y_pred[0])
                    print(mapping[a])
                    final = cv2.resize((cv2.putText(img2, mapping[a],(x_up,y_down+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)), (800, 600))
                    cv2.imshow('result', cv2.cvtColor(final,cv2.COLOR_BGR2RGB))

            if no_of_signs==0:
                cv2.imshow('result', cv2.cvtColor(cv2.resize(image, (800, 600)), cv2.COLOR_BGR2RGB))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            cv2.waitKey(0)
