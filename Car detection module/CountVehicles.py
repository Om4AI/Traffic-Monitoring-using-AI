import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

# -------------------- Utils -----------------------
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)


# --------------- Vehicle Count Function ------------------
def getVehicleCount(img_path):
    # Get the image
    path = img_path
    image_np = load_image_into_numpy_array(path)
    model_handle = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1"

    # Get the model for inference
    model = hub.load(model_handle)

    results = model(image_np)
    result = {key:value.numpy() for key,value in results.items()}

    l  = result['detection_scores'][0]
    cars  = 0
    trucks = 0
    for i in range(len(l)):
        if(l[i]>0.3 and result['detection_classes'][0][i]==3): cars+=1
        elif(l[i]>0.3 and result['detection_classes'][0][i]==8): trucks+=1


    # ----------- Draw the bounding boxes ---------------
    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
    )

    plt.figure(figsize=(24,32))
    plt.imshow(image_np_with_detections[0])
    plt.axis(False)
    plt.show()

    return (cars,trucks)