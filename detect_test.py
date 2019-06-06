import numpy as np
import cv2 as cv
import time
import tensorflow as tf

CAM_CONF = False
TPU = True

c_width = 0
c_height = 0
c_fps = 0
cap = cv.VideoCapture(-1)
    
if CAM_CONF:   
#frame width
    cap.set(3,c_width)
#frame height
    cap.set(4,c_height)
#frames per second
    cap.set(5,c_fps)
#saturation
    #cap.set(12,0)
#convert rgb
    #cap.set(16,1)
    
print('width, height, fps, convert rgb:')
print(str(cap.get(3)) + ', ' + str(cap.get(4)) + ', ' + str(cap.get(5)) + ', ' + str(cap.get(16)))

print('TPU:')
print(str(TPU))

labels = {}

if TPU:
    i = 0
    for row in open('lite_labelmap.txt'):
        # unpack the row and update the labels dictionary
        label = row.strip()
        labels[int(i)] = label.strip()
        i += 1
else:
    #load labels
    labels_path = 'parsed_labels.txt'
    label_file = open(labels_path, 'r')
    labels = eval(label_file.read())
    label_file.close()
print('labels loaded')

sess = None
graph_def = None

#ticks/sec
freq = cv.getTickFrequency()
font = cv.FONT_HERSHEY_SIMPLEX
frame_rate = 1

model_times = []
f_rates = []

if TPU:
    from edgetpu.detection.engine import DetectionEngine
    from PIL import Image
    model = DetectionEngine('detect.tflite')
else:
    # Read the graph.
    with tf.gfile.FastGFile('ssdlite_mnet.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    #load tensorflow model into memory
    sess = tf.Session()
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    print('tf model loaded')


while(True):
    
    # Capture frame-by-frame
    frame_s = time.time()
    ret, img = cap.read()
    
    # Read and preprocess an image.
    rows = img.shape[0]
    cols = img.shape[1]
    #inp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inp = np.copy

    # Run the model
    if TPU:
        inp = Image.fromarray(np.asarray(img))
        model_s = time.time()
        results = model.DetectWithImage(inp, threshold=0.3, keep_aspect_ratio=True, relative_coord=False)
    else:
        inp = np.copy(img)
        model_s = time.time()
        (num, scores, boxes, classes) = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                                sess.graph.get_tensor_by_name('detection_classes:0')],
                                                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    model_e = time.time()
    model_times.append(model_e - model_s)
    
    # Visualize detected bounding boxes.
    # loop over the results
    car_detect = 0
    if TPU:
        for r in results:
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            label = labels[r.label_id]
            if label == 'car':
                car_detect = 1

            # draw the bounding box and label on the image
            cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv.putText(img, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    else:
        num_detections = int(num[0])
        for i in range(num_detections):
            classId = int(classes[0][i])
            score = float(scores[0][i])
            bbox = [float(v) for v in boxes[0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                label = labels[classId]
                if label == 'car':
                    car_detect = 1
                textY = y - 15 if y - 15 > 15 else y + 15
                text = "{}: {:.2f}%".format(label, score * 100)
                cv.putText(img, text, (int(x), int(textY)), font, .5, (0, 255, 0), 2)
                
    print(car_detect)
                
    cv.putText(img, "FPS: {0:.2f}".format(frame_rate), (30,50), font, .5, (255,255,0), 2,cv.LINE_AA) 
    cv.imshow('TensorFlow MobileNet-SSD', img)
    
    frame_e = time.time()
    frame_rate = 1/(frame_e - frame_s)
    f_rates.append(frame_rate)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
avg_model_time = sum(model_times) / len(model_times)
avg_f_rate = sum(f_rates) / len(f_rates)
print('Average model run time (s):')
print(avg_model_time)
print('Average frame rate: (s)')
print(avg_f_rate)
