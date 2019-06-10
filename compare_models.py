import numpy as np
import cv2 as cv
import time
import tensorflow as tf

#this is for comparison of tflite models 

#def def_input_stream():
#    valid = False
#    while not valid:
#        stream = input('define input stream (0 for camera, 1 for file): ')
#        if stream.strip().isdigit():
#            valid = True
#        else:
#            print('Invalid input')
#    
#    return stream
#    
#def init_config_ca(width, height, fps, saturation, conv_rgb):
#    cap = cv.VideoCapture(-1)
#    cap.set(3,width)
#    cap.set(4,height)
#    cap.set(5,fps)
#    cap.set(12,saturation)
#    #convert rgb
#    cap.set(16,1)
#    
#    print('width, height, fps, convert rgb:')
#    print(str(cap.get(3)) + ', ' + str(cap.get(4)) + ', ' + str(cap.get(5)) + ', ' + str(cap.get(16)))

##########load labels
labels = {}
i = 0
for row in open('labels/lite_labelmap.txt'):
    # unpack the row and update the labels dictionary
    label = row.strip()
    labels[int(i)] = label.strip()
    i += 1

###########load models to engines
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
model_a = DetectionEngine('tflite-models/coral_objdtct_mnet_ssd_v1.tflite')
model_b = DetectionEngine('tflite-models/detect.tflite', model_a.device_path())
print('edgetpu models loaded')

############camera
cap = cv.VideoCapture(-1)

############runtime
model_a_times = []
model_b_times = []
while True:
    
    # Capture frame-by-frame
    frame_s = time.time()
    ret, img = cap.read()
    
    # Read and preprocess an image.
    rows = img.shape[0]
    cols = img.shape[1]
    #inp = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inp = np.copy

    # Run the model
    inp = Image.fromarray(np.asarray(img))
    time_s = time.time()
    results_a = model_a.DetectWithImage(inp, threshold=0.3, keep_aspect_ratio=True, relative_coord=False)
    time_e = time.time()
    
    model_a_times.append(time_e - time_s)
    
    time_s = time.time()
    results_b = model_b.DetectWithImage(inp, threshold=0.3, keep_aspect_ration=True, relative_coord=False)
    time_e = time.time()
    
    model_b_times.append(time_e - time_s)
    
    #Visualize detected bounding boxes.
    #loop over the results
    car_detect = 0
    for r in results_a:
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
        
    for r in results_b:
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
        cv.putText(img, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
                
    #print(car_detect) 
    #cv.putText(img, "FPS: {0:.2f}".format(frame_rate), (30,50), font, .5, (255,255,0), 2,cv.LINE_AA) 
    cv.imshow('TensorFlow MobileNet-SSD', img)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
a_avg_time = sum(model_a_times) / len(model_a_times)
b_avg_time sum(model_b_times) / len(model_b_times)

print('Average model run times (s):')
print('A: ' + str(a_avg_time))
print('B: ' + str(b_avg_time))



    

