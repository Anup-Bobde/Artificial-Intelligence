# Project for Object Detection by ultralytics(YOLO-it is pretrained model)
# This is unstructured data analysis i.e.image analysis-yolo v8 model(also try yolo v11 model)
# Also watch other projects on ultralytics website
''' 
After running this program the output image will be automatically stored in the 
predict folder of runs folder in VS Code_DataAnalytics folder & this runs folder
will be automatically created by the system but our original image is present in the
YOLO folder & we have copy paste runs folder in our YOLO folder.
And weights folder also automatically created in VS Code_DataAnalytics folder, 
it contain yolo8n file. 
'''
# For more details for above explanation see video in screen recordings
# Note: The internet should be on to run this program

from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  


# predict on an image
detection_output = model.predict(source=r"C:\Users\Anup\VS Code_DataAnalytics\YOLO\1.JPG", conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())


