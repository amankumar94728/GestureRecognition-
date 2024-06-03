This project is a prototype to detect a specific gesture within a video sequence. The 
gesture will be defined by an input image or a short video clip. . If the gesture is detected, overlay the 
word "DETECTED" in bright green on the top right corner of the output frame(s).

NOTE:

## if you wish to run it on your own test image or video , place it in input folder and change the name in the files `handrecog` `handrecogimg`

### To install requirement for the project type pip3 -r requirements.txt and run 
  1. For mac/Linux python3 handrecog.py  -->  For Video
                python3 handrecogimg.py  -->  For image

  2. For Window    python handrecog.py  -->  For Video
                python handrecogimg.py  -->  For image  

I have used "google/vit-base-patch16-224-in21k" pretent model.

The sample are present in output folder.