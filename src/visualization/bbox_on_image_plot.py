import cv2 as cvdef bbox_on_image(im, bbox, label):    # Load image and bbox dimensions    img = cv.imread(im)    # Dimension of bounding box    xmin, ymin, xmax, ymax = bbox        # Plot label dependent    if label == 1:        # Green rectangle        cv.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 255, 0), 2)        # Set text beer upper left corner over the bbox in same green color        cv.putText(img, 'Beer', (xmin,ymin-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,0), 2)    elif label == 2:        # Red rectangle        cv.rectangle(img, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)        # Put text coke above upper left corner of bbox in same red color        cv.putText(img, 'Coke', (xmin,ymin-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)    # Show the image.    cv.imshow("Objects detected: ", img)