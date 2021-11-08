from src.data.dataset import ColaBeerDataset
import numpy as np
from collections import OrderedDict

class CentroidTracker():
    def __init__(self, max_dissapear_frames: int = 50, max_dist_same_obj = 60):
        self._max_dissapear = max_dissapear_frames
        self._next_object_id = 0
        self._tracking_objects = OrderedDict()
        self._dissapeared = OrderedDict()
        self._max_dist_same_obj = max_dist_same_obj


    def _register(self, centroid):
        self._tracking_objects[self._next_object_id] = centroid
        self._dissapeared[self._next_object_id] = 0
        self._next_object_id += 1


    def _remove(self, object_id):
        del self._tracking_objects[object_id]
        del self._dissapeared[object_id]


    def update(self, boxes):
        centroids = []

        for xmin, ymin, xmax, ymax in boxes:
            x = int(xmin + ((xmax - xmin)//2))
            y = int(ymin + ((ymax - ymin)//2))

            centroids.append((x, y))

        assigned_centroid_idx = set()
        assigned_obj_id = set()

        if len(centroids) > 0:
            ## calculate distances to current objects
            distances = []
            for obj_id, curr_centroid in self._tracking_objects.items():
                obj_distances = []
                for centroid in centroids:
                    obj_distances.append(np.sqrt((curr_centroid[0] - centroid[0])**2 + 
                        (curr_centroid[1] - centroid[1])**2))
                distances.append((obj_id, obj_distances))

            ## greedily assign
            for obj_id, dist in distances:
                best_idx = np.argmin(dist)
                if best_idx not in assigned_centroid_idx and dist[best_idx] < self._max_dist_same_obj:
                    self._tracking_objects[obj_id] = centroids[best_idx]
                    self._dissapeared[obj_id] = 0
                    assigned_centroid_idx.add(best_idx)
                    assigned_obj_id.add(obj_id)

        ## track dissapeared
        to_remove = []
        for obj_id in self._tracking_objects:
            if obj_id not in assigned_obj_id:
                self._dissapeared[obj_id] += 1
                if self._dissapeared[obj_id] >= self._max_dissapear:
                    to_remove.append(obj_id)

        for obj_id in to_remove:
            self._remove(obj_id)
                

        ## add non assigned
        for idx, centroid in enumerate(centroids):
            if idx not in assigned_centroid_idx:
                self._register(centroid)


    def objects(self):
        return self._tracking_objects.items()


if __name__ == "__main__":
    import cv2

    dataset = ColaBeerDataset("../../data/train")
    tracker = CentroidTracker(20, 100)

    for img, targets in dataset:
        boxes = targets['boxes']
        tracker.update(boxes)

        img = np.array(img)
        img = img[:, :, ::-1].copy()

        for obj_id, centroid in tracker.objects():
            print(obj_id, centroid)
            img = cv2.circle(img, centroid, radius=0, color=(0, 0, 255), thickness=10)
            img = cv2.putText(img, str(obj_id), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('main', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
