import numpy as np
class BoundingBox:
    
    def __init__(self, vesselID, centre, width, height, depth):
        self.vesselID = vesselID
        self.centre = centre
        self.width = width
        self.height = height
        self.depth = depth
        self.visibility = 1.0

    def get_xmin(self):
        return self.centre[0]-self.width/2
    
    def get_xmax(self):
        return self.centre[0]+self.width/2
    
    def get_ymin(self):
        return self.centre[1]-self.height/2
    
    def get_ymax(self):
        return self.centre[1]+self.height/2
    
    def update_bounding_box(self, xmin, xmax, ymin, ymax):
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.centre = [xmin + self.width/2, ymin + self.height/2]

    def check_overlap(self, bounding_box):
        "Returns True if the bounding boxes overlaps each other, and false if there is no overlap"
        if self.get_xmin() >= bounding_box.get_xmax() or self.get_xmax() <= bounding_box.get_xmin() or self.get_ymin() >= bounding_box.get_ymax() or self.get_ymax() <= bounding_box.get_ymin():
            return False
        return True

    def check_overlap_or_adjacent(self, bounding_box):
        "Returns true if the bounding boxes either overlap or have a common boundry (i.e. touches along an edge) and false if not"
        if self.get_xmin() > bounding_box.get_xmax() or self.get_xmax() < bounding_box.get_xmin() or self.get_ymin() > bounding_box.get_ymax() or self.get_ymax() < bounding_box.get_ymin():
            return False
        return True

    def check_fully_covered(self, bounding_box):
        "Returns true if bounding_box fully covers this bounding box, and false if not" 
        if self.get_xmin() >= bounding_box.get_xmin() and self.get_xmax() <= bounding_box.get_xmax() and self.get_ymin() >= bounding_box.get_ymin() and self.get_ymax() <= bounding_box.get_ymax():
            return True
        return False     

    def update_size_if_covered(self, bounding_box):
        """Checks whether two neighbouring coners of the bounding box are both covered, if so it cuts the bounding box to the size that is visible. 
        If only one corner is covered or not two neighbouring corners, the bounding box stays unchanged. """
        if bounding_box.get_xmin() < self.get_xmin() and bounding_box.get_xmax() > self.get_xmax() and bounding_box.get_ymax() > self.get_ymax() and bounding_box.get_ymin() > self.get_ymin() and bounding_box.get_ymin() < self.get_ymax():
            self.update_bounding_box(self.get_xmin(), self.get_xmax(), self.get_ymin(), bounding_box.get_ymin())
        elif bounding_box.get_xmin() < self.get_xmin() and bounding_box.get_xmax() > self.get_xmax() and bounding_box.get_ymax() < self.get_ymax() and bounding_box.get_ymax() < self.get_ymin() and bounding_box.get_ymin() < self.get_ymin():
            self.update_bounding_box(self.get_xmin(), self.get_xmax(), bounding_box.get_ymax(), self.get_ymax())
        elif bounding_box.get_ymin() < self.get_ymin() and bounding_box.get_ymax() > self.get_ymax() and bounding_box.get_xmin() < self.get_xmin() and bounding_box.get_xmax() > self.get_xmin() and bounding_box.get_xmax() < self.get_xmax():
            self.update_bounding_box(bounding_box.get_xmax(), self.get_xmax(), self.get_ymin(), self.get_ymax())
        elif bounding_box.get_ymin() < self.get_ymin() and bounding_box.get_ymax() > self.get_ymax() and bounding_box.get_xmax() > self.get_xmax() and bounding_box.get_xmin() < self.get_xmax() and bounding_box.get_xmin() > self.get_xmin(): 
            self.update_bounding_box(self.get_xmin(), bounding_box.get_xmin(), self.get_ymin(), self.get_ymax())
    
    def create_bbs_for_overlapping_or_adjacent_area(self, bounding_box):
        """ Returns two fake bounding boxes to represent overlapping areas for two overlapping bounding boxes. 
        Should only be used when the bounding boxes are overlapping (Can add a check for this).
        This is used when two bounding boxes overlaps another, and lets us check if those two bouning boxes together cover two neighbouring corners.
        """

        # Horizontal
        horizontal_xmin = np.min(np.array([self.get_xmin(), bounding_box.get_xmin()]))
        horizontal_xmax = np.max(np.array([self.get_xmax(), bounding_box.get_xmax()]))
        horizontal_ymin = np.min(np.array([self.get_ymin(), bounding_box.get_ymin()]))
        horizontal_ymax = np.max(np.array([self.get_ymax(), bounding_box.get_ymax()]))

        horizontal_width = horizontal_xmax - horizontal_xmin
        horizontal_height = horizontal_ymax - horizontal_ymin
        horizontal_centre = [horizontal_xmin + horizontal_width/2, horizontal_ymin + horizontal_height/2]
        
        horizontal_bb = BoundingBox(None, horizontal_centre, horizontal_width, horizontal_height, self.depth)

        # Vertical
        vertical_xmin = np.max(np.array([self.get_xmin(), bounding_box.get_xmin()]))
        vertical_xmax = np.min(np.array([self.get_xmax(), bounding_box.get_xmax()]))
        vertical_ymin = np.max(np.array([self.get_ymin(), bounding_box.get_ymin()]))
        vertical_ymax = np.min(np.array([self.get_ymax(), bounding_box.get_ymax()]))

        vertical_width = vertical_xmax - vertical_xmin
        vertical_height = vertical_ymax - vertical_ymin
        vertical_centre = [vertical_xmin + vertical_width/2, vertical_ymin + vertical_height/2]
        
        vertical_bb = BoundingBox(None, vertical_centre, vertical_width, vertical_height, self.depth)

        return horizontal_bb, vertical_bb

    
    def add_bbs_for_overlapping_or_adjacent_areas(self, bbs_covering_this_bb):
        """Adds fake bounding boxes for overlapping og adjacent areas between the bounding boxes that overlaps this bounding box"""
        covering_bbs = []
        if len(bbs_covering_this_bb)>1:
            for i in range(len(bbs_covering_this_bb)-1):
                for j in range(i+1, len(bbs_covering_this_bb)):
                    if bbs_covering_this_bb[i].check_overlap_or_adjacent(bbs_covering_this_bb[j]):
                        horizontal_bb, vertical_bb = bbs_covering_this_bb[i].create_bbs_for_overlapping_or_adjacent_area(bbs_covering_this_bb[j])
                        covering_bbs.append(horizontal_bb)
                        covering_bbs.append(vertical_bb)
                covering_bbs.append(bbs_covering_this_bb[i])
        else:
            covering_bbs.append(bbs_covering_this_bb[0])
        return covering_bbs

    def update_bb_if_covered(self, bbs_covering_this_bb):
        covering_bbs = self.add_bbs_for_overlapping_or_adjacent_areas(bbs_covering_this_bb)
        sorted_bbs = sorted(covering_bbs, key=lambda bb: bb.depth, reverse=True)
        for i in range(len(sorted_bbs)):
            self.update_size_if_covered(sorted_bbs[i])
    
    def get_points_for_visualizing(self):
        cx = self.centre[0]
        cy = self.centre[1]
        w = self.width
        h = self.height
        xs = [cx-w/2, cx-w/2, cx+w/2, cx+w/2, cx-w/2] 
        ys = [cy-h/2, cy+h/2, cy+h/2, cy-h/2, cy-h/2]
        return xs, ys
    
    def update_visibility(self, covering_bbs):
        total_area = self.height * self.width
        total_covered_area = 0
        sorted_bbs = sorted(covering_bbs, key=lambda bb: bb.depth, reverse=False)
        overlapping_areas = []
        for bb in sorted_bbs:
            overlap_bb, overlap_area = self.find_overlapping_area(bb)
            if len(overlapping_areas) > 0:
                for overlap in overlapping_areas:
                    if overlap_bb.check_overlap(overlap):
                        _, area = overlap_bb.find_overlapping_area(overlap)
                        overlap_area -= area
            total_covered_area += overlap_area
            overlapping_areas.append(overlap_bb)
        self.visibility = 1 - total_covered_area/total_area
    
    def find_overlapping_area(self, bb):
        overlap_xmax = np.min(np.array([self.get_xmax(), bb.get_xmax()]))
        overlap_xmin = np.max(np.array([self.get_xmin(), bb.get_xmin()]))
        overlap_ymax = np.min(np.array([self.get_ymax(), bb.get_ymax()]))
        overlap_ymin = np.max(np.array([self.get_ymin(), bb.get_ymin()]))

        x_overlap = overlap_xmax - overlap_xmin
        y_overlap = overlap_ymax - overlap_ymin
        overlap_area = x_overlap * y_overlap

        overlap_centre = [overlap_xmin + x_overlap/2, overlap_ymin + overlap_ymin/2]

        overlap_bb = BoundingBox(None, overlap_centre, x_overlap, y_overlap, None)
        return overlap_bb, overlap_area