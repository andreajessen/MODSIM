class ProjectedPoint:
    
    def __init__(self, image_coordinate, depth):
        self.image_coordinate = image_coordinate
        self.depth = depth
    
    def get_x(self):
        return self.image_coordinate[0]

    def get_y(self):
        return self.image_coordinate[1]
    
    def get_depth(self):
        return self.depth