class BoundingBox:
    
    def __init__(self, vesselID, centre, width, height, depth):
        self.vesselID = vesselID
        self.centre = centre
        self.width = width
        self.height = height
        self.depth = depth

    def get_xmin(self):
        return self.centre[0]-self.width/2
    
    def get_xmax(self):
        return self.centre[0]+self.width/2
    
    def get_ymin(self):
        return self.centre[1]-self.height/2
    
    def get_ymax(self):
        return self.centre[1]+self.height/2