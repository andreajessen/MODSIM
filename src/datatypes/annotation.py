from datatypes.boundingBox import BoundingBox

class Annotation:

    def __init__(self,  boundingBox, label, vesselID = None):
        self.bb = boundingBox
        self.label = label
        self.vesselID = vesselID