from datatypes.boundingBox import BoundingBox

class Detection:

    def __init__(self,  boundingBox, label, vesselID, confidenceScore=None) -> None:
        self.bb = boundingBox
        self.label = label
        self.vesselID = vesselID
        self.confidenceScore = confidenceScore
        