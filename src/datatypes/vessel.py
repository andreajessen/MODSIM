class Vessel():

    def __init__(self, air_draft, beam, length, label):
        """
        Beam (int): The width of the widest point of the boat
        air_draft (int): The distance between the ship's waterline and the highest point of the boat; indicates the distance the vessel can safely pass under
        Length (int): Length of the boat
        Label (string): boat class (Sailboat, motorboat etc)
        """
        self.air_draft = air_draft
        self.beam = beam
        self.length = length
        self.label = label
        self.track = None
    
    def __init__(self):
        self.air_draft = 0
        self.beam = 0
        self.length = 0
        self.label = ""
        self.track = None
    
    def set_track(self, track):
        self.track = track
    
    def get_track(self):
        return self.track