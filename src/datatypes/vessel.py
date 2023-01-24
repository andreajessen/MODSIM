class Vessel():

    def __init__(self, air_draft=2, beam=2, length=4, label=""):
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
    
    def set_track(self, track):
        self.track = track
    
    def get_track(self):
        return self.track

    def get_beam(self):
        return self.beam
    
    def get_length(self):
        return self.length