def check_bb_overlap(bb1, bb2):
    # How do we want to represent the BBs?
    # Change this according to how the inputted BBs are structured: we need max x, y and min x, y for both BB
    bb1_xmin = bb1.centre[0] - bb1.width/2
    bb1_ymin = bb1.centre[1] - bb1.height/2
    bb1_xmax = bb1.centre[0] + bb1.width/2
    bb1_ymax = bb1.centre[1] + bb1.heigth/2
    bb2_xmin = bb2.centre[0] - bb2.width/2
    bb2_ymin = bb2.centre[1] - bb2.height/2
    bb2_xmax = bb2.centre[0] + bb2.width/2
    bb2_ymax = bb2.centre[1] + bb2.heigth/2

    if bb1_xmin >= bb2_xmax or bb1_xmax <= bb2_xmin or bb1_ymin >= bb2_ymax or bb1_ymax <= bb2_ymin:
        return False
    return True