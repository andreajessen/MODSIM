def check_bb_overlap(bb1, bb2):
    # How do we want to represent the BBs?
    # Change this according to how the inputted BBs are structured: we need max x, y and min x, y for both BB
    bb1_xmin = bb1[0][0]
    bb1_ymin = bb1[0][1]
    bb1_xmax = bb1[2][0]
    bb1_ymax = bb1[2][1]
    bb2_xmin = bb2[0][0]
    bb2_ymin = bb2[0][1]
    bb2_xmax = bb2[2][0]
    bb2_ymax = bb2[2][1]

    if bb1_xmin >= bb2_xmax or bb1_xmax <= bb2_xmin or bb1_ymin >= bb2_ymax or bb1_ymax <= bb2_ymin:
        return False
    return True