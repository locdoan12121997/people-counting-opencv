# https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
def on_segment(p, q, r):
    if max(p.x, r.x) >= q.x >= min(p.x, r.x) and max(p.y, r.y) >= q.y >= min(p.y, r.y):
        return True;
    return False;


def orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def is_inside(polygon, p):
    if len(polygon) < 3:
        return False

    extreme = Point((10000, p.y))
    count = 0
    i = 0
    while True:
        next = (i+1) % len(polygon)
        if do_intersect(polygon[i], polygon[next], p, extreme):
            if orientation(polygon[i], p, polygon[next]) == 0:
                return on_segment(polygon[i], p, polygon[next])
            count += 1
        i = next
        if i == 0:
            break
    return count % 2 == 1

class Point:
    def __init__(self, coordinate_tuple):
        self.x = coordinate_tuple[0]
        self.y = coordinate_tuple[1]
