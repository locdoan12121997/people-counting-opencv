class LineFunction(object):
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        self.a = 0.
        self.b = 0.
        self.c = 0.

    def fx(self, point):
        return (point[0] - self.point1[0]) * (self.point2[1] - self.point1[1]) - (point[1] - self.point1[1]) * (
                    self.point2[0] - self.point1[0])
