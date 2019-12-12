class StayDurationEstimator:
    def __init__(self, fps):
        self.frame_in = list()
        self.frame_out = list()
        self.fps = fps

    def add_frame_in(self, number_of_frame):
        self.frame_in.append(number_of_frame)

    def add_frame_out(self, number_of_frame):
        if len(self.frame_in) > len(self.frame_out):
            self.frame_out.append(number_of_frame)

    def calculate_average_duration(self):
        number_of_people = min(len(self.frame_in), len(self.frame_out))
        return (sum(self.frame_out[:number_of_people]) - sum(self.frame_in[:number_of_people])) / self.fps
