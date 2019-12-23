class BoxDrawController:
    def __init__(self, remain_frame=1000):
        self.enter_objects = dict()
        self.exit_objects = dict()
        self.remain_frame = remain_frame

    def enter_register(self, objectId):
        self.enter_objects[objectId] = self.remain_frame

    def exit_register(self, objectId):
        self.exit_objects[objectId] = self.remain_frame

    def enter_deregister(self, objectId):
        del self.enter_objects[objectId]

    def exit_deregister(self, objectId):
        del self.exit_objects[objectId]

    def degenerate(self):
        if self.enter_objects:
            self.enter_objects = { objectId : remain_time for objectId, remain_time in self.enter_objects.items() if remain_time>0}
            for (objectID, remain_time) in self.enter_objects.items():
                self.enter_objects[objectID] -= 1
        if self.exit_objects:
            self.exit_objects = { objectId : remain_time for objectId, remain_time in self.exit_objects.items() if remain_time>0}
            for (objectID, remain_time) in self.exit_objects.items():
                self.exit_objects[objectID] -= 1

    def get_enter_object_list(self):
        return self.enter_objects

    def get_exit_object_list(self):
        return self.exit_objects
