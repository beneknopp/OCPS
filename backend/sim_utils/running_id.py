class RunningId:
    def __init__(self):
        self.__running_id = 1
    def inc(self):
        self.__running_id += 1
    def get(self):
        return self.__running_id
    def get_and_inc(self):
        running_id = self.__running_id
        self.inc()
        return running_id
