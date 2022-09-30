class Response():

    @classmethod
    def get(cls, resp = None, err = None):
        return {"resp": resp, "err": err}

