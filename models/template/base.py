class BaseTemplate:
    def __init__(self,
                 prefix: str):
        self.prefix = prefix
    def encode(self, prompt):
        raise NotImplementedError