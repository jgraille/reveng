class Database:

    def __init__(self, config):
        self.server = config['server']
    
    def __repr__(self):
        return f"{self.server}"



