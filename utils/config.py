class ConfigDict(dict):
    """
    Access-by-attribute, case-insensitive dictionary
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if not k.islower() and k.isupper():
                self.pop(k)
                self[k.lower()] = v
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k.lower()] = ConfigDict(v)

    def __getattr__(self, name):
        return self.get(name.lower())

    def __setattr__(self, name, value):
        self.__setitem__(name.lower(), value)
