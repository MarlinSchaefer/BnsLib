import os


class TempFile(str):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if os.path.isfile(self):
            os.remove(self)
