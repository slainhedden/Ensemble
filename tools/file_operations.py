import os

class FileOperations:
    def __init__(self, base_dir='project_files'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def write_file(self, filename: str, content: str):
        with open(os.path.join(self.base_dir, filename), 'w') as f:
            f.write(content)

    def read_file(self, filename: str) -> str:
        with open(os.path.join(self.base_dir, filename), 'r') as f:
            return f.read()

    def append_file(self, filename: str, content: str):
        with open(os.path.join(self.base_dir, filename), 'a') as f:
            f.write(content)

    def list_files(self):
        return os.listdir(self.base_dir)