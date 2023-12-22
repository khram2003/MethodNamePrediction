import os
from typing import List


def get_code_filenames_from_directory(directory: str) -> List[str]:
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".kt") or file.endswith(".java") or file.endswith(".js") or file.endswith(
                    ".py") or file.endswith(".cpp") \
                    or file.endswith(".c") or file.endswith(".php") or file.endswith(
                ".go") or file.endswith(".rs") \
                    or file.endswith('rb'):
                filenames.append(os.path.join(root, file))
    return filenames
