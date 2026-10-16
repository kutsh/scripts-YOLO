import os

from pathlib import Path

import numpy as np

from PIL import Image



def list_files_recursive(directory: Path, extensions: list[str] | None = None):

    files = []

    for entry in os.listdir(directory):

        full_path = os.path.join(directory, entry)

        if os.path.isdir(full_path):

            files.extend(list_files_recursive(full_path, extensions))

        else:

            if extensions == None or any(extension in full_path for extension in extensions):

                files.append(Path(full_path))

    return files