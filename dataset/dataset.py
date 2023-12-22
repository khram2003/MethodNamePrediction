from dataclasses import dataclass
from typing import List, Any
from torch.utils.data import Dataset
from tqdm import tqdm

from tree_sitter import Language, Parser

from dataset.utils import get_code_filenames_from_directory

Language.build_library(
    'build/my-languages.so',

    [
        '../tree-sitter-kotlin',
        '../tree-sitter-java',
        '../tree-sitter-javascript',
        # '../tree-sitter-typescript',
        '../tree-sitter-python',
        '../tree-sitter-cpp',
        '../tree-sitter-c',
        # '../tree-sitter-c-sharp',
        '../tree-sitter-php',
        '../tree-sitter-go',
        '../tree-sitter-rust',
        '../tree-sitter-swift',
        # '../tree-sitter-scala',
        '../tree-sitter-ruby'
    ]
)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
KOTLIN_LANGUAGE = Language('build/my-languages.so', 'kotlin')
JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')
# TYPESCRIPT_LANGUAGE = Language('build/my-languages.so', 'typescript')
PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
C_LANGUAGE = Language('build/my-languages.so', 'c')
# C_SHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')
GO_LANGUAGE = Language('build/my-languages.so', 'go')
RUST_LANGUAGE = Language('build/my-languages.so', 'rust')
SWIFT_LANGUAGE = Language('build/my-languages.so', 'swift')
# SCALA_LANGUAGE = Language('build/my-languages.so', 'scala')
RUBY_LANGUAGE = Language('build/my-languages.so', 'ruby')

PARSERS = {
    'java': Parser(),
    'kt': Parser(),
    'js': Parser(),
    # 'ts': Parser(),
    'py': Parser(),
    'cpp': Parser(),
    'c': Parser(),
    # 'cs': Parser(),
    'php': Parser(),
    'go': Parser(),
    'rs': Parser(),
    'swift': Parser(),
    # 'scala': Parser(),
    'rb': Parser()
}

PARSERS['java'].set_language(JAVA_LANGUAGE)
PARSERS['kt'].set_language(KOTLIN_LANGUAGE)
PARSERS['js'].set_language(JAVASCRIPT_LANGUAGE)
# PARSERS['ts'].set_language(TYPESCRIPT_LANGUAGE)
PARSERS['py'].set_language(PYTHON_LANGUAGE)
PARSERS['cpp'].set_language(CPP_LANGUAGE)
PARSERS['c'].set_language(C_LANGUAGE)
# PARSERS['cs'].set_language(C_SHARP_LANGUAGE)
PARSERS['php'].set_language(PHP_LANGUAGE)
PARSERS['go'].set_language(GO_LANGUAGE)
PARSERS['rs'].set_language(RUST_LANGUAGE)
# PARSERS['swift'].set_language(SWIFT_LANGUAGE)
# PARSERS['scala'].set_language(SCALA_LANGUAGE)
PARSERS['rb'].set_language(RUBY_LANGUAGE)


@dataclass
class Method:
    name: str
    body: str


def get_methods_from_file(filepath: str) -> List[Method]:
    extension = filepath.split('.')[-1]
    parser = PARSERS[extension]
    content = open(filepath, 'r').read()
    tree = parser.parse(bytes(content, 'utf8'))

    methods = []
    for node in tree.root_node.children:
        if node.type == 'function_definition':
            start = node.start_point
            end = node.end_point
            method_code = content.splitlines()[start[0]:end[0] + 1]
            method_code = '\n'.join(method_code)
            methods.append(
                Method(node.children[1].text.decode(), method_code.replace(node.children[1].text.decode(), '<extra_id_0>', 1)))

    return methods


class MethodNameDataset(Dataset):

    def __init__(self, files_dir: str):
        self.methods = []
        filenames = get_code_filenames_from_directory(files_dir)
        for filename in tqdm(filenames):
            try:
                methods = get_methods_from_file(filename)
                self.methods.extend(methods)
            except Exception as e:
                print(f'Error parsing {filename}\n')
                print(e, '\n')

    def __len__(self):
        return len(self.methods)

    def __getitem__(self, idx):
        return self.methods[idx].body, self.methods[idx].name

if __name__ == '__main__':
    dataset = MethodNameDataset('/home/xram/Desktop/intellij-community')
    print(len(dataset))
    print(dataset[0][1])