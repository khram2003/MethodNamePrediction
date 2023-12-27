import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tree_sitter import Language, Parser
from tqdm import tqdm

Language.build_library(
    'build/my-languages.so',

    [
        # 'tree-sitter-kotlin',
        'tree-sitter-java',
        'tree-sitter-javascript',
        # '../tree-sitter-typescript',
        'tree-sitter-python',
        'tree-sitter-cpp',
        'tree-sitter-c',
        # '../tree-sitter-c-sharp',
        'tree-sitter-php',
        'tree-sitter-go',
        # 'tree-sitter-rust',
        # 'tree-sitter-swift',
        # '../tree-sitter-scala',
        'tree-sitter-ruby'
    ]
)

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
# KOTLIN_LANGUAGE = Language('build/my-languages.so', 'kotlin')
JAVASCRIPT_LANGUAGE = Language('build/my-languages.so', 'javascript')
# TYPESCRIPT_LANGUAGE = Language('build/my-languages.so', 'typescript')
PYTHON_LANGUAGE = Language('build/my-languages.so', 'python')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')
C_LANGUAGE = Language('build/my-languages.so', 'c')
# C_SHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
PHP_LANGUAGE = Language('build/my-languages.so', 'php')
GO_LANGUAGE = Language('build/my-languages.so', 'go')
# RUST_LANGUAGE = Language('build/my-languages.so', 'rust')
# SWIFT_LANGUAGE = Language('build/my-languages.so', 'swift')
# SCALA_LANGUAGE = Language('build/my-languages.so', 'scala')
RUBY_LANGUAGE = Language('build/my-languages.so', 'ruby')

PARSERS = {
    'java': Parser(),
    # 'kt': Parser(),
    'js': Parser(),
    # 'ts': Parser(),
    'py': Parser(),
    'cpp': Parser(),
    'c': Parser(),
    # 'cs': Parser(),
    'php': Parser(),
    'go': Parser(),
    # 'rs': Parser(),
    # 'swift': Parser(),
    # 'scala': Parser(),
    'rb': Parser()
}

PARSERS['java'].set_language(JAVA_LANGUAGE)
# PARSERS['kt'].set_language(KOTLIN_LANGUAGE)
PARSERS['js'].set_language(JAVASCRIPT_LANGUAGE)
# PARSERS['ts'].set_language(TYPESCRIPT_LANGUAGE)
PARSERS['py'].set_language(PYTHON_LANGUAGE)
PARSERS['cpp'].set_language(CPP_LANGUAGE)
PARSERS['c'].set_language(C_LANGUAGE)
# PARSERS['cs'].set_language(C_SHARP_LANGUAGE)
PARSERS['php'].set_language(PHP_LANGUAGE)
PARSERS['go'].set_language(GO_LANGUAGE)
# PARSERS['rs'].set_language(RUST_LANGUAGE)
# PARSERS['swift'].set_language(SWIFT_LANGUAGE)
# PARSERS['scala'].set_language(SCALA_LANGUAGE)
PARSERS['rb'].set_language(RUBY_LANGUAGE)


@dataclass
class Method:
    name: str
    body: str


def get_code_filenames_from_directory(directory: str) -> List[str]:
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java") or file.endswith(".js") or file.endswith(
                    ".py") or file.endswith(".cpp") \
                    or file.endswith(".c") or file.endswith(".php") or file.endswith(
                ".go") \
                    or file.endswith('rb'):
                filenames.append(os.path.join(root, file))
    return filenames


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
                Method(node.children[1].text.decode(),
                       method_code.replace(node.children[1].text.decode(), '<extra_id_0>', 1)))

    return methods


def get_all_methods(directory: str) -> List[Method]:
    filenames = get_code_filenames_from_directory(directory)
    methods = []
    for filename in tqdm(filenames):
        try:
            methods_ = get_methods_from_file(filename)
            methods.extend(methods_)
        except Exception as e:
            print(f'Error parsing {filename}\n')
            print(e, '\n')
    return methods


def get_methods_split(directory: str) -> Tuple[List[Method], List[Method], List[Method]]:
    methods = get_all_methods(directory)
    random.shuffle(methods)
    train_methods, val_test_methods = train_test_split(methods, test_size=0.2, random_state=42)
    val_methods, test_methods = train_test_split(val_test_methods, test_size=0.25, random_state=42)
    return train_methods, val_methods, test_methods
