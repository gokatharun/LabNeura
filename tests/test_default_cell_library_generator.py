import ast
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "cmake" / "GenerateEmbeddedLiberty.cmake"
TEMPLATE = (
    ROOT / "src" / "labneura" / "eda" / "parser" / "liberty"
    / "default_cell_library.h.in"
)
LIBERTY = ROOT / "assets" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
LITERAL = re.compile(
    r'^\s*std::string_view\("((?:\\.|[^"\\])*)", ([0-9]+)\),$'
)


def generate(tmp_path, data):
    source = tmp_path / "input.lib"
    output = tmp_path / "default_cell_library.h"
    source.write_bytes(data)
    subprocess.run(
        [
            "cmake",
            f"-DLABNEURA_EMBED_INPUT={source}",
            f"-DLABNEURA_EMBED_TEMPLATE={TEMPLATE}",
            f"-DLABNEURA_EMBED_OUTPUT={output}",
            "-P",
            str(GENERATOR),
        ],
        check=True,
        cwd=ROOT,
    )
    chunks = []
    literal_spellings = []
    declared_lengths = []
    for line in output.read_text(encoding="utf-8").splitlines():
        match = LITERAL.match(line)
        if match:
            literal_spellings.append(match.group(1).encode())
            chunks.append(ast.literal_eval(f'"{match.group(1)}"').encode())
            declared_lengths.append(int(match.group(2)))
    assert list(map(len, chunks)) == declared_lengths
    return output, chunks, literal_spellings


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b'library (tiny) {\n  comment : "quote \\\\ slash";\n}\n',
        b"a" * 4095,
        b"b" * 4096,
        b"c" * 4097,
    ],
)
def test_embedded_liberty_generator_handles_boundaries_and_escaping(tmp_path, data):
    output, chunks, literal_spellings = generate(tmp_path, data)
    assert b"".join(chunks) == data
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert all(len(spelling) <= 8192 for spelling in literal_spellings)
    expected_count = 0 if not data else (len(data) + 4095) // 4096
    assert f"std::array<std::string_view, {expected_count}>" in output.read_text()


def test_embedded_default_library_is_byte_exact_and_bounded(tmp_path):
    source = LIBERTY.read_bytes()
    _, chunks, literal_spellings = generate(tmp_path, source)
    assert b"".join(chunks) == source
    assert chunks and max(map(len, chunks)) <= 4096
    assert max(map(len, literal_spellings)) <= 8192
