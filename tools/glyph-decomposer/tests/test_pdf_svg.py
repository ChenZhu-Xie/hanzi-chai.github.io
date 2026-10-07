import pytest

from glyph_decomposer.pdf_svg import PdfCell, extract_cell_geometry


def test_extract_cell_selects_vector_symbol_over_nearby_text():
    page = """\
    <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:xlink="http://www.w3.org/1999/xlink">
      <defs>
        <symbol id="small"><path d="M0 0 L1 0 L1 -1 L0 -1 Z"/></symbol>
        <symbol id="glyph"><path d="M0 0 L10 0 L10 -10 L0 -10 Z"/></symbol>
      </defs>
      <use xlink:href="#small" x="11" y="21"/>
      <use xlink:href="#glyph" x="10" y="20"/>
    </svg>
    """
    cell = PdfCell(unicode=0x4E00, source="T", page=1, bbox=(9, 9, 21, 21))

    geometry = extract_cell_geometry(page, cell)

    assert geometry.is_valid
    assert geometry.area == pytest.approx(96 * 96)
    assert geometry.bounds == (2.0, 2.0, 98.0, 98.0)
