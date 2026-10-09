import pytest

from glyph_decomposer.pdf_svg import PdfCell, extract_cell_geometry, parse_cells


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


def test_parse_cells_indexes_sources_and_glyph_words_once_per_page(tmp_path):
    bbox = tmp_path / "bbox.xml"
    bbox.write_text(
        """<doc><page width="600" height="800">
        <word xMin="90" yMin="100" xMax="110" yMax="110">4E00</word>
        <word xMin="150" yMin="118" xMax="160" yMax="128">G0-0000</word>
        <word xMin="145" yMin="114" xMax="165" yMax="130">一</word>
        <word xMin="175" yMin="118" xMax="185" yMax="128">T1-0000</word>
        <word xMin="170" yMin="114" xMax="190" yMax="130">一</word>
        <word xMin="90" yMin="134" xMax="110" yMax="144">4E01</word>
        <word xMin="150" yMin="152" xMax="160" yMax="162">KP0-0000</word>
        <word xMin="145" yMin="148" xMax="165" yMax="164">丁</word>
        </page></doc>""",
        encoding="utf-8",
    )

    cells = parse_cells(bbox)

    assert cells == [
        PdfCell(0x4E00, "G", 1, (145.0, 114.0, 165.0, 130.0)),
        PdfCell(0x4E00, "T", 1, (170.0, 114.0, 190.0, 130.0)),
        PdfCell(0x4E01, "N", 1, (145.0, 148.0, 165.0, 164.0)),
    ]
