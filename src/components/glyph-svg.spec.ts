import { describe, expect, test } from "bun:test";
import { 图形盒子, type 矢量笔画数据 } from "hanzi-chai";
import { glyphToSvgMarkup, strokeToSvgPath } from "./glyph-svg";
import { glyphLeafStrokeIds } from "./glyph-svg";

describe("glyph SVG rendering", () => {
  const strokes: 矢量笔画数据[] = [
    {
      feature: "横撇",
      start: [20, 15],
      curveList: [
        { command: "h", parameterList: [60] },
        { command: "c", parameterList: [-6, 30, -30, 60, -72, 78] },
      ],
    },
    {
      feature: "捺",
      start: [20, 15],
      curveList: [{ command: "c", parameterList: [6, 30, 30, 60, 72, 78] }],
    },
  ];

  test("turns repository strokes into the same SVG paths used by GlyphView", () => {
    expect(strokeToSvgPath(strokes[0]!, 0, strokes)).toBe(
      "M20 15 h60 c-6 30 -30 60 -72 78",
    );
  });

  test("renders a standalone SVG document for command-line evidence", () => {
    const svg = glyphToSvgMarkup(图形盒子.从笔画列表构建(strokes));

    expect(svg).toContain('viewBox="0 0 100 100"');
    expect(svg).toContain('stroke-width="7"');
    expect(svg).toContain('stroke-linecap="square"');
    expect(svg).toContain('d="M20 15 h60 c-6 30 -30 60 -72 78"');
  });

  test("can render thin review strokes with red stroke-boundary points", () => {
    const svg = glyphToSvgMarkup(
      图形盒子.从笔画列表构建(strokes),
      false,
      { strokeWidthScale: 0.5, showStrokeBoundaryPoints: true },
    );

    expect(svg).toContain('stroke-width="3.5"');
    expect(svg).toContain('<circle cx="20" cy="15" r="1.5" fill="red"/>');
    expect(svg).toContain('<circle cx="80" cy="15" r="1.5" fill="red"/>');
    expect(svg).toContain('<circle cx="8" cy="93" r="1.5" fill="red"/>');
    expect(svg.match(/<circle/g)).toHaveLength(5);
    // Cubic control points are editor handles, not stroke boundaries.
    expect(svg).not.toContain('<circle cx="74" cy="45"');
    expect(svg).not.toContain('<circle cx="50" cy="75"');
  });

  test("can color strokes by their recursively resolved leaf component", () => {
    const svg = glyphToSvgMarkup(
      图形盒子.从笔画列表构建(strokes),
      false,
      { strokeColors: ["#2563eb", "#16a34a"] },
    );

    expect(svg).toContain('stroke="#2563eb"');
    expect(svg).toContain('stroke="#16a34a"');
  });

  test("can omit non-target strokes without changing the SVG coordinate box", () => {
    const svg = glyphToSvgMarkup(
      图形盒子.从笔画列表构建([
        { feature: "横", start: [10, 20], curveList: [{ command: "h", parameterList: [80] }] },
        { feature: "竖", start: [50, 10], curveList: [{ command: "v", parameterList: [80] }] },
      ]),
      false,
      { strokeVisibility: [true, false] },
    );

    expect(svg).toContain('d="M10 20 h80"');
    expect(svg).not.toContain('d="M50 10 v80"');
    expect(svg.match(/<path/g)).toHaveLength(1);
  });

  test("tracks leaf ownership through compound stroke order", () => {
    const glyphs = [
      {
        id: 1,
        type: "component" as const,
        strokes: strokes.slice(0, 1),
        operator: undefined,
        references: undefined,
        ambiguous: false,
      },
      {
        id: 2,
        type: "component" as const,
        strokes: strokes,
        operator: undefined,
        references: undefined,
        ambiguous: false,
      },
      {
        id: 3,
        type: "compound" as const,
        operator: "⿰" as const,
        references: [{ id: 1 }, { id: 2 }],
        strokes: [
          { index: 1, from: 1, to: 1 },
          { index: 0 },
          { index: 1, from: 0, to: 0 },
        ],
        ambiguous: false,
      },
    ];

    expect(glyphLeafStrokeIds(3, new Map(glyphs.map((x) => [x.id, x])))).toEqual([
      2,
      1,
      2,
    ]);
  });
});
