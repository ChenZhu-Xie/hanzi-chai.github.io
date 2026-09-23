import { describe, expect, test } from "bun:test";
import { 图形盒子, type 矢量笔画数据 } from "hanzi-chai";
import { glyphToSvgMarkup, strokeToSvgPath } from "./glyph-svg";

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
});
