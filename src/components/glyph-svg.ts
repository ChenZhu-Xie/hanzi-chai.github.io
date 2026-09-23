import type {
  N6,
  图形盒子,
  向量,
  基本字形数据,
  矢量笔画数据,
  笔画名称,
  绘制,
} from "hanzi-chai";
import { isEqual } from "lodash-es";

const drawLength = ({ command, parameterList }: 绘制) => {
  if (command === "h" || command === "v") {
    return parameterList[0];
  }
  if (command === "a") {
    return 0;
  }
  const [_x1, _y1, _x2, _y2, x3, y3] = parameterList as N6;
  return Math.sqrt(x3 * x3 + y3 * y3);
};

const curvedStrokeFeatures: 笔画名称[] = [
  "横折弯",
  "横折弯钩",
  "竖弯",
  "竖弯钩",
];
const secondHookFeatures: 笔画名称[] = [
  "竖钩",
  "横折钩",
  "竖折折钩",
  "横折折折钩",
  "弯钩",
  "横撇弯钩",
];
const thirdHookFeatures: 笔画名称[] = ["斜钩", "横斜钩", "撇钩"];
const fourthHookFeatures: 笔画名称[] = ["竖弯钩", "横折弯钩"];

/** Build the exact path used by GlyphView without depending on React or DOM. */
export function strokeToSvgPath(
  { start, feature, curveList }: 矢量笔画数据,
  strokeIndex: number,
  strokes: 矢量笔画数据[],
) {
  const commands: string[] = [`M${start.join(" ")}`];
  const segmentCount = curveList.length;
  const referenceLength = drawLength(curveList.at(-1)!);
  const beautifyCurve = curvedStrokeFeatures.includes(feature);
  const curveRadius = beautifyCurve
    ? Math.min(
        Math.min(
          Math.abs(curveList[segmentCount - 2]!.parameterList[0]),
          Math.abs(curveList[segmentCount - 1]!.parameterList[0]),
        ) * 0.3,
        6,
      )
    : 0;
  const firstHook = feature === "横钩";
  const secondHookTangent = 0.1;
  const secondHook = secondHookFeatures.includes(feature);
  const secondHookRadius = secondHook ? Math.min(referenceLength * 0.2, 5) : 0;
  const secondHookScale = Math.sqrt(1 + secondHookTangent ** 2);
  const secondHookDx =
    -secondHookRadius * (1 + secondHookTangent / secondHookScale);
  const secondHookDy = secondHookRadius / secondHookScale;
  const thirdHook = thirdHookFeatures.includes(feature);
  const fourthHook = fourthHookFeatures.includes(feature);
  const fourthHookRadius = fourthHook ? Math.min(referenceLength * 0.2, 5) : 0;

  for (const [index, { command, parameterList }] of curveList.entries()) {
    const last = index === segmentCount - 1;
    const penultimate = index === segmentCount - 2;
    const sign = (value: number) => (value >= 0 ? 1 : -1);

    if (beautifyCurve && penultimate) {
      commands.push(
        `v ${parameterList[0] - sign(parameterList[0]) * curveRadius}`,
      );
      commands.push(
        `a ${curveRadius} ${curveRadius} 0 0 0 ${curveRadius} ${curveRadius}`,
      );
    } else if (last && command === "h" && (beautifyCurve || fourthHook)) {
      const direction = sign(parameterList[0]);
      const shortening =
        (beautifyCurve ? curveRadius : 0) + (fourthHook ? fourthHookRadius : 0);
      commands.push(`h ${parameterList[0] - direction * shortening}`);
      if (fourthHook) {
        commands.push(
          `a ${fourthHookRadius} ${fourthHookRadius} 0 0 0 ${direction * fourthHookRadius} ${-fourthHookRadius}`,
        );
      }
    } else if (last && command === "v" && secondHook) {
      const direction = sign(parameterList[0]);
      const shortened = parameterList[0] - direction * secondHookRadius;
      if (shortened * direction > 0) commands.push(`v ${shortened}`);
      commands.push(
        `a ${secondHookRadius} ${secondHookRadius} 0 0 1 ${secondHookDx} ${direction * secondHookDy}`,
      );
    } else if (last && command === "h" && feature.endsWith("提")) {
      commands.push(`l ${parameterList[0]} ${-0.15 * parameterList[0]}`);
    } else if (command === "a") {
      commands.push("a 50,50 0 1,1 0,100");
      commands.push("a 50,50 0 1,1 0,-100");
    } else {
      commands.push(command.replace("z", "c") + parameterList.join(" "));
    }
  }

  const hookLength = Math.min(5 + referenceLength * 0.25, 15);
  if (firstHook) {
    let previousVerticalDot = false;
    let previousLength = 0;
    if (strokeIndex > 0) {
      const previous = strokes[strokeIndex - 1]!;
      if (previous.feature === "点" && isEqual(previous.start, start)) {
        previousVerticalDot = true;
        previousLength = drawLength(previous.curveList.at(-1)!);
      }
    }
    if (previousVerticalDot) {
      commands.push(`l ${0} ${previousLength}`);
    } else {
      commands.push(`l ${-hookLength} ${hookLength}`);
    }
  } else if (secondHook) {
    commands.push(`l ${-hookLength} ${-hookLength * secondHookTangent}`);
  } else if (thirdHook) {
    commands.push(`l ${hookLength * 0.3} ${-hookLength}`);
  } else if (fourthHook) {
    commands.push(`l 0 ${-hookLength}`);
  }

  return commands.join(" ");
}

/** Render a glyph as a standalone SVG document for browser and CLI parity. */
export interface GlyphSvgOptions {
  /** Scale the normal repository stroke width without changing geometry. */
  strokeWidthScale?: number;
  /** Mark every stroke start and segment boundary for human topology review. */
  showStrokePoints?: boolean;
  /** Optional per-stroke colors used by leaf-aware human review panels. */
  strokeColors?: string[];
}

function strokeBoundaryPoints({
  start,
  curveList,
}: 矢量笔画数据): 向量[] {
  const points: 向量[] = [[...start]];
  const current: 向量 = [...start];
  for (const { command, parameterList } of curveList) {
    if (command === "h") current[0] += parameterList[0];
    else if (command === "v") current[1] += parameterList[0];
    else if (command !== "a") {
      const values = parameterList as N6;
      current[0] += values[4];
      current[1] += values[5];
    }
    points.push([...current]);
  }
  return points;
}

export function glyphToSvgMarkup(
  glyph: 图形盒子,
  displayMode = false,
  options: GlyphSvgOptions = {},
) {
  const strokes = glyph.获取笔画列表();
  const { strokeWidth, viewBox } = glyph.确定笔画粗细和视窗(displayMode);
  const serializedStrokeWidth = Number(
    (strokeWidth * (options.strokeWidthScale ?? 1)).toPrecision(12),
  );
  const paths = strokes
    .map(
      (stroke, index) =>
        `<path d="${strokeToSvgPath(stroke, index, strokes)}" stroke="${options.strokeColors?.[index] ?? "black"}" stroke-width="${serializedStrokeWidth}" fill="none" stroke-linecap="square"/>`,
    )
    .join("");
  const points = options.showStrokePoints
    ? strokes
        .flatMap(strokeBoundaryPoints)
        .map(
          ([x, y]) =>
            `<circle cx="${x}" cy="${y}" r="1.5" fill="red"/>`,
        )
        .join("")
    : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" width="100" height="100">${paths}${points}</svg>`;
}

/** Resolve each flattened stroke back to its terminal component ID. */
export function glyphLeafStrokeIds(
  id: number,
  glyphById: ReadonlyMap<number, 基本字形数据>,
  seen = new Set<number>(),
): number[] {
  if (seen.has(id)) throw new Error(`字形 ${id} 存在循环引用`);
  const glyph = glyphById.get(id);
  if (!glyph) throw new Error(`字形 ${id} 不存在`);
  if (glyph.type === "component") return glyph.strokes.map(() => id);

  const nextSeen = new Set(seen).add(id);
  const parts = glyph.references.map(({ id: referenceId }) =>
    glyphLeafStrokeIds(referenceId, glyphById, nextSeen),
  );
  if (!glyph.strokes?.length) return parts.flat();

  return glyph.strokes.flatMap(({ index, from, to }) => {
    const part = parts[index] ?? [];
    return part.slice(from ?? 0, (to ?? part.length - 1) + 1);
  });
}
