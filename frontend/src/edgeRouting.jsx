import { BaseEdge } from "@xyflow/react";

// Renders the explicit path a caller's lane-routing computed (buildLanePath,
// or a caller's own equivalent); falls back to nothing if a caller ever hands
// this a malformed edge — never hit in practice, callers only use the
// "routed" edge type when they actually computed a path.
export function RoutedEdge({
  data,
  style,
  label,
  labelStyle,
  labelBgStyle,
  labelBgPadding,
  labelBgBorderRadius,
  markerStart,
  markerEnd,
}) {
  if (!data?.path) {
    return null;
  }
  return (
    <BaseEdge
      path={data.path}
      style={style}
      label={label}
      labelX={data.labelX}
      labelY={data.labelY}
      labelStyle={labelStyle}
      labelBgStyle={labelBgStyle}
      labelBgPadding={labelBgPadding}
      labelBgBorderRadius={labelBgBorderRadius}
      markerStart={markerStart}
      markerEnd={markerEnd}
    />
  );
}

export const edgeTypes = { routed: RoutedEdge };

// Builds an orthogonal path from one node's exit point to another's entry
// point, routed entirely through empty lanes so it never crosses a node card
// in between — unlike a direct line, which cuts straight through whatever
// sits between two far-apart nodes.
//
// sourceLaneY/targetLaneY are the y of the empty band each endpoint routes
// into right after leaving/before entering its node (a caller-computed
// row/rank gap). When they're the same lane, this is a simple 4-point path
// (exit, into the lane, across, into the target). When they differ, the path
// detours through `highwayX` — a fixed x past every node — to get from one
// lane to the other without cutting back across intervening node columns.
//
// `jitter` nudges the line itself (kept small — it still has to read as
// "roughly centered in the gap", not jammed against either node) so two
// routed edges sharing a lane don't draw exactly on top of each other.
// `labelOffset` only moves the label, slid along whichever segment the
// label sits on (the horizontal run for a same-lane path, the vertical
// highway run otherwise) — it can be much larger than `jitter` without
// making the line itself look off-center, which is what actually separates
// a label from another edge's label, or from a primary (non-routed) edge's
// own automatically-placed one that a small `jitter` can't move at all.
export function buildLanePath(
  sourceX,
  sourceExitY,
  sourceLaneY,
  targetX,
  targetEnterY,
  targetLaneY,
  highwayX,
  jitter = 0,
  labelOffset = 0,
) {
  const jitteredSourceLaneY = sourceLaneY + jitter;
  const jitteredTargetLaneY = targetLaneY + jitter;

  if (Math.abs(jitteredSourceLaneY - jitteredTargetLaneY) < 1) {
    return {
      d: [
        `M ${sourceX} ${sourceExitY}`,
        `L ${sourceX} ${jitteredSourceLaneY}`,
        `L ${targetX} ${jitteredTargetLaneY}`,
        `L ${targetX} ${targetEnterY}`,
      ].join(" "),
      labelX: (sourceX + targetX) / 2 + labelOffset,
      labelY: jitteredSourceLaneY,
    };
  }

  const jitteredHighwayX = highwayX + jitter;
  return {
    d: [
      `M ${sourceX} ${sourceExitY}`,
      `L ${sourceX} ${jitteredSourceLaneY}`,
      `L ${jitteredHighwayX} ${jitteredSourceLaneY}`,
      `L ${jitteredHighwayX} ${jitteredTargetLaneY}`,
      `L ${targetX} ${jitteredTargetLaneY}`,
      `L ${targetX} ${targetEnterY}`,
    ].join(" "),
    labelX: jitteredHighwayX,
    labelY: (jitteredSourceLaneY + jitteredTargetLaneY) / 2 + labelOffset,
  };
}
