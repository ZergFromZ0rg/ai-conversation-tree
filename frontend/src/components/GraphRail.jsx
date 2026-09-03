import { useEffect, useRef, useState } from "react";

export function GraphRail({ turnCount, threadCount, onOpen }) {
  const [bumped, setBumped] = useState(false);
  const previousCount = useRef(turnCount);

  useEffect(() => {
    if (turnCount > previousCount.current) {
      setBumped(true);
      const timer = setTimeout(() => setBumped(false), 600);
      previousCount.current = turnCount;
      return () => clearTimeout(timer);
    }
    previousCount.current = turnCount;
  }, [turnCount]);

  return (
    <button
      type="button"
      className={`graphRail${bumped ? " graphRailBump" : ""}`}
      onClick={onOpen}
      aria-label="Open conversation graph"
      title="Open conversation graph  (⌘/Ctrl+G)"
    >
      <span className="graphRailIcon" aria-hidden="true">🌳</span>
      <span className="graphRailCount">{turnCount}</span>
      <span className="graphRailLabel">{turnCount === 1 ? "turn" : "turns"}</span>
      {threadCount > 1 ? <span className="graphRailThreads">{threadCount} threads</span> : null}
    </button>
  );
}
