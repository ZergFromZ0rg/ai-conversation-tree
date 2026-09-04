import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

const emptyGraph = { nodes: [], edges: [] };

/**
 * Owns one conversation's turns + graph and the actions that mutate them.
 * `status` is one of: "idle" | "loading" | "sending" | "analyzing" | "error".
 * `pendingUserText` holds an optimistic user message while a turn is in flight.
 */
export function useConversation(conversationId) {
  const [turns, setTurns] = useState([]);
  const [graph, setGraph] = useState(emptyGraph);
  const [status, setStatus] = useState("idle");
  const [pendingUserText, setPendingUserText] = useState(null);

  const refresh = useCallback(async () => {
    if (!conversationId) {
      setTurns([]);
      setGraph(emptyGraph);
      setStatus("idle");
      return;
    }
    setStatus("loading");
    try {
      const [turnsPayload, graphPayload] = await Promise.all([
        api.getTurns(conversationId),
        api.getGraph(conversationId),
      ]);
      setTurns(turnsPayload.turns);
      setGraph({ nodes: graphPayload.nodes, edges: graphPayload.edges });
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }, [conversationId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const sendTurn = useCallback(
    async (rawText, model) => {
      const userText = rawText.trim();
      if (!conversationId || !userText) {
        return;
      }
      setPendingUserText(userText);
      setStatus("sending");
      try {
        await api.sendTurn(conversationId, userText, model);
        setPendingUserText(null);
        await refresh();
      } catch {
        setPendingUserText(null);
        setStatus("error");
      }
    },
    [conversationId, refresh],
  );

  const analyze = useCallback(async () => {
    if (!conversationId) {
      return;
    }
    setStatus("analyzing");
    try {
      await api.analyze(conversationId);
      await refresh();
    } catch {
      setStatus("error");
    }
  }, [conversationId, refresh]);

  return { turns, graph, status, pendingUserText, sendTurn, analyze, refresh };
}
