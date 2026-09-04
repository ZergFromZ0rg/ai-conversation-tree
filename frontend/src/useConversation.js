import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

const emptyGraph = { nodes: [], edges: [] };
const emptyConceptLinks = { concepts: [] };

/**
 * Owns one conversation's turns + graph and the actions that mutate them.
 * `status` is one of: "idle" | "loading" | "sending" | "analyzing" | "error".
 * `pendingUserText` holds an optimistic user message while a turn is in flight;
 * `streamingText` grows token by token alongside it as the reply streams in.
 */
export function useConversation(conversationId) {
  const [turns, setTurns] = useState([]);
  const [graph, setGraph] = useState(emptyGraph);
  const [conceptLinks, setConceptLinks] = useState(emptyConceptLinks);
  const [status, setStatus] = useState("idle");
  const [pendingUserText, setPendingUserText] = useState(null);
  const [streamingText, setStreamingText] = useState("");

  const refresh = useCallback(async () => {
    if (!conversationId) {
      setTurns([]);
      setGraph(emptyGraph);
      setConceptLinks(emptyConceptLinks);
      setStatus("idle");
      return;
    }
    setStatus("loading");
    try {
      const [turnsPayload, graphPayload, conceptLinksPayload] = await Promise.all([
        api.getTurns(conversationId),
        api.getGraph(conversationId),
        api.getConceptLinks(conversationId).catch(() => emptyConceptLinks),
      ]);
      setTurns(turnsPayload.turns);
      setGraph({ nodes: graphPayload.nodes, edges: graphPayload.edges });
      setConceptLinks({ concepts: conceptLinksPayload.concepts ?? [] });
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
      setStreamingText("");
      setStatus("sending");
      try {
        for await (const event of api.streamTurn(conversationId, userText, model)) {
          if (event.type === "delta") {
            setStreamingText((prev) => prev + event.text);
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
          // "done" carries the full persisted payload, but re-fetching below
          // (turns + graph + conceptLinks together) is simpler than
          // reconciling it in place and keeps one source of truth.
        }
        setPendingUserText(null);
        setStreamingText("");
        await refresh();
      } catch {
        setPendingUserText(null);
        setStreamingText("");
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

  return {
    turns,
    graph,
    conceptLinks,
    status,
    pendingUserText,
    streamingText,
    sendTurn,
    analyze,
    refresh,
  };
}
