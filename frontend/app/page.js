"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [department, setDepartment] = useState("cs");

  async function sendMessage() {
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: message,
          department_id: department
        })
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Request failed: ${res.status} ${text}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({
        answer: "Could not connect to backend.",
        sources: [],
        chunks: [],
        prompt_context: "",
      });
    }
  }

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>University Chatbot</h1>

      <h3>Select Department</h3>

      <button onClick={() => setDepartment("cs")}>Computer Science</button>
      <button onClick={() => setDepartment("math")}>Math</button>
      <button onClick={() => setDepartment("unknown")}>Not Sure</button>
      <p>Selected department: {department}</p>

      <br /><br />

      <input
        type="text"
        placeholder="Ask a question..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        style={{ width: "600px" }}
      />

      <button onClick={sendMessage}>Send</button>

      <br /><br />

      <h3>Response</h3>
      <div>{result?.answer || ""}</div>

      {result?.sources?.length ? (
        <>
          <h3>Sources</h3>
          <ul>
            {result.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </>
      ) : null}

      {result ? (
        <>
          <h3>Retrieved Chunks</h3>
          <p>These are the chunks retrieved and then passed into the prompt context.</p>
          <div style={{ display: "grid", gap: "16px" }}>
            {result.chunks?.map((chunk) => (
              <div
                key={chunk.chunk_id}
                style={{
                  border: "1px solid #ccc",
                  borderRadius: "8px",
                  padding: "16px",
                  background: "#fafafa",
                }}
              >
                <div><strong>Rank:</strong> {chunk.rank ?? "-"}</div>
                <div><strong>Heading:</strong> {chunk.heading || "-"}</div>
                <div><strong>Chunk ID:</strong> {chunk.chunk_id || "-"}</div>
                <div><strong>Chunk Type:</strong> {chunk.chunk_type || "-"}</div>
                <div><strong>Content Source:</strong> {chunk.content_source || "-"}</div>
                <div><strong>Source:</strong> {chunk.source || "-"}</div>
                <div><strong>Course Code:</strong> {chunk.course_code || "-"}</div>
                <div><strong>Referenced Courses:</strong> {(chunk.referenced_courses || []).join(", ") || "-"}</div>
                <div><strong>Hybrid Score:</strong> {chunk.hybrid_score ?? "-"}</div>
                <div><strong>Metadata Boost:</strong> {chunk.metadata_boost ?? "-"}</div>
                <div><strong>Final Score:</strong> {chunk.final_score ?? "-"}</div>
                <div><strong>Text:</strong></div>
                <pre style={{ whiteSpace: "pre-wrap", overflowX: "auto" }}>{chunk.text || ""}</pre>
              </div>
            ))}
          </div>

          <h3>Prompt Context</h3>
          <p>This is the exact context block sent to the answer model.</p>
          <pre style={{ whiteSpace: "pre-wrap", overflowX: "auto", background: "#f4f4f4", padding: "16px", borderRadius: "8px" }}>
            {result.prompt_context || ""}
          </pre>
        </>
      ) : null}
    </div>
  );
}
