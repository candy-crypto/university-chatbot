"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

export default function Home() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
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
      setResponse(data.answer);
    } catch (err) {
      console.error(err);
      setResponse("Could not connect to backend.");
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
      <div>{response}</div>
    </div>
  );
}
