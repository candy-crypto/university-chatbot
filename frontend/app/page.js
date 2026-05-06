"use client";
import { useState, useRef, useEffect } from "react";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

export default function Home() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [department, setDepartment] = useState("cs");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  }, [messages, loading]);

  async function sendMessage() {
    if (!message.trim()) return;
    setLoading(true);   // Starts the thinking
    setResult(null);    // Optional, it clears previous results
    setMessages((prev) => [...prev, { role: "user", text: message }]);
    setMessage("");
    
    try {
      // Commented actual code to test dummy code to see UI Interactions.
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
      setMessages((prev) => [...prev, { role: "bot", text: data.answer }]);
      setResult(data);

    } catch (err) {
      console.error(err);
      setResult({
        answer: "Could not connect to backend.",
        sources: [],
        chunks: [],
        prompt_context: "",
      });
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Unable to connect to the backend." }
      ]);
    } finally {
      setLoading(false);
    }
    
  }

return (
<div style={{ padding: "20px", fontFamily: "Arial" }}>

  <div className="header-bar">
    <h1 className="h1-title">New Mexico State University - Computer Science Department Chatbot</h1>
    <h1 className="h1-title">- UI & Bot Version: 0.5 -</h1>
  </div>

  <br />

  <div className="chat-window">
    <ul className="chatbox">
      {messages.length === 0 && !loading && (
        <li className="message bot">
          <div className="bubble">
            Welcome! Ask me anything about the Computer Science Department!
          </div>
        </li>
      )}

      {messages.map((msg, i) => (
        <li key={i} className={`message ${msg.role}`}>
          <div className="bubble">
            <div>{msg.text}</div>
          </div>
        </li>
      ))}

      {loading && (
        <li className="message bot">
          <div className="bubble typing">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </li>
      )}
      <div ref={messagesEndRef} />
    </ul>

    <div className="chat-input-container">
      <div className="input-wrapper">
        <textarea
          className = "chat-input"
          placeholder = "Ask a question..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
          }
                  
          }}
        />
        <button className="send-button" onClick={sendMessage} disabled={loading}>
          ➤
        </button>
      </div>
    </div>

    <div className="chat-disclaimer">
      AI can make mistakes, use responsibly.
    </div>
    </div>
</div>
  );
}
