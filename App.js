import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import your CSS file

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(""); // For upload feedback

  const uploadFile = async () => {
    if (!file) {
      setUploadMessage("Please select a file.");
      return;
    }

    setLoading(true);
    setUploadMessage("Uploading..."); // Provide feedback during upload
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadMessage(response.data.message); // Display server message
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadMessage("Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question) return;

    setLoading(true); // Show loading indicator
    setAnswer(""); // Clear previous answer
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/ask",
        { question },
        { headers: { "Content-Type": "application/json" } }
      );
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
      setAnswer("Error fetching answer.");
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className="app-container"> {/* Use a CSS class */}
      <h1>Document Q&A System</h1>

      <div className="upload-section">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} id="fileInput" />
        <label htmlFor="fileInput" className="upload-button">Choose File</label> {/* Styled label */}
        <button onClick={uploadFile} disabled={loading}>
          {loading ? "Uploading..." : "Upload"}
        </button>
        {uploadMessage && <p className="upload-message">{uploadMessage}</p>} {/* Display message */}

      </div>

      <div className="question-section">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question"
          className="question-input"
        />
        <button onClick={askQuestion} disabled={loading} className="ask-button">
          {loading ? "Asking..." : "Ask"}
        </button>
      </div>

      <div className="answer-section">
        <h2>Answer:</h2>
        {loading && <div className="loading-spinner"></div>} {/* Loading spinner */}
        {answer && <p className="answer-text">{answer}</p>} {/* Display the answer */}
      </div>
    </div>
  );
}

export default App;