import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, FileText, Image as ImageIcon, X } from 'lucide-react';
import MedicalCard from './MedicalCard';
import './App.css';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pdfs, setPdfs] = useState([]);
  const [image, setImage] = useState(null);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() && pdfs.length === 0 && !image) return;

    if (pdfs.length > 0 || image) {
      await handleUpload();
      return;
    }

    const questionText = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: questionText }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText })
      });
      const data = await response.json();

      if (!response.ok) {
        setMessages(prev => [...prev, { role: 'assistant', text: data.error || 'Something went wrong.', isError: true }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', data }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Network error. Please try again.', isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async () => {
    setMessages(prev => [...prev, { role: 'user', text: `Uploaded ${pdfs.length} PDF(s) and ${image ? '1 Image' : '0 Images'}.${input ? `\nQuestion: ${input}` : ''}` }]);
    setIsLoading(true);

    const formData = new FormData();
    pdfs.forEach(pdf => formData.append('pdfs', pdf));
    if (image) formData.append('image', image);

    setPdfs([]);
    setImage(null);
    setInput('');

    if (fileInputRef.current) fileInputRef.current.value = '';

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (!response.ok) {
        setMessages(prev => [...prev, { role: 'assistant', text: data.error || 'Upload failed.', isError: true }]);
      } else if (data.status) {
        setMessages(prev => [...prev, { role: 'assistant', text: data.status }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', data }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Network error. Please try again.', isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    const newPdfs = files.filter(f => f.type === 'application/pdf');
    const newImages = files.filter(f => f.type.startsWith('image/'));

    if (newPdfs.length) setPdfs(prev => [...prev, ...newPdfs]);
    if (newImages.length) setImage(newImages[0]);
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="brand-title">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
          CareGraph AI
        </div>
        <div className="brand-subtitle">Clinical decision support with guideline-backed answers</div>
      </header>

      <div className="chat-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
            <h2>How can I assist you today?</h2>
            <p>Ask a medical question, upload clinical guidelines (PDFs), or analyze lab results (Images).</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={msg.role === 'user' ? 'msg-user' : 'msg-assistant-wrapper'}>
              {msg.role === 'assistant' && (
                <div className="ai-avatar">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                </div>
              )}
              {msg.role === 'assistant' ? (
                msg.isError ? (
                  <div className="ai-message-error">{msg.text}</div>
                ) : msg.data ? (
                  <MedicalCard data={msg.data} />
                ) : (
                  <div className="ai-message-content">{msg.text}</div>
                )
              ) : (
                <div>{msg.text}</div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="msg-assistant-wrapper">
            <div className="ai-avatar">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
            </div>
            <div className="typing-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="command-bar-wrapper">
        <div className="command-bar">
          {(pdfs.length > 0 || image) && (
            <div className="attachments-preview">
              {pdfs.map((pdf, i) => (
                <div key={`pdf-${i}`} className="attachment-chip">
                  <FileText size={14} />
                  {pdf.name}
                  <button onClick={() => setPdfs(pdfs.filter((_, idx) => idx !== i))}><X size={14} /></button>
                </div>
              ))}
              {image && (
                <div className="attachment-chip">
                  <ImageIcon size={14} />
                  {image.name}
                  <button onClick={() => setImage(null)}><X size={14} /></button>
                </div>
              )}
            </div>
          )}

          <div className="input-row">
            <button className="icon-button" onClick={() => fileInputRef.current?.click()} title="Upload PDF or Lab Image">
              <Paperclip size={20} />
            </button>
            <input
              type="file"
              multiple
              className="file-input-hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,image/*"
            />

            <textarea
              className="chat-input"
              rows={1}
              placeholder={pdfs.length || image ? "Add a message (optional) or click Send to upload..." : "Ask a medical question..."}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />

            <div className="action-buttons">
              <button
                className="send-button"
                onClick={handleSend}
                disabled={isLoading || (!input.trim() && !pdfs.length && !image)}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
