import React, { useState } from 'react';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import './Chatbot.css'; // styles we'll add in step 2

const AdvancedChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hi! Ask me anything about sales forecasts or inventory.' }
  ]);
  const [input, setInput] = useState('');

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMessage = input.trim();
    setMessages(prev => [...prev, { from: 'user', text: userMessage }]);
    setInput('');

    // Simulated smart response
    let botReply = "I'm still learning. Could you rephrase?";
    const lower = userMessage.toLowerCase();

    if (lower.includes('forecast')) {
      botReply = "The forecast indicates rising demand for next week. 📈";
    } else if (lower.includes('inventory')) {
      botReply = "Inventory is sufficient to meet forecasted demand.";
    } else if (lower.includes('sku')) {
      botReply = "You can view SKU-level forecasts in the Forecast tab.";
    }

    setTimeout(() => {
      setMessages(prev => [...prev, { from: 'bot', text: botReply }]);
    }, 600);
  };

  return (
    <>
      <div className="chatbot-toggle-button" onClick={toggleChat}>
        {isOpen ? <CloseIcon /> : <ChatIcon />}
      </div>

      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <strong>SalesAI Assistant</strong>
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-msg ${msg.from}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chatbot-input-area">
            <input
              type="text"
              placeholder="Ask something..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
            />
            <button onClick={handleSend}>Send</button>
          </div>
        </div>
      )}
    </>
  );
};

export default AdvancedChatbot;
