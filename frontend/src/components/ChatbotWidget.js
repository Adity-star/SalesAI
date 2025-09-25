function ChatbotWidget() {
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <button className="chatbot-button" onClick={() => setOpen(true)}>
        Chat with AI
      </button>

      {open && (
        <div className="chatbot-modal">
          <button className="close-btn" onClick={() => setOpen(false)}>X</button>
          {/* Your chatbot UI here */}
          <Chatbot />
        </div>
      )}
    </>
  );
}
