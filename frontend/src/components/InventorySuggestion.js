function SignalsPanel({ signals }) {
  return (
    <section className="signals-panel">
      <h3>Alerts & Signals</h3>
      <ul>
        {signals.map((signal, idx) => (
          <li key={idx} className={`signal ${signal.level}`}>
            <strong>{signal.title}</strong>: {signal.message}
          </li>
        ))}
      </ul>
    </section>
  );
}
