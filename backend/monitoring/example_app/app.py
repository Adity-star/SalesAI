from flask import Flask, Response
from prometheus_client import Counter, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

app = Flask(__name__)

registry = CollectorRegistry()
REQUESTS = Counter('example_app_requests_total', 'Total HTTP requests', registry=registry)
ERRORS = Counter('example_app_errors_total', 'Total errors', registry=registry)

@app.route('/')
def index():
    REQUESTS.inc()
    return "ok"

@app.route('/error')
def error():
    REQUESTS.inc()
    ERRORS.inc()
    return "error", 500

@app.route('/metrics')
def metrics():
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
