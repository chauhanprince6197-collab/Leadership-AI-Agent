"""
main.py — Application entrypoint

Development:  python main.py
Production:   gunicorn "main:create_app()" --workers 2 --bind 0.0.0.0:5050
"""

import structlog
from app import create_app
from config.settings import settings

# Configure structured JSON logging (parses cleanly in Datadog/CloudWatch)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


def create_application():
    """Gunicorn factory — called as `gunicorn "main:create_application()"`"""
    return create_app()


if __name__ == "__main__":
    app = create_app()
    logger.info("starting_dev_server", host=settings.host, port=settings.port)
    # Flask dev server — single process, NOT for production
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        use_reloader=False,   # reloader causes double-init of vector store
    )

