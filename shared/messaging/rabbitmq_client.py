"""
RabbitMQ client — connection factory and channel management.

Provides reusable connection/channel creation with dead-letter exchange
configuration and separate queues for edge and server runtimes.
"""

import json
import time
from typing import Optional, Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel

from shared.utils.logging import get_logger

logger = get_logger("rabbitmq_client")


class RabbitMQClient:
    """Manages RabbitMQ connections, channel setup, and queue declarations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        user: str = "guest",
        password: str = "guest",
        heartbeat: int = 600,
        prefetch_count: int = 1,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.heartbeat = heartbeat
        self.prefetch_count = prefetch_count

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[BlockingChannel] = None

    def connect(self) -> BlockingChannel:
        """Establish connection and return a configured channel.

        Sets up the exchange, queues, and dead-letter infrastructure.
        """
        credentials = pika.PlainCredentials(self.user, self.password)
        params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=self.heartbeat,
            blocked_connection_timeout=300,
        )

        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=self.prefetch_count)

        logger.info(
            f"Connected to RabbitMQ at {self.host}:{self.port}",
            extra={"event": "rabbitmq_connected"},
        )

        return self._channel

    def setup_infrastructure(
        self,
        exchange: str = "ai.analysis",
        edge_queue: str = "ai.analysis.edge.requested",
        server_queue: str = "ai.analysis.server.requested",
        edge_image_queue: str = "ai.analysis.edge.image.requested",
        edge_video_queue: str = "ai.analysis.edge.video.requested",
        server_image_queue: str = "ai.analysis.server.image.requested",
        server_video_queue: str = "ai.analysis.server.video.requested",
        dlx_exchange: str = "ai.analysis.dlx",
        dlq_queue: str = "ai.analysis.dead-letter",
    ) -> None:
        """Declare exchanges, queues, and bindings.

        Args:
            exchange: Main topic exchange name.
            edge_queue: Queue for edge/YOLO11 jobs.
            server_queue: Queue for server/RF-DETR jobs.
            dlx_exchange: Dead-letter exchange name.
            dlq_queue: Dead-letter queue name.
        """
        if not self._channel:
            raise RuntimeError("Not connected. Call connect() first.")

        ch = self._channel

        # Dead-letter exchange and queue
        ch.exchange_declare(
            exchange=dlx_exchange, exchange_type="fanout", durable=True
        )
        ch.queue_declare(queue=dlq_queue, durable=True)
        ch.queue_bind(queue=dlq_queue, exchange=dlx_exchange)

        # Main exchange
        ch.exchange_declare(
            exchange=exchange, exchange_type="direct", durable=True
        )

        # C# Backend integration: identity-exchange (topic)
        backend_exchange = "identity-exchange"
        ch.exchange_declare(
            exchange=backend_exchange, exchange_type="topic", durable=True
        )

        # Edge queue with dead-letter routing
        ch.queue_declare(
            queue=edge_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": dlx_exchange,
            },
        )
        ch.queue_bind(
            queue=edge_queue,
            exchange=exchange,
            routing_key="edge",
        )
        ch.queue_bind(
            queue=edge_queue,
            exchange=backend_exchange,
            routing_key="identity.event.aianalysisrequestedevent.edge",
        )

        # Server queue with dead-letter routing
        ch.queue_declare(
            queue=server_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": dlx_exchange,
            },
        )
        ch.queue_bind(
            queue=server_queue,
            exchange=exchange,
            routing_key="server",
        )
        ch.queue_bind(
            queue=server_queue,
            exchange=backend_exchange,
            routing_key="identity.event.aianalysisrequestedevent.server",
        )

        # Media-specific work queues. The runtime ingress consumer dispatches
        # into these queues, allowing images and videos to run concurrently.
        for queue in (
            edge_image_queue,
            edge_video_queue,
            server_image_queue,
            server_video_queue,
        ):
            ch.queue_declare(
                queue=queue,
                durable=True,
                arguments={"x-dead-letter-exchange": dlx_exchange},
            )

        logger.info(
            f"RabbitMQ infrastructure ready: exchange={exchange}, "
            f"edge_queue={edge_queue}, server_queue={server_queue}, "
            f"media_queues={[edge_image_queue, edge_video_queue, server_image_queue, server_video_queue]}, "
            f"backend_exchange={backend_exchange}",
            extra={"event": "rabbitmq_infrastructure_ready"},
        )

    def consume(
        self,
        queue_name: str,
        callback: Callable,
        auto_ack: bool = False,
    ) -> None:
        """Start consuming messages from a queue.

        Args:
            queue_name: Queue to consume from.
            callback: Message handler function.
            auto_ack: Whether to auto-acknowledge messages.
        """
        if not self._channel:
            raise RuntimeError("Not connected. Call connect() first.")

        self._channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=auto_ack,
        )

        logger.info(
            f"Consuming from queue: {queue_name}",
            extra={"event": "rabbitmq_consuming", "queue": queue_name},
        )

        self._channel.start_consuming()

    def close(self) -> None:
        """Close the connection gracefully."""
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
                logger.info(
                    "RabbitMQ connection closed",
                    extra={"event": "rabbitmq_disconnected"},
                )
        except Exception as e:
            logger.warning(f"Error closing RabbitMQ connection: {e}")

    @property
    def channel(self) -> Optional[BlockingChannel]:
        return self._channel


def consume_with_reconnect(
    host: str,
    port: int,
    user: str,
    password: str,
    queue_name: str,
    callback: Callable,
    exchange: str = "ai.analysis",
    edge_queue: str = "ai.analysis.edge.requested",
    server_queue: str = "ai.analysis.server.requested",
    edge_image_queue: str = "ai.analysis.edge.image.requested",
    edge_video_queue: str = "ai.analysis.edge.video.requested",
    server_image_queue: str = "ai.analysis.server.image.requested",
    server_video_queue: str = "ai.analysis.server.video.requested",
    dlx_exchange: str = "ai.analysis.dlx",
    dlq_queue: str = "ai.analysis.dead-letter",
    heartbeat: int = 600,
    prefetch_count: int = 1,
    reconnect_delay: int = 10,
) -> None:
    """Run a consumer loop with automatic reconnection.

    This function never returns under normal operation. It reconnects
    on connection loss with a configurable delay.
    """
    while True:
        client = RabbitMQClient(
            host=host,
            port=port,
            user=user,
            password=password,
            heartbeat=heartbeat,
            prefetch_count=prefetch_count,
        )
        try:
            client.connect()
            client.setup_infrastructure(
                exchange=exchange,
                edge_queue=edge_queue,
                server_queue=server_queue,
                edge_image_queue=edge_image_queue,
                edge_video_queue=edge_video_queue,
                server_image_queue=server_image_queue,
                server_video_queue=server_video_queue,
                dlx_exchange=dlx_exchange,
                dlq_queue=dlq_queue,
            )
            client.consume(queue_name, callback, auto_ack=False)
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning(
                f"RabbitMQ connection error: {e}. "
                f"Reconnecting in {reconnect_delay}s...",
                extra={"event": "rabbitmq_reconnect"},
            )
        except Exception as e:
            logger.error(
                f"Consumer error: {e}. Reconnecting in {reconnect_delay}s...",
                exc_info=True,
                extra={"event": "rabbitmq_consumer_error"},
            )
        finally:
            client.close()

        time.sleep(reconnect_delay)


def create_media_dispatcher(image_queue: str, video_queue: str) -> Callable:
    """Route an ingress message to a durable media-specific work queue."""

    def dispatch(ch, method, properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
            media_type = str(
                payload.get("mediaType", payload.get("MediaType", "Image"))
            ).lower()
            target_queue = video_queue if media_type == "video" else image_queue
            published = ch.basic_publish(
                exchange="",
                routing_key=target_queue,
                body=body,
                properties=properties,
                mandatory=True,
            )
            if published is False:
                raise RuntimeError(f"RabbitMQ rejected dispatch to {target_queue}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, TypeError):
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception:
            logger.exception("Failed to dispatch media job")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    return dispatch
