import json
import unittest
from types import SimpleNamespace

from shared.messaging.rabbitmq_client import create_media_dispatcher


class FakeChannel:
    def __init__(self):
        self.published = []
        self.acked = []
        self.nacked = []

    def basic_publish(self, **kwargs):
        self.published.append(kwargs)
        return True

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue=False):
        self.nacked.append((delivery_tag, requeue))


class MediaQueueDispatcherTests(unittest.TestCase):
    def dispatch(self, media_type):
        channel = FakeChannel()
        callback = create_media_dispatcher("image-queue", "video-queue")
        callback(
            channel,
            SimpleNamespace(delivery_tag="delivery-1"),
            SimpleNamespace(content_type="application/json"),
            json.dumps({"requestId": "request-1", "mediaType": media_type}).encode(),
        )
        return channel

    def test_image_is_dispatched_to_image_queue(self):
        channel = self.dispatch("Image")
        self.assertEqual(channel.published[0]["routing_key"], "image-queue")
        self.assertEqual(channel.acked, ["delivery-1"])
        self.assertEqual(channel.nacked, [])

    def test_video_is_dispatched_to_video_queue(self):
        channel = self.dispatch("Video")
        self.assertEqual(channel.published[0]["routing_key"], "video-queue")
        self.assertEqual(channel.acked, ["delivery-1"])

    def test_invalid_json_is_dead_lettered(self):
        channel = FakeChannel()
        callback = create_media_dispatcher("image-queue", "video-queue")
        callback(
            channel,
            SimpleNamespace(delivery_tag="bad-delivery"),
            None,
            b"not-json",
        )
        self.assertEqual(channel.published, [])
        self.assertEqual(channel.nacked, [("bad-delivery", False)])


if __name__ == "__main__":
    unittest.main()
