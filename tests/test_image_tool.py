"""Tests for adapter/image_utils.py — message image extraction (body first,
then quoted chain), mirroring astrbot_plugin_serpapi_imgsearch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter.image_utils import describe_no_image, find_message_image


class FakeImage:
    def __init__(self, url: str = "", file: str = "") -> None:
        self.url = url
        self.file = file


class FakeReply:
    def __init__(self, chain) -> None:
        self.chain = chain


def test_body_image_is_preferred_over_reply_image():
    body_img = FakeImage(file="body.png")
    reply_img = FakeImage(url="reply.png")
    messages = [FakeReply(chain=[reply_img]), body_img]
    assert find_message_image(messages) is body_img


def test_reply_chain_image_found_when_body_has_none():
    reply_img = FakeImage(file="quoted.png")
    messages = [FakeReply(chain=[FakeImage(), reply_img])]
    assert find_message_image(messages) is reply_img


def test_images_without_url_or_file_are_skipped():
    messages = [FakeImage(), FakeReply(chain=[FakeImage()])]
    assert find_message_image(messages) is None


def test_no_image_returns_none():
    assert find_message_image([]) is None
    assert find_message_image([FakeReply(chain=[])]) is None
    assert find_message_image(None) is None


def test_error_message_guides_user():
    msg = describe_no_image()
    assert "未在消息中找到图片" in msg
    assert "引用" in msg
