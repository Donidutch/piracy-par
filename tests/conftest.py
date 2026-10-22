import socket

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries to open a network connection."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access attempted during tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
