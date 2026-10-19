from __future__ import annotations

from types import SimpleNamespace

from app.tools.memory import _asked, remember
from app.user_files import read_memory


def _ctx(transcript: str) -> SimpleNamespace:
    return SimpleNamespace(client_id="c", user_lang="ru", profile_id=4242,
                           is_authenticated=True, transcript=transcript)


async def test_question_does_not_become_a_memory():
    res = await remember(ctx=_ctx("Что бы сказал Женя, если позвать его в пятницу в бар?"),
                         action="append", content="Женя не любит бары в пятницу.")
    assert res.data["error"] == "not_asked" and res.data["unknown"]
    assert "бары" not in (await read_memory(4242) or "")


async def test_explicit_request_is_saved():
    res = await remember(ctx=_ctx("Запомни, что паспорт лежит в синей папке."),
                         action="append", content="Паспорт лежит в синей папке.")
    assert res.data.get("snippet") == "Паспорт лежит в синей папке."


def test_cues():
    assert _asked("append", "Remember that my dentist is Dr. Weber.")
    assert _asked("append", "Merk dir, dass meine Tochter Anna heißt.")
    assert _asked("append", "")  # confirmed pending action: nothing to check
    assert not _asked("append", "Schreib Anna, dass ich später komme.")
    assert not _asked("replace", "Запомни, что я люблю кофе.")
    assert _asked("replace", "Забудь всё, что я тебе говорил.")
