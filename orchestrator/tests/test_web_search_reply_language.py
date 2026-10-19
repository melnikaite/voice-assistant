from app.i18n import t
from app.tools.web_search import _summary_messages


def test_reply_language_line_follows_user_lang_not_query():
    results = [{"title": "Bitcoin", "body": "Bitcoin trades at $67,000."}]
    for lang in ("en", "ru", "de"):
        user_msg = _summary_messages("bitcoin price", results, lang)[1]["content"]
        assert user_msg.rstrip().endswith(t("search.reply_language", lang))
