"""
Storage package — public API.

All external imports go through here (``from .storage import X``).
Individual sub-modules handle one domain each:
  db               — shared connection factory + threading lock
  schema           — init_schema() + migrations
  sessions         — session rows + conversation history
  utterances       — utterance rows + semantic-memory candidates
  reminders        — reminder CRUD
  speaker_profiles — speaker profile CRUD + multi-sample averaging
  custom_voices    — user-cloned XTTS voices (reference-audio path store)
  token_usage      — LLM token-usage rows + daily/per-tool/per-user aggregates
  pending_actions  — Sprint-2 deferred-action queue for high-risk tool calls
  items            — personal item store (links, text, videos, screenshots)
  categories       — hierarchical folders + checklist nodes for the item store
"""
from .schema import init_schema
from .sessions import get_recent_history, start_session
from .utterances import (
    get_candidate_utterances,
    save_utterance,
    update_utterance_embedding,
)
from .reminders import (
    add_reminder,
    cancel_reminder_db,
    get_missed_reminders,
    get_pending_reminders,
    list_upcoming_reminders,
    mark_reminder_delivered,
    mark_reminder_fired,
)
from .speaker_profiles import (
    delete_speaker_profile,
    get_speaker_profile_by_name,
    get_speaker_profiles,
    save_speaker_profile,
    set_speaker_tts_voice,
    update_speaker_profile,
)
from .custom_voices import (
    delete_custom_voice,
    get_custom_voice_by_id,
    get_custom_voices,
    save_custom_voice,
)
from .token_usage import (
    PRICING,
    add_token_usage,
    compute_projected_cost,
    get_daily_usage,
    get_per_tool_usage,
    get_per_user_usage,
)
from .pending_actions import (
    DEFAULT_TTL_S,
    enqueue_action,
    get_pending_action,
    list_approved_actions,
    list_pending_actions,
    list_recent_actions,
    mark_approved,
    mark_executed,
    mark_rejected,
)
from .auth_sessions import (
    create_session,
    get_session,
    revoke_session,
    sweep_expired_sessions,
)
from .push_subscriptions import (
    delete_by_endpoint as delete_push_subscription,
    get_by_endpoint as get_push_subscription,
    list_for_profile as list_push_subscriptions,
    touch_last_used as touch_push_subscription,
    upsert_subscription as upsert_push_subscription,
)
from .voice_messages import (
    VOICE_MESSAGES_DIR,
    audio_path as voicemail_audio_path,
    count_unread as count_unread_voicemail,
    delete_voice_message,
    get_voice_message,
    list_for_recipient as list_voicemail,
    list_outgoing_voicemail,
    list_unseen_replies_for_sender,
    mark_listened as mark_voicemail_listened,
    mark_reply_delivered as mark_voicemail_reply_delivered,
    save_reply as save_voicemail_reply,
    save_voice_message,
    set_summary as set_voicemail_summary,
)
from .items import (
    ITEMS_DIR,
    create_item,
    delete_item,
    fts_search as fts_search_items,
    get_item,
    get_item_embeddings,
    list_items,
    move_item,
    purge_expired_trash,
    purge_item,
    reorder_item,
    restore_item,
    set_item_embedding,
    set_item_media_path,
    set_item_summary,
    toggle_checked,
    update_item,
)
from .categories import (
    create_category,
    delete_category,
    get_category,
    list_categories,
    list_category_shares,
    list_subtree,
    move_category,
    rename_category,
    resolve_category_by_name,
    restore_category,
    share_category,
    unshare_category,
)

__all__ = [
    # schema
    "init_schema",
    # sessions
    "start_session",
    "get_recent_history",
    # utterances
    "save_utterance",
    "update_utterance_embedding",
    "get_candidate_utterances",
    # reminders
    "add_reminder",
    "mark_reminder_fired",
    "mark_reminder_delivered",
    "get_pending_reminders",
    "get_missed_reminders",
    "list_upcoming_reminders",
    "cancel_reminder_db",
    # speaker profiles
    "save_speaker_profile",
    "update_speaker_profile",
    "get_speaker_profiles",
    "get_speaker_profile_by_name",
    "set_speaker_tts_voice",
    "delete_speaker_profile",
    # custom voices
    "save_custom_voice",
    "get_custom_voices",
    "get_custom_voice_by_id",
    "delete_custom_voice",
    # token usage
    "add_token_usage",
    "get_daily_usage",
    "get_per_tool_usage",
    "get_per_user_usage",
    "compute_projected_cost",
    "PRICING",
    # pending actions
    "enqueue_action",
    "list_pending_actions",
    "list_approved_actions",
    "list_recent_actions",
    "get_pending_action",
    "mark_approved",
    "mark_rejected",
    "mark_executed",
    "DEFAULT_TTL_S",
    # auth sessions
    "create_session",
    "get_session",
    "revoke_session",
    "sweep_expired_sessions",
    # push subscriptions (Web Push)
    "upsert_push_subscription",
    "list_push_subscriptions",
    "get_push_subscription",
    "delete_push_subscription",
    "touch_push_subscription",
    # voicemail
    "VOICE_MESSAGES_DIR",
    "save_voice_message",
    "get_voice_message",
    "list_voicemail",
    "list_outgoing_voicemail",
    "list_unseen_replies_for_sender",
    "count_unread_voicemail",
    "mark_voicemail_listened",
    "mark_voicemail_reply_delivered",
    "save_voicemail_reply",
    "set_voicemail_summary",
    "delete_voice_message",
    "voicemail_audio_path",
    # items
    "ITEMS_DIR",
    "create_item",
    "get_item",
    "update_item",
    "delete_item",
    "restore_item",
    "purge_item",
    "list_items",
    "fts_search_items",
    "move_item",
    "reorder_item",
    "toggle_checked",
    "set_item_embedding",
    "set_item_summary",
    "set_item_media_path",
    "get_item_embeddings",
    "purge_expired_trash",
    # categories
    "create_category",
    "get_category",
    "list_categories",
    "list_subtree",
    "rename_category",
    "move_category",
    "delete_category",
    "restore_category",
    "resolve_category_by_name",
    "share_category",
    "unshare_category",
    "list_category_shares",
]
