from __future__ import annotations

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from crewai.tools import tool


_WEEKDAY_VI = [
    "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm",
    "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"
]


def _weekday_vi(d: date, tz: str) -> str:
    # Make a tz-aware datetime at midnight to get consistent weekday in that timezone
    dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=ZoneInfo(tz))
    return _WEEKDAY_VI[dt.weekday()]


@tool("get_datetime")
def get_datetime(tz: str = "Asia/Ho_Chi_Minh") -> str:
    """
    Return current date/time in a LLM-friendly string.
    Includes:
      - ISO8601 datetime with timezone offset
      - Vietnamese human-readable weekday + dd/mm/yyyy HH:MM:SS
    """
    now = datetime.now(ZoneInfo(tz))
    weekday = _WEEKDAY_VI[now.weekday()]

    iso = now.isoformat(timespec="seconds")
    human = f"{weekday}, {now:%d/%m/%Y %H:%M:%S} ({tz})"

    return f"now_iso={iso}\nnow_human={human}\ntz={tz}"


@tool("date_add_delta_days")
def date_add_delta_days(delta_days: int, tz: str = "Asia/Ho_Chi_Minh") -> str:
    """
    Compute the target DATE by adding delta_days from today (in the given timezone).
    delta_days:
      - positive: future
      - negative: past

    Returns base date + target date in ISO + Vietnamese human format.
    """
    now = datetime.now(ZoneInfo(tz))
    base = now.date()
    target = (now + timedelta(days=int(delta_days))).date()

    base_iso = base.isoformat()
    target_iso = target.isoformat()

    base_human = f"{_weekday_vi(base, tz)}, {base:%d/%m/%Y} ({tz})"
    target_human = f"{_weekday_vi(target, tz)}, {target:%d/%m/%Y} ({tz})"

    return (
        f"base_date_iso={base_iso}\n"
        f"base_date_human={base_human}\n"
        f"delta_days={int(delta_days)}\n"
        f"target_date_iso={target_iso}\n"
        f"target_date_human={target_human}\n"
        f"tz={tz}"
    )
