"""
date_filter.py
==============
Utilitas filter tanggal untuk pipeline Reputation Monitoring UNSIKA.

Masalah yang diselesaikan:
- News scraper kadang mengambil ulang artikel lama (2022, 2023, dst)
  meski di-scrape hari ini → published_date harus dicek, bukan scraped_at
- Tweets created_at selalu null → fallback ke scraped_at
- Data dari load_all_data() sudah jadi pd.Timestamp, bukan string

Penggunaan:
    from date_filter import filter_recent_news, filter_recent_tweets, DEFAULT_DAYS
"""

import logging
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 90


def _parse_date(raw: Union[str, datetime, None]) -> Optional[datetime]:
    """
    Parse berbagai format tanggal ke datetime timezone-aware.

    Mendukung:
      - pd.Timestamp         → dari load_all_data() setelah pd.to_datetime()
      - datetime             → python native datetime
      - str RFC 2822         → "Wed, 28 Dec 2022 08:00:00 GMT" (published_date RSS)
      - str ISO 8601         → "2026-06-08T00:53:07.947Z" (scraped_at)
      - pd.NaT / None / NaN → return None
    """
    if raw is None:
        return None

    # Handle pd.NaT dan float NaN tanpa import pandas
    # pd.NaT == pd.NaT adalah True, tapi kita cek via string representasi
    raw_str = str(raw)
    if raw_str in ("NaT", "nan", "None", ""):
        return None

    # Sudah datetime (termasuk pd.Timestamp yang subclass datetime)
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=timezone.utc) if raw.tzinfo is None else raw

    # String: coba RFC 2822 dulu (format Google News RSS)
    if isinstance(raw, str):
        try:
            return parsedate_to_datetime(raw)
        except Exception:
            pass
        # Coba ISO 8601
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    return None


def filter_recent_news(
    news_list: list,
    days: int = DEFAULT_DAYS,
    date_field: str = "published_date",
    fallback_field: str = "scraped_at",
) -> list:
    """
    Filter berita berdasarkan tanggal publikasi.
    Jika published_date kosong/NaT → fallback ke scraped_at.
    """
    if not news_list:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    result = []
    skipped_old = 0
    skipped_no_date = 0

    for news in news_list:
        raw_date = news.get(date_field)

        # Cek apakah NaT/None/NaN → pakai fallback
        raw_str = str(raw_date)
        if raw_date is None or raw_str in ("NaT", "nan", "None", ""):
            raw_date = news.get(fallback_field)

        dt = _parse_date(raw_date)

        if dt is None:
            skipped_no_date += 1
            result.append(news)  # tidak bisa parse → ikutkan (safe default)
            continue

        if dt >= cutoff:
            result.append(news)
        else:
            skipped_old += 1

    logger.info(
        f"[filter_recent_news] Total: {len(news_list)} | "
        f"Lolos ({days}h): {len(result)} | "
        f"Dibuang (terlalu lama): {skipped_old} | "
        f"Tanpa tanggal (diikutkan): {skipped_no_date}"
    )
    return result


def filter_recent_tweets(
    tweets_list: list,
    days: int = DEFAULT_DAYS,
    date_field: str = "created_at",
    fallback_field: str = "scraped_at",
) -> list:
    """
    Filter tweets berdasarkan tanggal.
    created_at sering null → otomatis fallback ke scraped_at.
    """
    if not tweets_list:
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    result = []
    skipped_old = 0
    skipped_no_date = 0
    used_fallback = 0

    for tweet in tweets_list:
        raw_date = tweet.get(date_field)

        raw_str = str(raw_date)
        if raw_date is None or raw_str in ("NaT", "nan", "None", ""):
            raw_date = tweet.get(fallback_field)
            if raw_date:
                used_fallback += 1

        dt = _parse_date(raw_date)

        if dt is None:
            skipped_no_date += 1
            result.append(tweet)
            continue

        if dt >= cutoff:
            result.append(tweet)
        else:
            skipped_old += 1

    logger.info(
        f"[filter_recent_tweets] Total: {len(tweets_list)} | "
        f"Lolos ({days}h): {len(result)} | "
        f"Dibuang (terlalu lama): {skipped_old} | "
        f"Pakai fallback scraped_at: {used_fallback} | "
        f"Tanpa tanggal (diikutkan): {skipped_no_date}"
    )
    return result


def filter_recent_data(
    news_list: list,
    tweets_list: list,
    days: int = DEFAULT_DAYS,
) -> tuple:
    """
    Shortcut: filter news dan tweets sekaligus.
    Returns: Tuple (news_filtered, tweets_filtered)
    """
    return filter_recent_news(news_list, days=days), filter_recent_tweets(tweets_list, days=days)