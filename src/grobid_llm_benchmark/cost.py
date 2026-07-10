"""Estimate the API cost of an LLM run before spending anything.

Token counts are approximated from the render settings (page images dominate input) and a
per-article output budget. Image tokens follow Anthropic's area-based rule
(``w*h/750``); for OpenAI-family models this is a rough proxy but adequate for a
budgeting estimate. Prices are per-million-token rates supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Rate:
    input_per_m: float
    output_per_m: float


# Indicative list rates (USD per 1M tokens); override via CLI if they drift.
RATES: dict[str, Rate] = {
    "sonnet": Rate(3.0, 15.0),
    "haiku": Rate(1.0, 5.0),
    "opus": Rate(5.0, 25.0),
    "gpt-4o": Rate(2.5, 10.0),
    "gpt-4o-mini": Rate(0.15, 0.6),
}


def image_tokens(dpi: int, page_w_in: float = 8.5, page_h_in: float = 11.0) -> int:
    """Approximate vision tokens for one rendered page at ``dpi``."""
    w = page_w_in * dpi
    h = page_h_in * dpi
    return int(w * h / 750)


def estimate(
    n_articles: int,
    rate: Rate,
    max_pages: int = 4,
    dpi: int = 110,
    prompt_tokens: int = 1000,
    output_tokens: int = 3000,
) -> dict:
    per_article_in = image_tokens(dpi) * max_pages + prompt_tokens
    per_article_out = output_tokens
    in_cost = per_article_in * n_articles / 1_000_000 * rate.input_per_m
    out_cost = per_article_out * n_articles / 1_000_000 * rate.output_per_m
    total = in_cost + out_cost
    return {
        "n_articles": n_articles,
        "input_tokens_per_article": per_article_in,
        "output_tokens_per_article": per_article_out,
        "input_cost_usd": round(in_cost, 2),
        "output_cost_usd": round(out_cost, 2),
        "total_usd": round(total, 2),
        "usd_per_article": round(total / n_articles, 4) if n_articles else 0.0,
    }
