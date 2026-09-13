"""Deterministic Buy-or-Wait financial planning agent.

Run from the repository root: python code/main.py
The agent reads only participant-facing dataset files and writes ../output.csv.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
HORIZON = 90
OUT_FIELDS = ["request_id", "amount_safe_to_pay", "affordability_status",
              "recommended_payment_method", "payment_plan",
              "earliest_date_for_full_payment", "spending_changes_needed",
              "decision_explanation"]


def dec(value, default=Decimal("0")):
    try:
        if value is None or str(value).strip() == "":
            return default
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return default


def dt(value):
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def yes(value):
    return str(value).strip().lower() in {"true", "yes", "1"}


def money(v):
    """Canonical non-scientific representation, retaining supplied precision."""
    v = Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(v, "f").rstrip("0").rstrip(".")
    return text or "0"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_month(d):
    # Dataset recurrences are monthly; clamping handles dates like the 31st.
    import calendar
    month = d.month + 1
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


@dataclass
class Flow:
    when: date
    amount: Decimal             # positive credit, negative debit
    event_id: str = ""
    recurring: bool = False
    category: str = ""
    flexible: bool = False
    minimum: Decimal = Decimal("0")


class Agent:
    def __init__(self):
        self.profiles = {r["user_id"]: r for r in read_csv("financial_profiles.csv")}
        self.events = read_csv("financial_events.csv")
        self.options = defaultdict(list)
        for r in read_csv("request_payment_options.csv"):
            self.options[r["request_id"]].append(r)
        self.messages = read_csv("messages.csv")
        self.images = read_csv("images.csv")
        self.rates = {(r["rate_date"], r["from_currency"], r["to_currency"]): dec(r["rate"])
                      for r in read_csv("exchange_rates.csv")}
        self.by_user = defaultdict(list)
        for r in self.events:
            self.by_user[r["user_id"]].append(r)
        self.image_for_event = {r["related_event_id"]: r["image_id"] for r in self.images if r["related_event_id"]}
        self.ocr_cache = {}

    def image_amount(self, event):
        """Best-effort local OCR, used only for genuinely blank event amounts."""
        event_id = event["event_id"]
        if event_id in self.ocr_cache:
            return self.ocr_cache[event_id]
        image_id = self.image_for_event.get(event_id)
        if not image_id:
            return Decimal("0")
        png = DATA / "media" / "images" / f"{image_id}.png"
        try:
            raw = subprocess.run(["tesseract", str(png), "stdout"], capture_output=True,
                                 text=True, timeout=20, check=False).stdout
            # Prefer labelled money values over arbitrary IDs/dates.
            amounts = re.findall(r"(?:INR|IDR|ZAR|USD|EUR|₹|\$|€|Rp)\s*([\d,.]+)", raw, re.I)
            value = dec(amounts[-1]) if amounts else Decimal("0")
        except (OSError, subprocess.SubprocessError):
            value = Decimal("0")
        self.ocr_cache[event_id] = value
        return value

    def converted_amount(self, event, profile):
        value = dec(event.get("amount"))
        if not value and not str(event.get("amount", "")).strip():
            value = self.image_amount(event)
        source, target = event.get("currency"), profile["home_currency"]
        if value and source and source != target:
            keydate = event.get("settlement_date") or event.get("event_date")
            rate = self.rates.get((keydate, source, target))
            if rate:
                value *= rate
            else:
                reverse = self.rates.get((keydate, target, source))
                if reverse:
                    value /= reverse
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def valid_future_row(self, row, today):
        status = row.get("status", "").lower()
        settle = row.get("settlement_date") or row.get("event_date")
        if not settle or dt(settle) < today:
            return False
        if status in {"failed", "cancelled"}:
            return False
        # Never count unconfirmed incoming cash.
        if (row.get("direction") == "credit" and status in {"pending", "unrealized"}):
            return False
        # A scheduled payroll is an explicitly confirmed salary, unlike an uncertain credit.
        if row.get("direction") == "credit" and status == "scheduled" and row.get("event_type") != "income":
            return False
        if status == "unrealized":
            return False
        return status in {"settled", "pending", "scheduled"}

    def recurring_groups(self, rows, profile):
        """Infer monthly recurrence only from >=2 similar historical cash rows."""
        groups = defaultdict(list)
        for row in rows:
            if row.get("status", "").lower() not in {"settled", "pending", "scheduled"}:
                continue
            if not row.get("amount") or row.get("event_type") in {"investment_value", "non_cash"}:
                continue
            key = (row.get("event_type"), row.get("description"), row.get("category"), row.get("direction"), row.get("currency"))
            groups[key].append(row)
        answer = []
        for key, items in groups.items():
            items.sort(key=lambda x: dt(x.get("settlement_date") or x["event_date"]))
            dates = [dt(x.get("settlement_date") or x["event_date"]) for x in items]
            monthly_pairs = sum(25 <= (b-a).days <= 35 for a, b in zip(dates, dates[1:]))
            # Exact merchant descriptions are noisy for weekly groceries / transport;
            # they are projected below at category cadence instead.
            if monthly_pairs >= 1 and key[2] not in {"groceries", "transport", "dining", "entertainment", "shopping"}:
                answer.append(items)
        return answer

    def flows(self, user, today, changes=()):
        profile = self.profiles[user]
        end = today + timedelta(days=HORIZON)
        rows = self.by_user[user]
        flows = []
        # Future supplied commitments, with latest duplicate lifecycle record preferred.
        for row in rows:
            if self.valid_future_row(row, today):
                when = dt(row.get("settlement_date") or row["event_date"])
                if when <= end:
                    amount = self.converted_amount(row, profile)
                    sign = Decimal("1") if row.get("direction") == "credit" else Decimal("-1")
                    flows.append(Flow(when, sign * amount, row["event_id"], False, row.get("category", ""),
                                      row.get("flexibility", "").lower() == "flexible", dec(row.get("minimum_allowed_amount"))))
        # Reconstruct repeats after the last known cycle, avoiding dates already provided.
        known = {(f.when, f.event_id) for f in flows}
        supplied_dates = {dt(r.get("settlement_date") or r["event_date"]) for r in rows if r.get("settlement_date") or r.get("event_date")}
        for items in self.recurring_groups(rows, profile):
            latest = items[-1]
            last = dt(latest.get("settlement_date") or latest["event_date"])
            candidate = last
            while candidate < today:
                candidate = add_month(candidate)
        # Essential variable expenses use observed category cadence (weekly/biweekly
        # in this corpus), rather than accidentally treating a returning merchant as
        # a monthly bill.  Use the recent mean, slightly rounded upward for safety.
        protected = set(filter(None, profile.get("expense_categories_to_protect", "").split("|")))
        for category in {r.get("category", "") for r in rows}:
            items = [r for r in rows if r.get("category") == category and r.get("direction") == "debit"
                     and r.get("status", "").lower() == "settled" and r.get("amount")]
            items.sort(key=lambda x: dt(x.get("settlement_date") or x["event_date"]))
            if len(items) < 4 or category not in {"groceries", "transport", "dining", "entertainment", "shopping"}:
                continue
            recent = items[-8:]
            gaps = sorted((dt(b.get("settlement_date") or b["event_date"]) - dt(a.get("settlement_date") or a["event_date"])).days
                          for a, b in zip(recent, recent[1:]))
            cadence = gaps[len(gaps)//2] if gaps else 0
            if not 5 <= cadence <= 22: continue
            observed = [self.converted_amount(x, profile) for x in recent]
            average = (sum(observed) / len(observed) * Decimal("1.05")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            candidate = dt(recent[-1].get("settlement_date") or recent[-1]["event_date"])
            while candidate < today: candidate += timedelta(days=cadence)
            while candidate <= end:
                exists = any((f.when == candidate and f.category == category) for f in flows)
                if not exists:
                    prototype = recent[-1]
                    flows.append(Flow(candidate, -average, prototype["event_id"], True, category,
                                      prototype.get("flexibility", "").lower() == "flexible", dec(prototype.get("minimum_allowed_amount"))))
                candidate += timedelta(days=cadence)
        # Extend ordinary salary only when history supports a monthly cycle. Bonuses
        # and commissions are deliberately excluded as uncertain income.
        salary = [r for r in rows if r.get("event_type") == "income" and r.get("direction") == "credit"
                  and not re.search(r"bonus|commission|refund|lottery", r.get("description", ""), re.I)
                  and r.get("amount") and r.get("status", "").lower() in {"settled", "scheduled"}]
        salary.sort(key=lambda x: dt(x.get("settlement_date") or x["event_date"]))
        if len(salary) >= 2:
            latest = salary[-1]
            candidate = dt(latest.get("settlement_date") or latest["event_date"])
            while candidate < today: candidate = add_month(candidate)
            sal = self.converted_amount(latest, profile)
            while candidate <= end:
                if not any(f.when == candidate and f.amount > 0 and f.category == latest.get("category", "") for f in flows):
                    flows.append(Flow(candidate, sal, latest["event_id"], True, latest.get("category", "")))
                candidate = add_month(candidate)
            while candidate <= end:
                # Do not double count a supplied future occurrence of same pattern.
                same = [r for r in rows if (r.get("description"), r.get("category"), r.get("direction")) ==
                        (latest.get("description"), latest.get("category"), latest.get("direction")) and
                        dt(r.get("settlement_date") or r["event_date"]) == candidate]
                if not same:
                    amount = self.converted_amount(latest, profile)
                    sign = Decimal("1") if latest.get("direction") == "credit" else Decimal("-1")
                    flows.append(Flow(candidate, sign * amount, latest["event_id"], True, latest.get("category", ""),
                                      latest.get("flexibility", "").lower() == "flexible", dec(latest.get("minimum_allowed_amount"))))
                candidate = add_month(candidate)
        # Apply a change to every future recurrence based on that source event.
        altered = []
        for f in flows:
            amount = f.amount
            for kind, eid, target in changes:
                if f.recurring and f.event_id == eid and f.amount < 0:
                    if kind == "stop": amount = Decimal("0")
                    elif kind == "reduce": amount = -target
            altered.append(Flow(f.when, amount, f.event_id, f.recurring, f.category, f.flexible, f.minimum))
        return altered

    @staticmethod
    def balances(start, today, flows, payments=()):
        per_day = defaultdict(Decimal)
        for f in flows: per_day[f.when] += f.amount
        for when, amount in payments: per_day[when] -= amount
        bal, result = start, []
        for offset in range(HORIZON + 1):
            day = today + timedelta(days=offset)
            bal += per_day[day]
            result.append((day, bal))
        return result

    def safe(self, profile, today, flows, payments):
        start = dec(profile["current_available_balance"])
        floor = dec(profile["minimum_balance_to_keep"])
        return min(b for _, b in self.balances(start, today, flows, payments)) >= floor

    def capacity_today(self, profile, today, flows, requested):
        start = dec(profile["current_available_balance"])
        floor = dec(profile["minimum_balance_to_keep"])
        min_base = min(b for _, b in self.balances(start, today, flows))
        return max(Decimal("0"), min(requested, min_base-floor)).quantize(Decimal("0.01"))

    def earliest_full(self, profile, today, flows, requested):
        for offset in range(HORIZON + 1):
            candidate = today + timedelta(days=offset)
            if self.safe(profile, today, flows, [(candidate, requested)]):
                return candidate
        return None

    def eligible_options(self, request, profile):
        methods = set(filter(None, profile.get("payment_methods_user_will_consider", "").split("|")))
        max_months = dec(profile.get("max_installment_months"), Decimal("999"))
        out = []
        for option in self.options[request["request_id"]]:
            method = option["payment_method"]
            if method not in methods: continue
            if method == "installments" and dec(option["number_of_payments"]) > max_months: continue
            out.append(option)
        return out

    def option_payments(self, option):
        first, every, count, amount = dt(option["first_payment_date"]), int(option.get("payment_frequency_days") or 0), int(option["number_of_payments"]), dec(option["payment_amount"])
        return [(first + timedelta(days=every*i), amount) for i in range(count)]

    def change_candidates(self, profile, flows):
        reduce_categories = set(filter(None, profile.get("expense_categories_user_is_willing_to_reduce", "").split("|")))
        stop_categories = set(filter(None, profile.get("expense_categories_user_is_willing_to_stop", "").split("|")))
        choices = []
        seen = set()
        for f in flows:
            if not (f.recurring and f.flexible and f.amount < 0 and f.event_id not in seen): continue
            seen.add(f.event_id)
            if f.category in stop_categories: choices.append(("stop", f.event_id, Decimal("0")))
            if f.category in reduce_categories and -f.amount > f.minimum:
                choices.append(("reduce", f.event_id, f.minimum))
        # Single changes and pairs are enough for the max-3 action contract and preserve parsimony.
        variants = [(c,) for c in choices]
        for i, a in enumerate(choices):
            for b in choices[i+1:]:
                if a[1] != b[1]: variants.append((a, b))
        return variants

    def fmt_plan(self, payments):
        return "|".join(f"{d.isoformat()}:{money(a)}" for d, a in sorted(payments))

    def explanation(self, profile, amount, method, minimum, changes, later=False):
        cur = profile["home_currency"]
        if method == "not_recommended":
            return f"No eligible plan protects the {cur} {money(minimum)} minimum through the next 90 days."
        if changes:
            action = " and ".join("stop flexible spending" if c[0] == "stop" else "reduce flexible spending" for c in changes)
            return f"This plan keeps at least {cur} {money(minimum)} available after you {action}."
        if later:
            return f"Waiting preserves the {cur} {money(minimum)} minimum; the full amount is safe on the stated date."
        return f"This plan keeps at least {cur} {money(minimum)} available throughout the 90-day forecast."

    def decide(self, request):
        profile = self.profiles[request["user_id"]]
        today, requested = dt(request["request_date"]), dec(request["requested_amount"])
        flows = self.flows(request["user_id"], today)
        safe_today = self.capacity_today(profile, today, flows, requested)
        earliest = self.earliest_full(profile, today, flows, requested)
        methods = set(filter(None, profile.get("payment_methods_user_will_consider", "").split("|")))
        deadline = dt(request["desired_completion_date"])
        minimum = dec(profile["minimum_balance_to_keep"])
        # Candidate tuple: method, payments, changes, option_id, full-date.
        candidates = []
        if "full_payment" in methods and self.safe(profile, today, flows, [(today, requested)]):
            candidates.append(("full_payment", [(today, requested)], (), "", today))
        for option in self.eligible_options(request, profile):
            if option["payment_method"] != "installments": continue
            pays = self.option_payments(option)
            if pays[-1][0] <= deadline and self.safe(profile, today, flows, pays):
                candidates.append(("installments", pays, (), option["payment_option_id"], earliest))
        if (yes(request["allows_partial_payment"]) and "partial_payment" in methods and
                Decimal("0") < safe_today < requested and earliest and earliest <= deadline):
            pays = [(today, safe_today), (earliest, requested-safe_today)]
            if self.safe(profile, today, flows, pays):
                candidates.append(("partial_payment", pays, (), "", earliest))
        # Spending changes can make an otherwise unavailable plan safe. Prefer the least intervention.
        if not candidates:
            for changes in self.change_candidates(profile, flows):
                changed = self.flows(request["user_id"], today, changes)
                if "full_payment" in methods and self.safe(profile, today, changed, [(today, requested)]):
                    candidates.append(("full_payment", [(today, requested)], changes, "", self.earliest_full(profile, today, flows, requested)))
                for option in self.eligible_options(request, profile):
                    if option["payment_method"] == "installments":
                        pays = self.option_payments(option)
                        if pays[-1][0] <= deadline and self.safe(profile, today, changed, pays):
                            candidates.append(("installments", pays, changes, option["payment_option_id"], earliest))
                if candidates: break
        if candidates:
            # required ranking after meeting deadline/no changes: total cost, earliest first payment, fewer, option id
            def rank(c):
                method, pays, changes, oid, _ = c
                return (0 if pays[-1][0] <= deadline else 1, len(changes), sum(a for _, a in pays), pays[0][0], len(pays), oid)
            method, pays, changes, oid, full_date = sorted(candidates, key=rank)[0]
            status = "affordable_now" if method == "full_payment" and not changes and pays[0][0] == today else "affordable_with_plan"
            change_text = "none" if not changes else "|".join(
                f"stop:{eid}" if kind == "stop" else f"reduce_to:{eid}:{money(target)}" for kind, eid, target in changes)
            return {"request_id": request["request_id"], "amount_safe_to_pay": money(safe_today),
                    "affordability_status": status, "recommended_payment_method": method,
                    "payment_plan": self.fmt_plan(pays),
                    "earliest_date_for_full_payment": (today if earliest == today else earliest).isoformat() if earliest else "",
                    "spending_changes_needed": change_text,
                    "decision_explanation": self.explanation(profile, requested, method, minimum, changes)}
        # Waiting has a plan only if user will consider paying in full and date is forecast-safe.
        if earliest and "full_payment" in methods:
            return {"request_id": request["request_id"], "amount_safe_to_pay": money(safe_today),
                    "affordability_status": "affordable_later", "recommended_payment_method": "wait",
                    "payment_plan": self.fmt_plan([(earliest, requested)]),
                    "earliest_date_for_full_payment": earliest.isoformat(), "spending_changes_needed": "none",
                    "decision_explanation": self.explanation(profile, requested, "wait", minimum, (), True)}
        return {"request_id": request["request_id"], "amount_safe_to_pay": money(safe_today),
                "affordability_status": "not_affordable", "recommended_payment_method": "not_recommended",
                "payment_plan": "none", "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                "decision_explanation": self.explanation(profile, requested, "not_recommended", minimum, ())}


def validate(rows, requests):
    assert len(rows) == len(requests)
    by_id = {r["request_id"]: r for r in requests}
    for row in rows:
        requested = dec(by_id[row["request_id"]]["requested_amount"])
        assert Decimal("0") <= dec(row["amount_safe_to_pay"]) <= requested
        assert row["affordability_status"] in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
        assert row["recommended_payment_method"] in {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
        if row["recommended_payment_method"] == "not_recommended": assert row["payment_plan"] == "none"


def main():
    agent = Agent()
    requests = read_csv("requests.csv")
    rows = [agent.decide(r) for r in requests]
    validate(rows, requests)
    with (ROOT / "output.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} validated predictions to {ROOT / 'output.csv'}")


if __name__ == "__main__":
    main()
