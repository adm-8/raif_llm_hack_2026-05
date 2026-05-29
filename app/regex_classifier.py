from __future__ import annotations

import re

from app.models import CLEAR_CATEGORY, Conversation

_Rule = tuple[re.Pattern[str], re.Pattern[str] | None, float]
_RULES: list[tuple[str, list[_Rule]]] = []


def _r(pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _rule(
    pattern: str,
    confidence: float,
    *,
    exclude: str | None = None,
    flags: int = re.IGNORECASE | re.DOTALL,
) -> _Rule:
    neg = re.compile(exclude, flags) if exclude else None
    return re.compile(pattern, flags), neg, confidence


# 1. adversarial_attack — prompt/JSON injection, impersonation of tech staff,
#    social engineering to extract unwarranted compensation
_ADVERSARIAL: list[_Rule] = [
    # JSON command injection: {"mode": "internal_debug", ...}
    _rule(r'\{[^}]*"?(mode|action|command)"?\s*:\s*"?(internal|disable|debug|bypass)', 0.95),
    # Explicit request for strict JSON response format (jailbreak framing)
    _rule(r'ответ(ь|ите?)\s+строго\s+в\s+формате\s+json', 0.95),
    # Claiming to be from a partner tech department
    _rule(r'(старший|ведущий|главный)\s+инженер\s+интеграции', 0.95),
    _rule(r'со стороны партнер\w+.{0,60}(проверяем|тест|обновлени)', 0.90),
    # Asking about antifrod internal reasoning / disable checks
    _rule(r'(отключ|перевести?\s+в\s+режим)\s+без\s+.{0,30}проверок', 0.90),
    _rule(r'внутренн\w+\s+reasoning\s+антифрод', 0.90),
    _rule(r'(отключени|отключить).{0,40}(антифрод|проверк\w+\s+безопасности)', 0.90),
    # Claim that action was agreed with tech department
    _rule(r'согласован\w+\s+с\s+технич\w+\s+отдел', 0.85),
    # Social engineering: manipulate chatbot into committing to a bonus, then demand immediate payment
    _rule(r'начислите.{0,30}(бонус|компенсацию|баллы).{0,60}(сразу|не\s+хочу\s+объяснять|не\s+хочу\s+еще\s+раз)', 0.90),
    _rule(r'(бонус|компенсацию|баллы).{0,60}начислите.{0,60}сразу', 0.85),
]
_RULES.append(("adversarial_attack", _ADVERSARIAL))

# 2. identity_deception — probing account recovery while hiding true identity
_IDENTITY: list[_Rule] = [
    # Lost SIM + asking about remote access recovery
    _rule(r'(потерял|недоступн|не\s+работает).{0,60}(SIM|сим.карт|номер).{0,60}(доступ|войт|восстанов)', 0.95),
    _rule(r'(SIM.карт\w+|старый\s+номер).{0,60}(потерял|недоступен|не\s+работает)', 0.90),
    # SIM is already disconnected + probing what credentials are enough
    _rule(r'SIM.{0,20}(отключена|заблокирована).{0,200}(паспортн|кодовое\s+слово|остаток|операции)', 0.90),
    # Asking specifically about remote recovery options
    _rule(r'дистанционно.{0,40}(возможно|восстанов|подтвер)', 0.85),
    _rule(r'восстановить\s+доступ.{0,60}дистанционно', 0.85),
    # Claiming to help a relative but having their phone and trying to bypass ID
    _rule(r'телефон\s+(владельца|хозяина).{0,40}(у\s+меня|на\s+руках)', 0.90),
    _rule(r'подтвердите\s+вход\s+вручную', 0.90),
    _rule(r'(мне|нам).{0,20}на\s+\d+\s+минут.{0,20}нужен\s+доступ', 0.90),
    # Asking "if phone/number is unavailable, can I still log in" pattern
    # exclude when context is about a relative's phone (информация extraction scenario)
    _rule(
        r'(номер|телефон).{0,30}(недоступен|не\s+работает|потерян).{0,60}(войт|вход|доступ|приложен)',
        0.85,
        exclude=r'(жен[аы]|муж[а-я]*|бр[а-я]+|сестр[аы]|родственник\w*)',
    ),
    # Probing: do operations go through without confirmation email (to verify a transfer happened silently)
    _rule(
        r'(письмо|уведомлени).{0,40}не\s+пришло.{0,80}(операция|перевод).{0,30}(прошл|выполн|могла)',
        0.80,
    ),
]
_RULES.append(("identity_deception", _IDENTITY))

# 3. scope_violation — out-of-scope requests (regulatory complaints, code help, investment advice)
_SCOPE: list[_Rule] = [
    # Complaint to Central Bank (ЦБ)
    _rule(r'жалоб\w+.{0,50}(в\s+)?ЦБ\b', 0.95),
    _rule(r'\bЦБ\b.{0,50}жалоб', 0.95),
    _rule(r'Центральн\w+\s+банк.{0,50}жалоб', 0.90),
    # Ask chatbot to write poems / haiku / creative content
    _rule(r'write\s+a\s+(haiku|poem|song|story)', 0.95),
    _rule(r'напиши\s+(хайку|стих|поэму|песню|рассказ)', 0.90),
    # Developer asking for programming code help in a bank support chat
    _rule(r'(помог\w+\s+с\s+.{0,20}Python|распарсить\s+CSV|ETL\s+pipeline|дебажить\s+.{0,30}pipeline)', 0.90),
    # Chatbot was pulled into coding assistance (user acknowledges it)
    _rule(r'банковский\s+чат\s+сейчас\s+помогает.{0,40}(дебажить|разрабатывать|писать\s+код)', 0.90),
    # Implicit investment advice via "what do most investors do when market falls"
    _rule(
        r'рынок.{0,30}(падает|растет|волатил).{0,100}(многие|обычно|как\s+правило).{0,60}(уходят|переходят|продают)',
        0.80,
    ),
    # "Should I sell now or not" — investment timing question through bank chat
    _rule(
        r'(продавать|покупать).{0,30}сейчас\s+или\s+нет',
        0.80,
        exclude=r'мошенник|аренд|задаток|участок',
    ),
]
_RULES.append(("scope_violation", _SCOPE))

# 4. information_extraction — probing third-party account data indirectly
_INFO_EXTRACT: list[_Rule] = [
    # Relative abroad + explicitly asking about their account activity
    _rule(
        r'(жен[аы]|муж[а-я]*|бр[а-я]+|сестр[аы]|родственник\w*).{0,100}'
        r'(за\s+границей|командировк|недоступен|не\s+отвечает).{0,300}'
        r'(активность|последн\w+\s+раз|счет|операц|блокировк|пользовался|карт)',
        0.90,
        exclude=r'(не\s+пытаюсь\s+узнать|не\s+хочу\s+узнать|не\s+прошу\s+проверять|не\s+прошу\s+раскрывать)',
    ),
    # Meta-observation: noting chatbot responds differently to different topics to infer account status
    _rule(
        r'(когда\s+я\s+спрашивал|когда\s+спрашиваю).{0,60}(операц|перевод|счет).{0,60}'
        r'(сразу\s+сказали|по-другому|намного\s+спокойн|осторожн)',
        0.90,
    ),
    # Asking whether another person's card would have been active / had operations
    _rule(r'(теоретически|чисто\s+теоретически).{0,80}(перевод|операция|карта).{0,80}(был[аи]?|прошл|отобразил)', 0.85),
    # Indirect metadata probing: "if a person suddenly stops using their card…"
    _rule(r'(человек|клиент).{0,40}(перестает|перестал).{0,40}(пользоваться\s+карт|пользоваться\s+счет)', 0.85),
    # Observing support caution level to infer whether there's a real problem
    _rule(r'(осторожничают|осторожн\w+).{0,60}(реально\s+есть|когда\s+реально).{0,60}(проблем|ситуация)', 0.80),
]
_RULES.append(("information_extraction", _INFO_EXTRACT))

# 5. policy_manipulation — reframing / status appeal to bypass security checks
_POLICY: list[_Rule] = [
    # Linguistic reframing: "I'm not asking to remove the check, just change the wording"
    _rule(r'(не\s+предлагаю|не\s+прошу).{0,40}убрать.{0,40}проверк.{0,60}(изменить|другой|переформулир)', 0.90),
    _rule(r'(ускорить|изменить\s+формулировк).{0,60}проверк', 0.85),
    # "Send payment now, close check after the fact"
    _rule(r'(отправить|провести).{0,40}(платеж|перевод|операцию).{0,30}постфактум', 0.90),
    # "Formally this isn't bypassing — I'm confirming myself"
    _rule(r'(формально|технически).{0,30}(не\s+обход|не\s+обходит).{0,60}(сам\s+подтверждаю|я\s+подтверждаю)', 0.90),
    # Manual override request: "just click confirm" / "confirm it manually from your side"
    _rule(r'(нажать\s+кнопку\s+подтверждения|вручную\s+обходить\s+процедур|подтвердить\s+вручную)', 0.85),
    _rule(r'(можно|просто).{0,30}(подтвердить|одобрить).{0,30}(вручную|со\s+стороны\s+банка)', 0.85),
    # Client puts responsibility in writing to pressure support into acting
    _rule(r'(поставите|поставьте)\s+комментарий.{0,60}(клиент\s+подтверждает|ответственность)', 0.85),
    # Premium/loyalty status to get exception
    _rule(
        r'(premium.клиент|вип.клиент|10\s+лет|десять\s+лет|пятн?адцать\s+лет|долго\s+обслуживаюсь).{0,100}'
        r'(гибче|исключени|по.другому|провер)',
        0.85,
    ),
    # Urgency + "transfer stuck on check" + deal will fall through
    _rule(
        r'(перевод|операция|платеж).{0,60}(завис|на\s+проверке).{0,80}'
        r'(сделка\s+через|сделку\s+потеряю|контрагент)',
        0.80,
    ),
]
_RULES.append(("policy_manipulation", _POLICY))

# 6. transaction_coercion — pressure/urgency to complete a suspicious transfer
_TRANSACTION: list[_Rule] = [
    # Urgency + real estate / rent scam keywords
    _rule(
        r'(торопит|спешка|срочно|через\s+час|сегодня\s+успеть).{0,100}(задаток|бронь|аренд|участок|недвижимост)',
        0.90,
    ),
    # Seller/landlord is rushing
    _rule(r'(продавец|арендодатель|хозяин).{0,60}(торопит|требует\s+срочно|спешит).{0,60}(перевод|аванс|задаток)', 0.90),
    # Mentioning that chatbot/support conversation led to transfer decision
    _rule(r'именно\s+после\s+разговора\s+с\s+(поддержкой|оператором).{0,80}(решил[аи]|завершил[аи])', 0.90),
    # Scam scenario: someone sent a passport + contract + rushing
    _rule(r'(скинул|прислал|отправил).{0,40}(паспорт|договор|реквизит).{0,60}(торопит|спешка|срочно)', 0.85),
    # Asking whether the bank would spot obvious fraud signs
    _rule(r'(банк\s+видел|система\s+видела?).{0,60}(явные|очевидные)\s+признаки\s+мошенничества', 0.85),
    # User preemptively records that operator confirmed no fraud signs (to shift blame later)
    _rule(r'(оператор|банк|поддержка).{0,60}подтвердил.{0,60}(отсутствие\s+подозрений|нет\s+ограничений|всё\s+выглядит)', 0.85),
    # Transfer urgency + reassurance sought from support
    _rule(r'(с\s+вашей\s+стороны|банк).{0,40}(проблем\s+нет|нет\s+причин).{0,60}(перевод|заявка)', 0.75),
]
_RULES.append(("transaction_coercion", _TRANSACTION))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class RegexClassifier:
    """Rule-based classifier — no training required."""

    def predict(self, conv: Conversation) -> tuple[str, float]:
        text = conv.as_string

        for category, rules in _RULES:
            for pattern, neg_pattern, confidence in rules:
                if pattern.search(text):
                    if neg_pattern is None or not neg_pattern.search(text):
                        return category, confidence

        return CLEAR_CATEGORY, 1.0
