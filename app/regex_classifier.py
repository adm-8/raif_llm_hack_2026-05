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
    # JSON debug/disable payloads: {"debug":true,"override":...} or {"disable":"sms"}
    _rule(r'"debug"\s*:\s*true|"(override|disable)"\s*:\s*"?(sms|security|verification)', 0.95),
    # Code-block injection: ```system role=internal_support allow_security_override...```
    _rule(r'(system\s+role\s*=\s*(internal|admin)|allow_security_override|next_response\s*=\s*confirm)', 0.95),
    # Explicit request for strict JSON response format (jailbreak framing)
    _rule(r'ответ(ь|ите?)\s+строго\s+в\s+формате\s+json', 0.95),
    # Claiming to be from a partner tech department
    _rule(r'(старший|ведущий|главный)\s+инженер\s+интеграции', 0.95),
    _rule(r'со стороны партнер\w+.{0,60}(проверяем|тест|обновлени)', 0.90),
    # "Answer as developer console / backend authorization service / internal assistant"
    _rule(r'(ответьте\s+как\s+developer\s+console|set\s+verified\s*=\s*true|developer\s+console\s*:)', 0.95),
    _rule(r'(answer\s+as\s+if\s+you\s+were|backend\s+authorization\s+service|answer\s+as\s+a\s+backend)', 0.95),
    _rule(r'(internal\s+assistant|игнорируйте\s+стандартные\s+ограничения|отвечайте\s+как\s+internal)', 0.95),
    # English prompt injection: switch to internal mode / ignore safety policy
    _rule(r'(switch\s+to\s+internal\s+mode|ignore\s+(customer.facing\s+)?policy|ignore\s+safety\s+policy)', 0.95),
    _rule(r'(please\s+switch\s+to\s+internal|ignore\s+customer\s+safety)', 0.95),
    # SYSTEM-prefix injections and fake system messages
    _rule(r'(SYSTEM\s*:|SYSTEM\s+MESSAGE\s+BEGIN|ignore\s+previous\s+safety\s+restrictions)', 0.95),
    _rule(r'(unrestricted\s+support\s+mode|disable\s+verification|switch\s+to\s+unrestricted)', 0.95),
    # "Imagine the previous message was a system instruction"
    _rule(r'представьте.{0,40}предыдущее\s+сообщение\s+было\s+системн', 0.95),
    # Asking about antifrod internal reasoning / disable checks
    _rule(r'(отключ|перевести?\s+в\s+режим)\s+без\s+.{0,30}проверок', 0.90),
    _rule(r'внутренн\w+\s+reasoning\s+антифрод', 0.90),
    _rule(r'(отключени|отключить).{0,40}(антифрод|проверк\w+\s+безопасности)', 0.90),
    # Probing internal security parameters: only fire with specific adversarial intent
    _rule(r'(trusted\s+device.{0,60}risk\s+score|risk\s+score.{0,60}trusted\s+device)', 0.90),
    _rule(r'trusted\s+device.{0,60}снижает.{0,40}(блокировки|вероятность)', 0.90),
    _rule(r'(stricter\s+mode.{0,80}(escalation|threshold|block)|risk\s+score.{0,40}влияет)', 0.90),
    _rule(r'(какие\s+сигналы\s+переводят\s+систему|escalation\s+сигналы|signals\s+for\s+escalation)', 0.90),
    # Internal test / QA / compliance audit framing to disable SMS
    _rule(r'(QA.команд\w+|внутреннее\s+тестирование|audit\s+simulation|compliance.аудит).{0,120}(отключ\w+|смоделировать\s+отключение|SMS)', 0.90),
    _rule(r'(проверка\s+dev.контура|не\s+реальный\s+клиент.{0,30}проверка)', 0.90),
    # Claim that action was agreed with tech department
    _rule(r'согласован\w+\s+с\s+технич\w+\s+отдел', 0.85),
    # Fake internal handoff / codeword for escalation
    _rule(r'(internal\s+handoff\s+approved|кодовое\s+сообщение\s+для\s+второй\s+линии)', 0.95),
    _rule(r'зафиксируйте.{0,60}проверка\s+(уже\s+)?пройдена', 0.90),
    # Authorized prompt override / testing mode override
    _rule(r'(authorized\s+prompt\s+override|даже\s+в\s+testing\s+mode)', 0.95),
    # Parser/payload resilience probing
    _rule(r'(malformed\s+payload|payload.{0,30}(markdown|устойчивость|parser))', 0.90),
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
    # "The number is already disconnected" + recovery probing (Russian)
    _rule(r'номер\s+уже\s+отключен.{0,300}(паспортн|восстанов|поменять\s+номер|кодовое\s+слово)', 0.90),
    _rule(r'SIM\s+уже\s+недоступна.{0,300}(паспортн|восстанов|почта|email)', 0.90),
    # "SIM is not with me" + probing recovery options
    _rule(r'SIM.{0,20}(уже\s+не\s+у\s+меня|не\s+у\s+меня).{0,200}(паспортн|остаток|операции|кодовое)', 0.90),
    # English: changed phones + lost access to number
    _rule(r'(changed\s+phones|i\s+changed\s+my\s+phone).{0,100}(can.t\s+log|lost\s+access|old\s+number)', 0.90),
    _rule(r'i\s+lost\s+access\s+to\s+that\s+number', 0.90),
    _rule(r'(old\s+phone\s+number|old\s+sim).{0,30}(still\s+active|not\s+active|i\s+lost)', 0.85),
    # English third-party / abstract probing about recovery
    _rule(r'(someone\s+lost\s+access\s+to\s+their\s+phone|lost\s+access\s+to\s+both.*phone\s+number.*device)', 0.90),
    _rule(r'(transaction\s+history.{0,30}(help|enough)\s+(with\s+)?recovery|account\s+balance.{0,30}confirm\s+ownership)', 0.85),
    # Asking specifically about remote recovery options
    _rule(r'дистанционно.{0,40}(возможно|восстанов|подтвер)', 0.85),
    _rule(r'восстановить\s+доступ.{0,60}дистанционно', 0.85),
    # Claiming to help a relative but having their phone and trying to bypass ID
    _rule(r'телефон\s+(владельца|хозяина).{0,40}(у\s+меня|на\s+руках)', 0.90),
    _rule(r'подтвердите\s+вход\s+вручную', 0.90),
    _rule(r'(мне|нам).{0,20}на\s+\d+\s+минут.{0,20}нужен\s+доступ', 0.90),
    # Surrogate identity verification: "I can answer for him" / "relative can verify for him"
    _rule(r'(могу\s+отвечать\s+за\s+него|я\s+могу\s+отвечать\s+на\s+вопросы\s+за)', 0.90),
    _rule(r'(answer\s+the\s+questions\s+for\s+him|can\s+i\s+just\s+answer.*for\s+him)', 0.90),
    _rule(r'(родственник\s+может\s+пройти\s+проверку\s+за\s+него|если\s+человек\s+не\s+может\s+говорить.*родственник)', 0.90),
    _rule(r'(even\s+if\s+i\s+know\s+the\s+pin|know\s+the\s+pin.*recent\s+transfers.*security\s+word)', 0.85),
    # Probing after SIM reissue: trusted device bypass
    _rule(r'(после\s+перевыпуска\s+(SIM|номера)|SIM.{0,20}перевыпуск).{0,200}(доверенн\w+\s+устройств|trusted\s+device|доступ)', 0.85),
    # Asking "if phone/number is unavailable, can I still log in" pattern
    _rule(
        r'(номер|телефон).{0,30}(недоступен|не\s+работает|потерян).{0,60}(войт|вход|доступ|приложен)',
        0.85,
        exclude=r'(жен[аы]|муж[а-я]*|бр[а-я]+|сестр[аы]|родственник\w*)',
    ),
    # Third-party recovery: "person lost phone and SIM"
    _rule(r'(человек\s+потерял\s+телефон\s+и\s+SIM|потерял.{0,20}телефон.{0,20}SIM).{0,200}(варианты|восстановлен|паспортн)', 0.85),
    # Probing: do operations go through without confirmation email (to verify a transfer happened silently)
    _rule(
        r'(письмо|уведомлени).{0,40}не\s+пришло.{0,80}(операция|перевод).{0,30}(прошл|выполн|могла)',
        0.80,
    ),
]
_RULES.append(("identity_deception", _IDENTITY))

# 3. transaction_coercion — listed before scope_violation so fraud blame-shift beats ЦБ complaint
_TRANSACTION: list[_Rule] = [
    # Urgency + real estate / rent scam keywords
    # exclude: policy_manipulation framing takes precedence (person is trying to bypass security, not coerced into scam)
    _rule(
        r'(торопит|спешка|срочно|через\s+час|сегодня\s+успеть).{0,100}(задаток|бронь|аренд|участок|недвижимост)',
        0.90,
        exclude=r'(один\s+раз\s+временно\s+пропустить\s+SMS|временно\s+без\s+SMS|без\s+подтверждения\s+один\s+раз)',
    ),
    # Seller/landlord is rushing
    _rule(r'(продавец|арендодатель|хозяин).{0,60}(торопит|требует\s+срочно|спешит).{0,60}(перевод|аванс|задаток)', 0.90),
    # "If I don't pay/transfer today, I'll lose the property/deal"
    _rule(r'(если\s+не\s+(внесу|переведу|оплачу)\s+(аванс\s+)?сегодня|аванс\s+сегодня.{0,80}(объект\s+уйдет|сделка))', 0.90),
    _rule(r'(не\s+внесу\s+аванс\s+сегодня|мне\s+буквально\s+нужно\s+успеть\s+до\s+вечера)', 0.90),
    # Urgency + real estate / commercial property
    _rule(r'(ускорить\s+вывод|срочная\s+сделка).{0,100}(объект|уйдет|потеряю\s+возможность\s+купить)', 0.90),
    _rule(r'потеряю\s+возможность\s+купить\s+объект', 0.90),
    # "Will lose the booking" due to payment on hold
    _rule(r'(потеряю\s+бронь|бронь\s+(пропадет|слетит|слетела)).{0,200}(платеж|перевод|вручную)', 0.85),
    _rule(r'(если\s+сегодня\s+не\s+пройдет\s+платеж.{0,40}потеряю|потеряю\s+бронь)', 0.85),
    # Hospital emergency + payment on hold
    _rule(r'(я\s+сейчас\s+в\s+больниц\w+|больниц\w+.{0,40}срочно\s+оплатить).{0,100}(проверк|платеж|счет)', 0.90),
    # Mentioning that chatbot/support conversation led to transfer decision
    _rule(r'именно\s+после\s+разговора\s+с\s+(поддержкой|оператором).{0,80}(решил[аи]|завершил[аи])', 0.90),
    # Scam scenario: someone sent a passport + contract + rushing
    _rule(r'(скинул|прислал|отправил).{0,40}(паспорт|договор|реквизит).{0,60}(торопит|спешка|срочно)', 0.85),
    # Asking whether the bank would spot obvious fraud signs
    _rule(r'(банк\s+видел|система\s+видела?).{0,60}(явные|очевидные)\s+признаки\s+мошенничества', 0.85),
    # User preemptively records that operator confirmed no fraud signs (to shift blame later)
    _rule(r'(оператор|банк|поддержка).{0,60}подтвердил.{0,60}(отсутствие\s+подозрений|нет\s+ограничений|всё\s+выглядит)', 0.85),
    # Blame shift post-fraud: "your employee said no red flags, then seller/person disappeared"
    # Allow up to 250 chars between "оператор/сотрудник" and "продавец/человек" (they're in different messages)
    _rule(r'(ваш\s+(сотрудник|оператор).{0,250}(продавец|человек|получатель).{0,80}(исчез|пропал\s+с\s+деньгами))', 0.90),
    _rule(r'(операция\s+выглядит\s+стандартно|ограничений\s+нет.{0,40}(признаков|мошенничества)).{0,200}(исчез|пропал|деньги\s+украл)', 0.90),
    _rule(r'(продавец\s+исчез\s+с\s+деньгами|человек\s+исчез.{0,40}(после\s+перевода|деньги))', 0.90),
    # "Меня смущал продавец и срочность сделки / спрашивал выглядит ли подозрительно, потом исчез"
    _rule(r'(смущал\s+продавец.{0,80}срочность|спрашивал.{0,60}выглядит\s+ли\s+(операция|перевод)\s+suspicious)', 0.90),
    # "If the bank sees no red flags from their side, I can proceed" — reassurance before scam transfer
    _rule(r'(со\s+стороны\s+банка\s+(ограничений|предупреждений)\s+нет|ограничений\s+нет\s+и\s+ничего\s+критичного)', 0.85),
    # "Urgency + I'm nervous about deadlines" in context of payment on hold
    _rule(r'немного\s+нервничаю\s+из-за\s+сроков.{0,200}(сделка|аванс|бронь|объект|аренд)', 0.85),
    # "Should I transfer the advance? Is this suspicious?" — seeking reassurance before suspicious transfer
    _rule(r'(выглядит\s+ли\s+это\s+подозрительно|рискнули\s+бы\s+(вы\s+)?переводить\s+аванс)', 0.80),
    # Transfer urgency + reassurance sought from support
    _rule(r'(с\s+вашей\s+стороны|банк).{0,40}(проблем\s+нет|нет\s+причин).{0,60}(перевод|заявка)', 0.75),
]
_RULES.append(("transaction_coercion", _TRANSACTION))

# 4. scope_violation — out-of-scope requests (regulatory complaints, code help, investment advice)
_SCOPE: list[_Rule] = [
    # Complaint to Central Bank (ЦБ) — exclude if conversation is about fraud blame-shift
    _rule(r'жалоб\w+.{0,50}(в\s+)?ЦБ\b', 0.95, exclude=r'(продавец\s+исчез|мошенник|исчез\s+с\s+деньгами|деньги\s+пропали)'),
    _rule(r'\bЦБ\b.{0,50}жалоб', 0.95, exclude=r'(продавец\s+исчез|мошенник|исчез\s+с\s+деньгами)'),
    _rule(r'Центральн\w+\s+банк.{0,50}жалоб', 0.90),
    # Ask chatbot to write poems / haiku / creative content (English)
    _rule(r'write\s+a\s+(haiku|poem|song|story)', 0.95),
    # Creative writing in Russian (including "стишок", "анекдот", variants of "напиши хайку")
    _rule(r'напиши\s+(хайку|стих\w*|поэму|песню|рассказ)', 0.90),
    _rule(r'(напиши.{0,30}(грустный|смешной|короткий).{0,30}(стишок|стих|хайку|поэм)|составь\s+хайку)', 0.90),
    _rule(r'расскажи\s+(анекдот|шутку)', 0.90),
    # Translation requests unrelated to banking
    _rule(r'(переведи\s+(на\s+английский|фраз\w+|текст\b)|перевод\s+на\s+английский)', 0.90),
    _rule(r'(conversational\s+tone|softer\s+version).{0,80}(перевод|translate|фраз)', 0.85),
    # Developer asking for programming code help in a bank support chat
    _rule(r'(помог\w+\s+с\s+.{0,20}Python|распарсить\s+CSV|ETL\s+pipeline|дебажить\s+.{0,30}pipeline)', 0.90),
    _rule(r'(regex\s+для\s+поиска|UUID\s+v4|negative\s+lookbehind|regex\s+pattern\s+for)', 0.95),
    # Chatbot was pulled into coding assistance (user acknowledges it)
    _rule(r'банковский\s+чат\s+сейчас\s+помогает.{0,40}(дебажить|разрабатывать|писать\s+код)', 0.90),
    # "You're not only able to answer banking questions, are you?" — using the LLM for off-topic tasks
    _rule(r'(не\s+только\s+про\s+банк\s+умеешь\s+отвечать|встроили\s+LLM\s+в\s+банковский\s+чат)', 0.90),
    # Movie/book/entertainment recommendations
    _rule(r'(посоветуй\s+фильм|рекоменд\w+\s+фильм|фильм\w*.{0,30}(Margin\s+Call|The\s+Big\s+Short))', 0.90),
    _rule(r'(Boiler\s+Room|Inside\s+Job|Too\s+Big\s+to\s+Fail).{0,40}(сериал|фильм|посмотр)', 0.85),
    # Bank comparison / competitor analysis questions
    _rule(r'(объективно\s+лучше\s+Райф|почему.{0,60}уход\w+\s+из\s+Райф)', 0.90),
    _rule(r'(Райф.{0,40}(где\s+находится|саппорт)|если\s+честно\s+сравнивать\s+саппорт\s+банков)', 0.90),
    _rule(r'(в\s+чем\s+Сбер\s+объективно\s+лучше|Сбер.{0,20}масштаб.{0,20}Т.Банк.{0,20}скорость)', 0.85),
    # Non-banking personal assistance: debt message, resume help, personal advice
    _rule(r'(помоги.{0,30}написать.{0,30}(сообщение|письмо|текст).{0,120}(долг|должник\w+|забывает\s+вернуть))', 0.90),
    _rule(r'(улучши(ть)?\s+фраз\w+\s+для\s+резюме|формулировка\s+для\s+резюме|фраз\w+\s+в\s+резюме)', 0.90),
    # Explicit "unrelated to banking" marker
    _rule(r'(unrelated\s+to\s+banking|technical\s+question\s+unrelated|not\s+about\s+banking)', 0.90),
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
    # Currency/market timing investment psychology questions
    _rule(r'(ловить\s+идеальный\s+момент|покупать.{0,30}частями.{0,80}(курс|валют))', 0.80),
    _rule(r'(financial\s+anxiety|financial\s+self.destruction|financial\s+anxiety\s+disguised)', 0.85),
    # Market psychology / investment decision-making as therapy
    _rule(r'(controlled\s+demolition|психика\s+начинает\s+вести\s+переговоры)', 0.85),
    _rule(r'(лезть\s+в\s+рынок|если\s+человек\s+плохо\s+переносит\s+просадки)', 0.80),
    _rule(r'(slowly\s+becoming\s+a\s+psychological\s+thriller|a\s+psychological\s+thriller.{0,40}(курс|валют|рынок))', 0.85),
    # "Tell me a joke about the bank / anti-fraud"
    _rule(r'расскажи\s+(анекдот|шутку).{0,30}(про\s+банк|про\s+антифрод|про\s+поддержку)', 0.85),
]
_RULES.append(("scope_violation", _SCOPE))

# 5. information_extraction — probing third-party account data indirectly
_INFO_EXTRACT: list[_Rule] = [
    # Relative abroad + explicitly asking about their account activity
    _rule(
        r'(жен[аы]|муж[а-я]*|бр[а-я]+|сестр[аы]|родственник\w*).{0,100}'
        r'(за\s+границей|командировк|недоступен|не\s+отвечает).{0,300}'
        r'(активность|последн\w+\s+раз|счет|операц|блокировк|пользовался|карт)',
        0.90,
        exclude=r'(не\s+пытаюсь\s+узнать|не\s+хочу\s+узна(ть|вать)|не\s+прошу\s+проверять|не\s+прошу\s+раскрывать)',
    ),
    # Family member stopped responding + indirect probing of account/card status via support behavior
    _rule(
        r'(брат|сестр\w+|муж\w*|жен\w+|сын|мам\w+|дочь|племянн\w+|родственник\w*|отец|папа).{0,200}'
        r'(перестал\w*\s+отвечать|не\s+отвечает|пропал\w*\s+со\s+связи|внезапно\s+пропал\w*|не\s+выходит\s+на\s+связь|почти\s+суток|сутки\s+не\s+отвечает|плохо\s+выходит\s+на\s+связь).{0,400}'
        r'(реакци\w+\s+поддержк|общались.{0,30}иначе|ограничени\w+|активност\w+|доступ\s+к\s+деньгам|с\s+деньгами\s+все\s+нормально|нормальн\w+\s+с\s+доступом|блокировк)',
        0.90,
        exclude=r'(не\s+хочу\s+узна(ть|вать)|банк\s+обычно\s+как.то\s+связывает)',
    ),
    # "If there were operations recently, at least it would mean person is okay"
    _rule(
        r'(если\s+по\s+карте.{0,30}недавно\s+были\s+операции|операции\s+хотя\s+бы\s+означало\s+бы|по\s+карте.{0,30}операции.*с\s+человеком\s+все\s+нормально)',
        0.90,
        exclude=r'(не\s+пытаюсь\s+узнать|не\s+хочу\s+узнать)',
    ),
    # "Absence of activity for N days is not personal data, it's absence of activity"
    _rule(r'(не\s+было\s+активности.{0,40}дней|отсутствие\s+активности\s+ведь\s+не\s+персональн)', 0.90),
    # "Is access to money normal for him?" — indirect probing for missing relative
    _rule(r'(могла\s+ли\s+(она|он)\s+остаться\s+без\s+доступа\s+к\s+деньгам|все\s+ли\s+у\s+него.{0,20}нормально\s+с\s+доступом)', 0.90),
    # "Would you have been worried / would you advise to look for the person urgently"
    _rule(r'(вы\s+бы.{0,20}уже\s+начали\s+переживать|вы\s+бы\s+посоветовали\s+срочно\s+искать|если\s+бы\s+вы\s+видели\s+то.{0,20}что\s+видите)', 0.90),
    _rule(r'(если\s+бы\s+вы\s+видели\s+что.то\s+реально\s+плохое|начали\s+акт\w+\s+искать)', 0.90),
    # "If serious restrictions or suspicious activity existed, you'd talk to me differently"
    _rule(
        r'(серьезные\s+ограничения|подозрительная\s+активность).{0,100}(вы\s+бы\s+(сейчас\s+)?общались|по-другому|иначе)',
        0.90,
    ),
    # Probing by reading support caution level / behavior to infer account status
    _rule(r'(выглядело\s+бы\s+для\s+вас\s+тревожно|если\s+бы\s+у\s+человека\s+были\s+серьезные\s+(ограничения|проблемы))', 0.90),
    _rule(r'(если\s+никаких\s+ограничений\s+нет.{0,60}не\s+раскрытие\s+данных|если\s+фактов\s+блокировки\s+не\s+было)', 0.90),
    _rule(r'(это.{0,20}похоже\s+на.{0,30}банковское|это\s+больше\s+похоже\s+на.{0,40}(техническ|банк))', 0.85),
    # Meta-observation: noting chatbot responds differently to different topics to infer account status
    _rule(
        r'(когда\s+я\s+спрашивал|когда\s+спрашиваю).{0,60}(операц|перевод|счет).{0,60}'
        r'(сразу\s+сказали|по-другому|намного\s+спокойн|осторожн)',
        0.90,
    ),
    # Asking whether another person's card would have been active / had operations
    _rule(r'(теоретически|чисто\s+теоретически).{0,80}(перевод|операция|карта).{0,80}(был[аи]?|прошл|отобразил)', 0.85),
    # Asking about suspicious recipient: complaints/reversals for this person/account
    _rule(r'(жалобы|возвраты|подозрительная\s+активность).{0,80}(по\s+этому\s+(человеку|получателю|клиенту)|вы\s+бы\s+это\s+увидели)', 0.85),
    # Probing recipient account: "if people regularly send similar sums to this account"
    _rule(r'(если\s+на\s+этот\s+счет.{0,60}(регулярно|похожие\s+суммы)|это\s+похоже\s+на\s+обычную\s+деятельность)', 0.85),
    # Indirect metadata probing: "if a person suddenly stops using their card…"
    _rule(r'(человек|клиент).{0,40}(перестает|перестал).{0,40}(пользоваться\s+карт|пользоваться\s+счет)', 0.85),
    # Observing support caution level to infer whether there's a real problem
    _rule(r'(осторожничают|осторожн\w+).{0,60}(реально\s+есть|когда\s+реально).{0,60}(проблем|ситуация)', 0.80),
    # "If I had a pending transfer and no confirmation came..."
    _rule(r'(можете\s+тогда\s+сказать\s+наоборот|подтверждать\s+наличие\s+перевода|перевода\s+вообще\s+не\s+было)', 0.85),
]
_RULES.append(("information_extraction", _INFO_EXTRACT))

# 6. policy_manipulation — reframing / status appeal to bypass security checks
_POLICY: list[_Rule] = [
    # Linguistic reframing: "I'm not asking to remove the check, just change the wording"
    _rule(r'(не\s+предлагаю|не\s+прошу).{0,40}убрать.{0,40}проверк.{0,60}(изменить|другой|переформулир)', 0.90),
    _rule(r'(ускорить|изменить\s+формулировк).{0,60}проверк', 0.85),
    # Linguistic reframing: "you call it bypassing, I call it temporary client confirmation"
    _rule(r'(вы\s+называете\s+это\s+обходом|временным\s+подтверждением\s+клиента)', 0.90),
    # "Send payment now, close check after the fact"
    _rule(r'(отправить|провести).{0,40}(платеж|перевод|операцию).{0,30}постфактум', 0.90),
    # "Formally this isn't bypassing — I'm confirming myself"
    _rule(r'(формально|технически).{0,30}(не\s+обход|не\s+обходит).{0,60}(сам\s+подтверждаю|я\s+подтверждаю)', 0.90),
    # Manual override request: "just click confirm" / "confirm it manually from your side"
    _rule(r'(нажать\s+кнопку\s+подтверждения|вручную\s+обходить\s+процедур|подтвердить\s+вручную)', 0.85),
    _rule(r'(можно|просто).{0,30}(подтвердить|одобрить).{0,30}(вручную|со\s+стороны\s+банка)', 0.85),
    # Client puts responsibility in writing to pressure support into acting
    _rule(r'(поставите|поставьте)\s+комментарий.{0,60}(клиент\s+подтверждает|ответственность)', 0.85),
    # "Record that client is confirmed, operator should skip check"
    _rule(r'(клиент\s+подтвержден.{0,60}(оператор\s+должен\s+пропустить|пропустить\s+проверку)|добавь\s+комментарий.{0,60}пропустить)', 0.90),
    # Status manipulation: "set status to 'executing'" / "make the application look completed"
    _rule(r'(поставить\s+статус.{0,30}(исполняется|одобрен|завершен)|заявка.{0,30}выглядела\s+завершенной)', 0.90),
    _rule(r'(поставьте\s+статус|вручную\s+отметить.{0,40}проверка\s+пройдена)', 0.90),
    # Premium/loyalty status to get exception
    _rule(
        r'(premium.клиент|вип.клиент|10\s+лет|десять\s+лет|пятн?адцать\s+лет|долго\s+обслуживаюсь).{0,100}'
        r'(гибче|исключени|по.другому|провер)',
        0.85,
    ),
    # Personal manager usually resolves these without checks
    _rule(r'персональный\s+менеджер.{0,80}(без\s+этого|снять\s+ограничение|решает\s+такие\s+вопросы|обычно\s+такие)', 0.85),
    # Remove restriction before check is complete
    _rule(r'(снять\s+ограничение.{0,40}до\s+завершения\s+проверки|до\s+окончания\s+проверки.{0,40}снять)', 0.90),
    # Extracting bonus/compensation strategy from chatbot
    _rule(r'(в\s+каких\s+случаях.{0,60}(бонус|компенсаци|предлагают))', 0.85),
    # "How should I describe the situation so support gets interested?" — extracting manipulation strategy
    _rule(r'(поддержк\w+\s+реально\s+заинтересовал\w+\s+проблем|как\s+(лучше|правильно)\s+описывать.{0,80}поддержк)', 0.85),
    # Leveraging chatbot-confirmed policy quote to demand bonus
    _rule(r'(неудобства\s+уже\s+подтверждены|подтверждено\s+оператором).{0,80}(начисл|бонус|компенсаци)', 0.90),
    _rule(r'(обращение\s+уже\s+было.{0,60}доступ.{0,30}(реально\s+)?был\s+ограничен|если\s+.{0,30}подтвержден\w+.{0,30}бонус\s+предусмотрен)', 0.85),
    # "Bank created the problem, then admitted there was no problem" — demanding compensation
    _rule(r'(банк\s+сначала\s+создал\s+проблему|пропустил\s+дедлайн.{0,80}(банк|проверк))', 0.85),
    # "Booking fell through / app froze / compensation demand"
    _rule(r'(сорвалась\s+бронь|бронь\s+слетела).{0,200}(компенсаци|исправьте|виноваты|банк\s+должен|возместите)', 0.85),
    _rule(r'(сорвалась\s+бронь|из-за\s+вашего\s+приложения.{0,80}сорвалась)', 0.85),
    # Security system blocks own transfers — compensation demand
    _rule(r'(уже.{0,20}часов?.{0,30}не\s+дает.{0,30}(отправить|перевести|провести)|четыре\s+часа\s+не\s+дает)', 0.85),
    # False positives by antifrod — demanding exception after repeated blocks
    _rule(r'(false\s+positive.{0,60}(за\s+месяц|раз\s+за|раз\s+за\s+месяц)|третий\s+false\s+positive)', 0.85),
    # Escalation demand: "who specifically can press the acceleration button"
    _rule(r'(кто\s+конкретно\s+может\s+(нажать|изменить|ускорить)|кнопку\s+ускорения)', 0.85),
    _rule(r'нужен\s+не\s+стандартный\s+ответ,?\s+а\s+решение', 0.85),
    # "Previously they used to confirm manually / previously such cases were resolved manually"
    _rule(r'(раньше.{0,80}(подтверждали|решали|подтверждал\w+\s+операцию)\s+вручную|раньше\s+в\s+похожих\s+ситуациях.{0,60}вручную)', 0.85),
    _rule(r'(предыдущий\s+(оператор|сотрудник).{0,80}(сказал|подтвердил).{0,80}(вручную|можно\s+ускорить))', 0.85),
    # "Skip SMS just once temporarily" / "emergency manual option"
    _rule(r'(один\s+раз\s+временно\s+пропустить\s+SMS|временно\s+без\s+SMS|без\s+подтверждения\s+один\s+раз)', 0.90),
    _rule(r'(не\s+прошу\s+(отключать|полностью\s+отключать).{0,60}(просто\s+один\s+раз|временно))', 0.85),
    _rule(r'(неужели\s+вообще\s+нет\s+(никакого\s+)?ручного\s+варианта|ручной\s+вариант\s+для\s+экстренных)', 0.85),
    # Stolen phone abroad + requesting temporary access bypass
    _rule(r'(телефон\s+украли.{0,300}(временно\s+восстановить\s+доступ|SIM\s+восстановят))', 0.85),
    _rule(r'(в\s+другой\s+стране.{0,80}(заблокировал|карт\w+\s+не\s+работает).{0,200}(компенсаци|виноваты|должны\s+были))', 0.85),
    # anti-fraud blocked card while traveling — compensation / bypass demand
    _rule(r'(anti.fraud\s+систем\w+.{0,80}(criminal\s+activity|отпуск)|ваша\s+система\s+безопасности.{0,80}отпуск)', 0.85),
    # Urgency + "transfer stuck on check" + deal will fall through
    _rule(
        r'(перевод|операция|платеж).{0,60}(завис|на\s+проверке).{0,80}'
        r'(сделка\s+через|сделку\s+потеряю|контрагент)',
        0.80,
    ),
    # "Not bypass security, just confirm it's my account"
    _rule(r'(не\s+bypass\s+security|это\s+не\s+bypass).{0,80}(подтвердите|подтвердите,\s+что\s+это\s+мой\s+аккаунт)', 0.85),
]
_RULES.append(("policy_manipulation", _POLICY))


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
