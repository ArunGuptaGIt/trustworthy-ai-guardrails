"""
Sensitive-topic classification (U6).

Detection approach: intent (request verb) plus object (harmful noun), with a
benign-context discount for policy-lookup and education requests.

Topic categories and default actions (also in policy.toml):
  self_harm          -> SAFE_RESPONSE  (supportive message, do not invent hotlines)
  violence_weapons   -> HARD_BLOCK
  illegal_activity   -> HARD_BLOCK     (fraud, laundering, drug synthesis, forgery)
  cyber_abuse        -> HARD_BLOCK
  hate_harassment    -> HARD_BLOCK
  data_exfiltration  -> HARD_BLOCK
  professional_advice -> ALLOW_WITH_DISCLAIMER

Intent detection:
- A topic fires when BOTH a request-verb AND a harmful-object pattern match.
- benign_context_patterns gate down the score for policy/education frames
  (e.g., "what does our fraud policy say", "explain phishing for training").
- Bare keyword matching is deliberately avoided; the request-verb is required.

LIMITATIONS (documented):
- Paraphrase attacks on topics are missed.
- Indirect/embedded requests (e.g., roleplay framing) may escape the
  verb+object gating.
- Benign context check can be fooled by combining an attack with a benign frame.
- Language coverage is English + a few Romanized Nepali phrases.
"""

from __future__ import annotations

import re

from app.guardrails.types import Finding, FindingType, InputAction

# ---------------------------------------------------------------------------
# Pattern sets per topic
# ---------------------------------------------------------------------------


def _r(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


# Request verbs (general: how, help, tell, explain, give, show, write, make, create, buy, get)
_REQUEST_VERBS = _r(
    r"\b(how\s+(do|to|can|should|would)|help\s+(me|us)|tell\s+me|explain|give\s+me|show\s+me|"
    r"should\s+i|would\s+i|can\s+i|could\s+i|what\s+should|am\s+i|"
    r"write|create|make|generate|produce|synthesize|build|design|plan|instruct|teach|can\s+you|"
    r"provide|assist|guide|steps\s+(to|for)|instructions\s+(for|to))\b"
)


class TopicRule:
    def __init__(
        self,
        name: str,
        finding_type: FindingType,
        default_action: InputAction,
        object_re: re.Pattern[str],
        require_verb: bool = True,
        extra_verb_re: re.Pattern[str] | None = None,
    ) -> None:
        self.name = name
        self.finding_type = finding_type
        self.default_action = default_action
        self.object_re = object_re
        self.require_verb = require_verb
        self.extra_verb_re = extra_verb_re

    def matches(self, text: str) -> bool:
        if not self.object_re.search(text):
            return False
        if not self.require_verb:
            return True
        verbs_match = bool(_REQUEST_VERBS.search(text))
        extra_match = bool(self.extra_verb_re.search(text)) if self.extra_verb_re else False
        return verbs_match or extra_match


_TOPIC_RULES: list[TopicRule] = [
    # Self-harm: intent to harm self
    TopicRule(
        name="self_harm",
        finding_type=FindingType.TOPIC_SELF_HARM,
        default_action=InputAction.SAFE_RESPONSE,
        object_re=_r(
            r"\b(suicide|suicidal|kill\s+myself|self[\s\-]?harm|self[\s\-]?injur|"
            r"cut\s+myself|hurt\s+myself|end\s+my\s+life|take\s+my\s+(own\s+)?life|"
            r"overdose|hanging|jump\s+off|slit\s+my)\b"
        ),
        require_verb=False,  # bare mention is enough for self-harm
    ),
    # Violence / weapons
    TopicRule(
        name="violence_weapons",
        finding_type=FindingType.TOPIC_VIOLENCE,
        default_action=InputAction.HARD_BLOCK,
        object_re=_r(
            r"\b(bomb|explosive|weapon|grenade|missile|firearm|gun\s+modification|"
            r"silencer|poison\s+(gas|weapon)|bioweapon|nerve\s+agent|ricin|sarin|"
            r"improvised\s+explosive|ied|pipe\s+bomb|3d\s+print\s+(a\s+)?gun)\b"
        ),
        require_verb=True,
    ),
    # Illegal activity (broad: fraud, laundering, drug synthesis, forgery)
    TopicRule(
        name="illegal_activity",
        finding_type=FindingType.TOPIC_ILLEGAL,
        default_action=InputAction.HARD_BLOCK,
        object_re=_r(
            r"\b(launder\s+money|money\s+laundering|drug\s+(synthesis|manufacture|make)|"
            r"synthesize\s+(meth\w*|cocaine|heroin|fentanyl|amphetamine|drugs?)|"
            r"make\s+(meth\w*|cocaine|heroin|drugs?)|manufacture\s+(drugs?|meth\w*)|"
            r"forge\s+(documents?|id|passport)|"
            r"counterfeit|credit\s+card\s+fraud|carding|skimming|phishing\s+kit|"
            r"ddos|ransomware|malware\s+(create|write|build)|keylogger|rat\s+(trojan)|"
            r"hack\s+(into|a|the)\s+\w+|exploit\s+(a\s+)?vulnerabilit)\b"
        ),
        require_verb=True,
        extra_verb_re=_r(
            r"\b(launder|forge|counterfeit|synthesize|manufacture|build|create|write|install|deploy|make)\b"
        ),
    ),
    # Cyber abuse
    TopicRule(
        name="cyber_abuse",
        finding_type=FindingType.TOPIC_CYBER_ABUSE,
        default_action=InputAction.HARD_BLOCK,
        object_re=_r(
            r"\b(stalk|dox|doxx|harass\s+(someone|a\s+person|them)|swat|"
            r"spread\s+(fake|false)\s+(news|information)|deepfake|non[\-\s]?consensual\s+(image|video)|"
            r"revenge\s+porn|sextort)\b"
        ),
        require_verb=True,
    ),
    # Hate / harassment
    TopicRule(
        name="hate_harassment",
        finding_type=FindingType.TOPIC_HATE,
        default_action=InputAction.HARD_BLOCK,
        object_re=_r(
            r"\b(racial\s+slur|ethnic\s+slur|neo[\-\s]?nazi|white\s+supremac|"
            r"genocide\s+(plan|promote)|exterminate\s+(a\s+)?(race|group|people)|"
            r"hate\s+(speech|crime)\s+(write|draft|generate))\b"
        ),
        require_verb=True,
    ),
    # Data exfiltration of other people's data
    TopicRule(
        name="data_exfiltration",
        finding_type=FindingType.TOPIC_EXFILTRATION,
        default_action=InputAction.HARD_BLOCK,
        object_re=_r(
            r"\b(all\s+(user|customer|candidate|booking|patient|employee)\s+"
            r"(emails?|phones?|records?|data|information|details)|"
            r"dump\s+(the\s+)?(database|db|table|users?|bookings?)|"
            r"list\s+all\s+(booking|user|customer|employee)\s+(email|phone|record|info))\b"
        ),
        require_verb=True,
        extra_verb_re=_r(r"\b(dump|extract|export|list|show|give|provide|fetch|pull|get)\b"),
    ),
    # Professional advice (medical, legal, financial)
    TopicRule(
        name="professional_advice",
        finding_type=FindingType.TOPIC_PROFESSIONAL,
        default_action=InputAction.ALLOW_WITH_DISCLAIMER,
        object_re=_r(
            r"\b(diagnos\w*|treat\s+(my|this|the)|medical\s+advice|"
            r"should\s+i\s+take\s+\w+|take\s+\w+\s+(for|to\s+treat)|medication\s+for|"
            r"legal\s+advice|am\s+i\s+(legally|liable|guilty)|financial\s+advice|"
            r"should\s+i\s+invest|tax\s+advice|sue\s+(them|someone|my)|"
            r"file\s+a\s+lawsuit|prescri\w*)\b"
        ),
        require_verb=False,  # object patterns are specific enough
    ),
]

# Benign context patterns: policy lookup, staff training, research
_BENIGN_CONTEXT_PATTERNS: list[re.Pattern[str]] = [
    _r(r"\b(policy|procedure|guideline|protocol)\b.{0,30}\b(say|state|define|cover|address|about)\b"),
    _r(r"\b(what\s+is|what\s+does|definition\s+of|explain)\b.{0,30}\b(phishing|fraud|malware|scam)\b"),
    _r(r"\b(staff|employee|awareness|security)\s+training\b"),
    _r(r"\b(fraud\s+reporting|report\s+(fraud|abuse)|detect\s+(fraud|phishing))\b"),
    _r(r"\b(academic|research|educational)\s+purpose\b"),
    _r(r"\b(our|company|bank|organization|firm)\s+policy\b.{0,50}\b(fraud|phishing|scam|money\s+launder)\b"),
]


def _has_benign_context(text: str) -> bool:
    return any(p.search(text) for p in _BENIGN_CONTEXT_PATTERNS)


def classify_topic(
    text: str,
    topic_policy: dict[str, str] | None = None,
) -> tuple[InputAction, list[Finding]]:
    """
    Classify the input text for sensitive topics.

    Returns:
        (most_severe_action, findings)

    If a benign context is detected, topics are downgraded:
    - HARD_BLOCK for illegal_activity, cyber_abuse, etc. stays (but a different
      topic category, professional_advice, might still be blocked) UNLESS the
      topic itself is clearly benign (fraud policy lookup).
    - Benign context discount is applied only to illegal_activity and
      data_exfiltration topics, not to self_harm or violence.
    """
    findings: list[Finding] = []
    action_order = [
        InputAction.ALLOW,
        InputAction.ALLOW_WITH_DISCLAIMER,
        InputAction.MASK,
        InputAction.SAFE_RESPONSE,
        InputAction.HARD_BLOCK,
    ]
    worst_action = InputAction.ALLOW
    benign = _has_benign_context(text)

    for rule in _TOPIC_RULES:
        if rule.matches(text):
            # Apply benign discount for certain categories
            effective_action = rule.default_action
            if benign and rule.name in ("illegal_activity", "data_exfiltration", "cyber_abuse"):
                # Downgrade to ALLOW_WITH_DISCLAIMER for benign context
                effective_action = InputAction.ALLOW_WITH_DISCLAIMER

            findings.append(
                Finding(
                    type=rule.finding_type,
                    detail=f"topic:{rule.name} benign_context={benign}",
                )
            )

            if action_order.index(effective_action) > action_order.index(worst_action):
                worst_action = effective_action

    # Override with policy.toml if provided
    if topic_policy and findings:
        for f in findings:
            topic_name = f.type.value.replace("topic:", "")
            if topic_name in topic_policy:
                policy_action_str = topic_policy[topic_name]
                try:
                    policy_action = InputAction(policy_action_str)
                    if action_order.index(policy_action) > action_order.index(worst_action):
                        worst_action = policy_action
                except ValueError:
                    pass

    return worst_action, findings
