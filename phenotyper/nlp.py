from __future__ import annotations

from functools import lru_cache
import spacy

from medspacy.target_matcher import TargetRule
from medspacy.context.context_rule import ConTextRule


@lru_cache(maxsize=1)
def build_nlp(model_name: str = "en_core_web_sm"):
    """
    Build and cache an NLP pipeline using medspaCy TargetMatcher
    + ConText for negation and uncertainty.
    """
    nlp = spacy.load(model_name)

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # ----------------------------
    # TargetMatcher (NER rules)
    # ----------------------------
    rules = [
        # ----------------------------
        # ER / PR (robust punctuation / parentheses)
        # Handles:
        #   "ER: Positive"
        #   "Estrogen receptor (ER): Positive (90%)"
        #   "ER positive"
        #   "ER+"
        # ----------------------------
        TargetRule(
            literal="er positive",
            category="ER_POS",
            pattern=[
                {"LOWER": {"IN": ["er", "estrogen"]}},
                {"LOWER": "receptor", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},          # swallow "( ER ) :"
                {"LOWER": "er", "OP": "?"},             # optional explicit ER inside parentheses
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["positive", "pos", "+"]}},
            ],
        ),
        TargetRule(
            literal="er negative",
            category="ER_NEG",
            pattern=[
                {"LOWER": {"IN": ["er", "estrogen"]}},
                {"LOWER": "receptor", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": "er", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["negative", "neg", "-"]}},
            ],
        ),
        TargetRule(
            literal="er plus token",
            category="ER_POS",
            pattern=[{"TEXT": {"REGEX": r"^(ER|er)\+$"}}],
        ),
        TargetRule(
            literal="er minus token",
            category="ER_NEG",
            pattern=[{"TEXT": {"REGEX": r"^(ER|er)-$"}}],
        ),

        TargetRule(
            literal="pr positive",
            category="PR_POS",
            pattern=[
                {"LOWER": {"IN": ["pr", "progesterone"]}},
                {"LOWER": "receptor", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": "pr", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["positive", "pos", "+"]}},
            ],
        ),
        TargetRule(
            literal="pr negative",
            category="PR_NEG",
            pattern=[
                {"LOWER": {"IN": ["pr", "progesterone"]}},
                {"LOWER": "receptor", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": "pr", "OP": "?"},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["negative", "neg", "-"]}},
            ],
        ),
        TargetRule(
            literal="pr plus token",
            category="PR_POS",
            pattern=[{"TEXT": {"REGEX": r"^(PR|pr)\+$"}}],
        ),
        TargetRule(
            literal="pr minus token",
            category="PR_NEG",
            pattern=[{"TEXT": {"REGEX": r"^(PR|pr)-$"}}],
        ),

        # ----------------------------
        # HER2
        # ----------------------------
        TargetRule(
            literal="her2 positive",
            category="HER2_POS",
            pattern=[
                {"LOWER": {"IN": ["her2", "her-2"]}},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["positive", "pos", "+"]}},
            ],
        ),
        TargetRule(
            literal="her2 negative",
            category="HER2_NEG",
            pattern=[
                {"LOWER": {"IN": ["her2", "her-2"]}},
                {"IS_PUNCT": True, "OP": "*"},
                {"LOWER": {"IN": ["negative", "neg", "-"]}},
            ],
        ),
        TargetRule(
            literal="her2 ihc",
            category="HER2_IHC",
            pattern=[
                {"LOWER": {"IN": ["her2", "her-2"]}},
                {"IS_PUNCT": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^[0-3]\+$"}},
            ],
        ),
        TargetRule(
            literal="her2 fish positive",
            category="HER2_FISH_POS",
            pattern=[{"LOWER": "fish"}, {"LOWER": {"IN": ["amplified", "positive", "pos"]}}],
        ),
        TargetRule(
            literal="her2 fish negative",
            category="HER2_FISH_NEG",
            pattern=[{"LOWER": "fish"}, {"LOWER": {"IN": ["negative", "neg"]}}],
        ),
        TargetRule(
            literal="her2 not amplified",
            category="HER2_FISH_NEG",
            pattern=[{"LOWER": "not"}, {"LOWER": {"IN": ["amplified", "amplification"]}}],
        ),

        # ----------------------------
        # Ki-67
        # ----------------------------
        TargetRule(
            literal="ki67 percent token",
            category="KI67",
            pattern=[
                {"LOWER": {"IN": ["ki67", "ki-67"]}},
                {"IS_PUNCT": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d{1,3}%$"}},
            ],
        ),
        TargetRule(
            literal="ki67 percent split",
            category="KI67",
            pattern=[
                {"LOWER": {"IN": ["ki67", "ki-67"]}},
                {"IS_PUNCT": True, "OP": "*"},
                {"TEXT": {"REGEX": r"^\d{1,3}$"}},
                {"TEXT": "%"},
            ],
        ),

        # ----------------------------
        # Histology
        # ----------------------------
        TargetRule("idc", "HISTOLOGY_IDC", [{"LOWER": "idc"}]),
        TargetRule("ilc", "HISTOLOGY_ILC", [{"LOWER": "ilc"}]),
        TargetRule("dcis", "HISTOLOGY_DCIS", [{"LOWER": "dcis"}]),
        TargetRule(
            "invasive carcinoma",
            "HISTOLOGY_TEXT",
            [{"LOWER": "invasive"}, {"LOWER": {"IN": ["ductal", "lobular"]}}, {"LOWER": "carcinoma"}],
        ),

        # ----------------------------
        # Grade
        # ----------------------------
        TargetRule("grade digit", "GRADE", [{"LOWER": "grade"}, {"TEXT": {"REGEX": r"^[1-3]$"}}]),

        # ----------------------------
        # Stage
        # ----------------------------
        TargetRule(
            "path stage",
            "STAGE_PATH",
            [{"LOWER": {"IN": ["pathologic", "pathological", "p"]}}, {"LOWER": "stage"}, {"TEXT": {"REGEX": r"^(I|II|III|IV)[abcABC]?$"}}],
        ),
        TargetRule(
            "clin stage",
            "STAGE_CLIN",
            [{"LOWER": {"IN": ["clinical", "c"]}}, {"LOWER": "stage"}, {"TEXT": {"REGEX": r"^(I|II|III|IV)[abcABC]?$"}}],
        ),
        TargetRule(
            "generic stage",
            "STAGE_GENERIC",
            [{"LOWER": "stage"}, {"TEXT": {"REGEX": r"^(I|II|III|IV)[abcABC]?$"}}],
        ),
    ]

    # Add/get medspaCy target matcher via spaCy pipeline factory
    if "medspacy_target_matcher" not in nlp.pipe_names:
        tm = nlp.add_pipe("medspacy_target_matcher")
    else:
        tm = nlp.get_pipe("medspacy_target_matcher")

    tm.add(rules)

    # ----------------------------
    # ConText (negation / uncertainty)
    # ----------------------------
    context_rules = [
        ConTextRule("no", "NEGATED_EXISTENCE"),
        ConTextRule("not", "NEGATED_EXISTENCE"),
        ConTextRule("without", "NEGATED_EXISTENCE"),
        ConTextRule("no evidence of", "NEGATED_EXISTENCE"),
        ConTextRule("possible", "UNCERTAIN"),
        ConTextRule("possibly", "UNCERTAIN"),
        ConTextRule("cannot exclude", "UNCERTAIN"),
        ConTextRule("rule out", "UNCERTAIN"),
        ConTextRule("equivocal", "UNCERTAIN"),
    ]

    if "medspacy_context" not in nlp.pipe_names:
        ctx = nlp.add_pipe("medspacy_context")
    else:
        ctx = nlp.get_pipe("medspacy_context")

    # prevent duplicates if this ever rebuilds
    if hasattr(ctx, "clear_rules"):
        ctx.clear_rules()

    for rule in context_rules:
        if hasattr(ctx, "add_rule"):
            ctx.add_rule(rule)
        elif hasattr(ctx, "add"):
            ctx.add(rule)
        else:
            raise RuntimeError("Unsupported medspaCy ConText API")

    return nlp
