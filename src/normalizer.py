import re
import difflib
from typing import Iterable, List, Optional, Set


class TextNormalizer:
    """
    Standardizes clinical free-text for drugs and adverse events:
    1. Strips dosage strengths, units, salts (HCl, Besylate, etc.), and formulation types.
    2. Strips clinical severity qualifiers (severe, acute, mild, etc.) from events.
    3. Maps Brand/Trade names and salt variants to Active Pharmaceutical Ingredients (APIs).
    4. Normalizes colloquial symptom descriptions to MedDRA-aligned Preferred Terms (PTs).
    5. Applies fuzzy typo correction against canonical reference vocabularies.
    """

    # Brand / Trade Name -> Generic API Canonical Vocabulary (RxNorm alignment)
    BRAND_TO_GENERIC = {
        "lipitor": "atorvastatin",
        "glucophage": "metformin",
        "zestril": "lisinopril",
        "prinivil": "lisinopril",
        "norvasc": "amlodipine",
        "prilosec": "omeprazole",
        "synthroid": "levothyroxine",
        "levoxyl": "levothyroxine",
        "cozaar": "losartan",
        "neurontin": "gabapentin",
        "microzide": "hydrochlorothiazide",
        "hctz": "hydrochlorothiazide",
        "zoloft": "sertraline",
        "zocor": "simvastatin",
        "singulair": "montelukast",
        "plavix": "clopidogrel",
        "crestor": "rosuvastatin",
        "lasix": "furosemide",
        "tylenol": "acetaminophen",
        "paracetamol": "acetaminophen",
        "advil": "ibuprofen",
        "motrin": "ibuprofen",
        "diovan": "valsartan",
    }

    # Colloquial Terms / Spelling Variations -> MedDRA Preferred Terms (PT)
    EVENT_SYNONYMS = {
        # Gastrointestinal
        "throwing up": "nausea",
        "emesis": "nausea",
        "vomiting": "nausea",
        "nauseous": "nausea",
        "feeling sick": "nausea",
        "upset stomach": "nausea",
        "loose stools": "diarrhea",
        "diarrhoea": "diarrhea",
        "watery stool": "diarrhea",
        # Nervous system
        "head ache": "headache",
        "cephalalgia": "headache",
        "migraine": "headache",
        "lightheaded": "dizziness",
        "lightheadedness": "dizziness",
        "vertigo": "dizziness",
        "dizzy": "dizziness",
        "tiredness": "fatigue",
        "exhaustion": "fatigue",
        "lethargy": "fatigue",
        "malaise": "fatigue",
        "can't sleep": "insomnia",
        "sleeplessness": "insomnia",
        "sleep disorder": "insomnia",
        # Skin
        "skin rash": "rash",
        "hives": "rash",
        "urticaria": "rash",
        "eruption": "rash",
        "itching": "pruritus",
        "itchy skin": "pruritus",
        "itchiness": "pruritus",
        # Respiratory
        "shortness of breath": "dyspnea",
        "breathlessness": "dyspnea",
        "sob": "dyspnea",
        "dyspnoea": "dyspnea",
        # Musculoskeletal
        "joint pain": "arthralgia",
        "pain in joints": "arthralgia",
        "muscle pain": "myalgia",
        "muscle ache": "myalgia",
        "muscle aches": "myalgia",
        # Renal & Hepatic
        "acute kidney failure": "acute kidney injury",
        "renal failure": "acute kidney injury",
        "kidney damage": "acute kidney injury",
        "renal impairment": "acute kidney injury",
        "aki": "acute kidney injury",
        "liver failure": "hepatotoxicity",
        "liver damage": "hepatotoxicity",
        "hepatic injury": "hepatotoxicity",
        "elevated lfts": "hepatotoxicity",
        "elevated lft": "hepatotoxicity",
        "jaundice": "hepatotoxicity",
    }

    CANONICAL_DRUGS = sorted(list(set(BRAND_TO_GENERIC.values())))
    CANONICAL_EVENTS = sorted(list(set(EVENT_SYNONYMS.values())))

    # Regex for stripping dosage strengths, units, salts, hydrates, dosage forms,
    # and administration routes -- all of which are noise for ingredient matching.
    _DOSAGE_SALT_REGEX = re.compile(
        r"\b(\d+(\.\d+)?\s*(mg|mcg|ug|g|ml)|hcl|hydrochloride|dihydrochloride|calcium|potassium|sodium|"
        r"besylate|mesylate|fumarate|maleate|succinate|tartrate|acetate|sulfate|phosphate|"
        r"trihydrate|dihydrate|monohydrate|hemihydrate|anhydrous|micronized|"
        r"tablets|tablet|tabs|tab|capsules|capsule|caps|cap|patch|xr|er|sr|dr|cr|ir|otc|"
        r"oral|iv|im|sc|subcutaneous|sublingual|topical|transdermal|inhalation|inhaled|"
        r"nasal|intranasal|intravenous|intramuscular|rectal|ophthalmic|otic|"
        r"gel|cream|ointment|lotion|solution|soln|suspension|susp|syrup|elixir|"
        r"injection|inj|spray|drops)\b",
        re.IGNORECASE,
    )

    # Regex for stripping clinical severity adjectives
    _SEVERITY_REGEX = re.compile(
        r"\b(severe|mild|moderate|acute|chronic|extreme|slight|constant|intermittent|feeling|feeling of|sensation)\b",
        re.IGNORECASE,
    )

    _PUNCT_REGEX = re.compile(r"[,/_\-\+\(\)\[\]\.\*]")

    @staticmethod
    def _safe_fuzzy(term: str, vocab: Iterable[str], cutoff: float) -> Optional[str]:
        """Fuzzy-match ``term`` against ``vocab``, but only accept a genuine typo.

        The candidate must share ``term``'s first character and be within two
        characters of the same length. This corrects real misspellings
        (``atorvastatn`` -> ``atorvastatin``) while refusing cross-entity jumps
        such as ``valsartan`` -> ``losartan`` that a bare similarity cutoff lets
        through.
        """
        if not term:
            return None
        matches = difflib.get_close_matches(term, list(vocab), n=1, cutoff=cutoff)
        if not matches:
            return None
        candidate = matches[0]
        if candidate[:1] != term[:1]:
            return None
        if abs(len(candidate) - len(term)) > 2:
            return None
        return candidate

    @classmethod
    def clean_text(cls, text: str, strip_severity: bool = False) -> str:
        """Removes punctuation, dosage specifications, salts, and extra whitespace."""
        if not text:
            return ""
        cleaned = cls._DOSAGE_SALT_REGEX.sub(" ", text.lower())
        if strip_severity:
            cleaned = cls._SEVERITY_REGEX.sub(" ", cleaned)
        cleaned = cls._PUNCT_REGEX.sub(" ", cleaned)
        return " ".join(cleaned.split()).strip()

    @classmethod
    def normalize_drug(cls, raw_drug: str) -> str:
        """Cleans, standardizes brands/salts, and applies fuzzy typo fallback."""
        cleaned = cls.clean_text(raw_drug)
        if not cleaned:
            return ""

        # 1. Exact or dictionary synonym match
        if cleaned in cls.BRAND_TO_GENERIC:
            return cls.BRAND_TO_GENERIC[cleaned]
        if cleaned in cls.CANONICAL_DRUGS:
            return cleaned

        # 2. Token-level matching (e.g. "Atorvastatin Oral" -> "atorvastatin")
        for token in cleaned.split():
            if token in cls.BRAND_TO_GENERIC:
                return cls.BRAND_TO_GENERIC[token]
            if token in cls.CANONICAL_DRUGS:
                return token

        # 3. High-confidence typo correction only. A loose cutoff here silently
        #    rewrites one ingredient to another (e.g. valsartan -> losartan), so
        #    the match must look like a spelling error, not a different drug.
        fuzzy = cls._safe_fuzzy(cleaned, cls.CANONICAL_DRUGS, cutoff=0.86)
        if fuzzy:
            return fuzzy

        for token in cleaned.split():
            token_fuzzy = cls._safe_fuzzy(token, cls.CANONICAL_DRUGS, cutoff=0.86)
            if token_fuzzy:
                return token_fuzzy

        return cleaned

    @classmethod
    def normalize_event(cls, raw_event: str) -> str:
        """Standardizes symptom descriptions to MedDRA Preferred Terms with typo handling.

        Severity / temporal qualifiers (``acute``, ``chronic``, ``severe`` ...) are
        only stripped when doing so yields a *recognised* term. For many Preferred
        Terms the qualifier is part of the disease name itself
        (e.g. "Chronic obstructive pulmonary disease", "Acute myocardial
        infarction") and blind stripping produces a mangled non-term.
        """
        raw_clean = " ".join(raw_event.lower().split()).strip()
        if raw_clean in cls.EVENT_SYNONYMS:
            return cls.EVENT_SYNONYMS[raw_clean]
        if raw_clean in cls.CANONICAL_EVENTS:
            return raw_clean

        # Punctuation / dosage cleaning WITHOUT severity stripping. This is the
        # value we fall back to, so a mangled qualifier-stripped string is never
        # emitted as the normalized term.
        base = cls.clean_text(raw_event, strip_severity=False)
        if not base:
            return ""
        if base in cls.EVENT_SYNONYMS:
            return cls.EVENT_SYNONYMS[base]
        if base in cls.CANONICAL_EVENTS:
            return base

        # The severity-stripped variant is only USED when it resolves to a known
        # term (e.g. "severe migraine" -> "migraine" -> "headache").
        stripped = cls.clean_text(raw_event, strip_severity=True)
        if stripped and stripped != base:
            if stripped in cls.EVENT_SYNONYMS:
                return cls.EVENT_SYNONYMS[stripped]
            if stripped in cls.CANONICAL_EVENTS:
                return stripped

        # Phrase / substring search against known colloquial variants.
        for phrase, canonical in cls.EVENT_SYNONYMS.items():
            if phrase in base or phrase in raw_clean or (stripped and phrase in stripped):
                return canonical

        for token in base.split():
            if token in cls.EVENT_SYNONYMS:
                return cls.EVENT_SYNONYMS[token]
            if token in cls.CANONICAL_EVENTS:
                return token

        # High-confidence typo correction only (never cross-term jumps).
        fuzzy = cls._safe_fuzzy(base, cls.CANONICAL_EVENTS, cutoff=0.9)
        if fuzzy:
            return fuzzy

        return base

    @classmethod
    def normalize_drug_set(cls, drugs: Iterable[str]) -> Set[str]:
        result = set()
        for d in drugs:
            norm = cls.normalize_drug(str(d))
            if norm:
                result.add(norm)
        return result

    @classmethod
    def normalize_event_set(cls, events: Iterable[str]) -> Set[str]:
        result = set()
        for e in events:
            norm = cls.normalize_event(str(e))
            if norm:
                result.add(norm)
        return result