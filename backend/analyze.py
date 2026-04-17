"""Rules-engine compliance pipeline: five deterministic gates, LLM only for ambiguity.

Gate 1 — Blacklist (ECGT Annex I per-se bans). Pure Python, instant.
Gate 2 — Neutrality/offset claims (4a + 4c dual violation). Pure Python.
Gate 3 — Certified specific claims. Pure Python. LOW risk if cert covers claim.
Gate 4 — Comparative claims. Python detection + focused LLM check.
Gate 5 — Everything else. RAG retrieval + LLM for genuine ambiguity.
"""

import json
import logging
import os
import re
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

log = logging.getLogger("compliance.gates")
if os.environ.get("GATE_DEBUG"):
    logging.basicConfig(level=logging.DEBUG, format="  [GATE] %(message)s")
else:
    logging.basicConfig(level=logging.WARNING)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "eu_regulations"

client = OpenAI()


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# ECGT Recital 9 / Annex I point 4a — banned generic environmental claims
ECGT_BLACKLIST = [
    "eco-friendly", "ecofriendly",
    "environmentally friendly",
    "green",
    "nature's friend", "natures friend",
    "ecological",
    "climate friendly",
    "carbon friendly",
    "gentle on the environment",
    "biodegradable",
    "biobased",
    "energy efficient",
    "sustainable", "sustainably",
    "responsible", "responsibly",
    "conscious",
    "natural",
    "better for the planet",
    "better for the environment",
    "good for the planet",
    "kind to the environment",
    "circular",
    "zero waste",
    "renewable",
    "lower environmental impact",
    "low carbon",
    "low water",
    "organic",
]

# ECGT Annex I point 4c / Recital 12 — offset-based neutrality claims
NEUTRALITY_TERMS = [
    "carbon neutral", "climate neutral", "climate positive",
    "net zero", "carbon negative", "CO2 neutral",
    "carbon compensated", "carbon balanced", "carbon positive",
    "carbon offset", "climate compensated", "net positive",
]

# Certification scope — what each cert actually covers
CERTIFICATION_SCOPE = {
    "TENCEL": [
        "closed loop production",
        "solvent reused", "solvent recovery",
        "tencel lyocell", "tencel modal",
        "sustainably sourced wood",
        "eucalyptus", "beechwood",
    ],
    "GOTS": [
        "organic cotton", "organic wool", "organic linen",
        "organic fiber", "organic farming",
        "organic processing", "organic",
    ],
    "GRS": [
        "recycled polyester", "recycled nylon",
        "recycled wool", "recycled cotton",
        "recycled content", "recycled materials",
        "post-consumer recycled", "post-industrial recycled",
        "recycled",
    ],
    "Bluesign": [
        "responsible chemistry", "bluesign certified",
        "resource efficiency", "safer production",
        "chemical safety", "bluesign",
    ],
    "FSC": [
        "fsc certified", "responsibly sourced wood",
        "fsc viscose", "fsc lyocell",
        "sustainably sourced wood pulp",
    ],
    "OEKO-TEX 100": [
        "free from harmful substances",
        "tested for harmful substances",
        "oeko-tex certified", "safe for skin",
    ],
    "RWS": [
        "responsible wool", "rws certified",
        "animal welfare", "land management",
    ],
    "EU Ecolabel": ["__ALL_GENERIC__"],
    "Nordic Swan": ["__ALL_GENERIC__"],
    "Blue Angel": ["__ALL_GENERIC__"],
    "Fair Trade": ["fair trade", "fairly traded", "fair wages"],
    "B Corp": ["certified b corp", "b corp certified"],
}

# ISO 14024 Type I ecolabels that unlock generic environmental claims
TYPE1_ECOLABELS = {"EU Ecolabel", "Nordic Swan", "Blue Angel"}

# Certs with uncertain ECGT compliance — do not auto-resolve to LOW
UNCERTAIN_CERTS = {"B Corp", "SBTi", "Carbon credits"}

# Comparative claim patterns — require comparison language, not just a percentage
COMPARATIVE_PATTERNS = [
    r"\d+\s*%\s*(?:less|fewer|lower|reduction|saving)",
    r"less\s+(?:water|energy|carbon|emissions|waste)\s+than",
    r"lower\s+(?:impact|footprint|emissions|carbon)\s+than",
    r"compared\s+to\s+(?:conventional|standard|average|traditional)",
    r"better\s+than\s+(?:conventional|standard|average)",
]

# Self-comparison terms — own-product comparisons go to Gate 5
SELF_COMPARISON_TERMS = [
    "previous model", "our own", "our previous", "prior version",
    "earlier model", "own product",
]

VERIFICATION_PROVIDERS = [
    "Carbonfact", "Ecochain", "Higg Index", "SGS", "Bureau Veritas",
]

UNDEFINED_BASELINES = [
    "conventional", "standard", "average", "traditional",
    "typical", "normal", "regular", "ordinary",
    "industry average", "industry standard",
]

# RAG regulation names — must match ChromaDB metadata exactly
REGULATION_NAMES = {
    "ECGT": "EU Green Claims Directive (ECGT) - Directive 2024/825",
    "UCPD": "Unfair Commercial Practices Directive (UCPD) - Directive 2005/29/EC",
    "ESPR": "Ecodesign for Sustainable Products Regulation (ESPR) - Regulation 2024/1781",
}

SEVERITY_LABELS = {
    "ECGT": "BANNED (per-se violation)",
    "UCPD": "MISLEADING (case-by-case)",
    "ESPR": "ECODESIGN (product-specific)",
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_claims(claims_text: str) -> list[str]:
    """Split a claims string into individual claims."""
    if not claims_text:
        return []
    parts = re.split(r"[,;|]", claims_text)
    return [p.strip() for p in parts if p.strip()]


def _parse_certifications(certs_text: str) -> list[str]:
    """Parse certification text into a list of recognized cert names."""
    if not certs_text:
        return []
    raw = [c.strip() for c in re.split(r"[,;|]", certs_text) if c.strip()]
    if len(raw) == 1 and raw[0].lower().startswith("none"):
        return []
    certs = []
    for raw_cert in raw:
        raw_lower = raw_cert.lower()
        matched = False
        for known in CERTIFICATION_SCOPE:
            if known.lower() in raw_lower:
                certs.append(known)
                matched = True
                break
        if not matched:
            certs.append(raw_cert)
    return certs


# ═══════════════════════════════════════════════════════════════════════
# Gate 1 — Blacklist (ECGT Annex I per-se bans)
# ═══════════════════════════════════════════════════════════════════════

def gate_blacklist(claim: str, certifications: list[str]) -> dict | None:
    """Check claim against ECGT Annex I blacklist.

    Returns result dict if resolved, None if claim should proceed to next gate.
    """
    log.debug("Gate 1 BLACKLIST — claim=%r, certs=%s", claim, certifications)
    claim_lower = claim.lower()
    claim_normalized = claim_lower.replace("-", " ")

    # Find blacklist match (normalize hyphens for both sides)
    matched = None
    for term in ECGT_BLACKLIST:
        term_normalized = term.replace("-", " ")
        if term_normalized in claim_normalized:
            matched = term
            break

    if not matched:
        return None

    # Override: Type I ecolabel unlocks generic claims
    for cert in certifications:
        if cert in TYPE1_ECOLABELS:
            return {
                "claim_type": "BLACKLISTED",
                "overall_risk": "LOW",
                "findings": [{
                    "risk": "LOW",
                    "regulation": "ECGT Directive (EU) 2024/825",
                    "article": "Annex I point 4a / Article 2(s)",
                    "issue": (
                        f"Generic claim '{matched}' is normally banned under Recital 9, "
                        f"but {cert} demonstrates recognised excellent environmental "
                        f"performance per Article 2(s), unlocking this claim."
                    ),
                    "fix": f"Compliant. Ensure {cert} certification is displayed on same medium.",
                    "compliant_rewrite": None,
                }],
                "gate_resolved_at": 1,
            }

    # Skip if claim text mentions a certification name — not standalone
    for cert_name in CERTIFICATION_SCOPE:
        if cert_name.lower() in claim_lower:
            return None
    if "certified" in claim_lower:
        return None

    # Standalone generic claim → HIGH
    return {
        "claim_type": "BLACKLISTED",
        "overall_risk": "HIGH",
        "findings": [{
            "risk": "HIGH",
            "regulation": "ECGT Directive (EU) 2024/825",
            "article": "Annex I point 4a / Recital 9",
            "issue": (
                f"'{matched}' is explicitly listed in ECGT Recital 9 as a banned "
                f"generic environmental claim. Prohibited from September 27 2026 "
                f"in all EU consumer-facing communications."
            ),
            "fix": (
                "Remove entirely. Cannot be used without EU Ecolabel or "
                "equivalent ISO 14024 Type I certification."
            ),
            "compliant_rewrite": None,
        }],
        "gate_resolved_at": 1,
    }


# ═══════════════════════════════════════════════════════════════════════
# Gate 2 — Neutrality / offset claims (dual violation)
# ═══════════════════════════════════════════════════════════════════════

def gate_neutrality(claim: str) -> dict | None:
    """Check for offset-based neutrality terms. Always returns dual violation."""
    log.debug("Gate 2 NEUTRALITY — claim=%r", claim)
    claim_lower = claim.lower()

    matched = None
    for term in NEUTRALITY_TERMS:
        if term in claim_lower:
            matched = term
            break

    if not matched:
        return None

    return {
        "claim_type": "NEUTRALITY",
        "overall_risk": "HIGH",
        "findings": [
            {
                "risk": "HIGH",
                "regulation": "ECGT Directive (EU) 2024/825",
                "article": "Annex I point 4a / Recital 9",
                "issue": (
                    f"Generic environmental claim '{matched}' — banned under "
                    f"Annex I point 4a unless recognised excellent environmental "
                    f"performance demonstrated via EU Ecolabel or equivalent "
                    f"ISO 14024 Type I scheme."
                ),
                "fix": (
                    "No standard certification unlocks this claim. "
                    "EU Ecolabel would be required."
                ),
                "compliant_rewrite": None,
            },
            {
                "risk": "HIGH",
                "regulation": "ECGT Directive (EU) 2024/825",
                "article": "Annex I point 4c / Recital 12",
                "issue": (
                    f"Offset-based neutrality claim '{matched}' — banned under "
                    f"Annex I point 4c if based on carbon credits outside the "
                    f"product's own value chain. Cannot be remedied by any "
                    f"certification."
                ),
                "fix": (
                    "Remove entirely unless claim is based solely on verified "
                    "actual emissions reductions within the product's own value "
                    "chain, independently verified, with methodology publicly "
                    "disclosed."
                ),
                "compliant_rewrite": None,
            },
        ],
        "gate_resolved_at": 2,
    }


# ═══════════════════════════════════════════════════════════════════════
# Gate 3 — Certified specific claims
# ═══════════════════════════════════════════════════════════════════════

def gate_certified_specific(claim: str, certifications: list[str]) -> dict | None:
    """Check if claim is a specific factual claim covered by a recognised cert."""
    log.debug("Gate 3 CERTIFIED_SPECIFIC — claim=%r, certs=%s", claim, certifications)
    if not certifications:
        log.debug("Gate 3 — no certifications, skipping")
        return None

    claim_lower = claim.lower()

    for cert in certifications:
        if cert in UNCERTAIN_CERTS:
            continue  # B Corp, SBTi etc. go to Gate 5 for nuanced assessment
        if cert not in CERTIFICATION_SCOPE:
            continue
        covered_terms = CERTIFICATION_SCOPE[cert]

        # EU Ecolabel / Nordic Swan / Blue Angel unlock ALL generic claims
        if "__ALL_GENERIC__" in covered_terms:
            return {
                "claim_type": "CERTIFIED_SPECIFIC",
                "overall_risk": "LOW",
                "findings": [{
                    "risk": "LOW",
                    "regulation": "ECGT Directive (EU) 2024/825",
                    "article": "Annex I point 4a / Article 2(s)",
                    "issue": None,
                    "fix": (
                        f"Compliant. {cert} demonstrates recognised excellent "
                        f"environmental performance. Ensure certification is "
                        f"displayed on same medium."
                    ),
                    "compliant_rewrite": None,
                }],
                "gate_resolved_at": 3,
            }

        # Check if claim contains a term covered by this cert
        for term in covered_terms:
            if term in claim_lower:
                return {
                    "claim_type": "CERTIFIED_SPECIFIC",
                    "overall_risk": "LOW",
                    "findings": [{
                        "risk": "LOW",
                        "regulation": "ECGT Directive (EU) 2024/825",
                        "article": "Annex I point 4a / Article 2(p)",
                        "issue": None,
                        "fix": (
                            f"Compliant. Specific claim covered by {cert} "
                            f"certification. Ensure {cert} is displayed on same "
                            f"medium as this claim."
                        ),
                        "compliant_rewrite": None,
                    }],
                    "gate_resolved_at": 3,
                }

    # Check if claim explicitly names a cert the product holds
    for cert in certifications:
        if cert in UNCERTAIN_CERTS:
            continue
        if cert.lower() in claim_lower:
            return {
                "claim_type": "CERTIFIED_SPECIFIC",
                "overall_risk": "LOW",
                "findings": [{
                    "risk": "LOW",
                    "regulation": "ECGT Directive (EU) 2024/825",
                    "article": "Annex I point 4a / Article 2(p)",
                    "issue": None,
                    "fix": f"Compliant. {cert} certification substantiates this claim.",
                    "compliant_rewrite": None,
                }],
                "gate_resolved_at": 3,
            }

    # Partial substantiation: claim contains a blacklisted generic term but
    # product holds a recognised (non-uncertain) certification → MEDIUM.
    # The cert doesn't exactly cover the claim wording, but provides partial
    # substantiation that can make the claim compliant if displayed on same medium.
    claim_normalized = claim_lower.replace("-", " ")
    blacklisted_term = None
    for term in ECGT_BLACKLIST:
        if term.replace("-", " ") in claim_normalized:
            blacklisted_term = term
            break

    if blacklisted_term:
        relevant_certs = [c for c in certifications
                          if c in CERTIFICATION_SCOPE and c not in UNCERTAIN_CERTS]
        if relevant_certs:
            cert = relevant_certs[0]
            log.debug("Gate 3 — partial substantiation: blacklisted '%s' + cert %s → MEDIUM",
                       blacklisted_term, cert)
            return {
                "claim_type": "CERTIFIED_SPECIFIC",
                "overall_risk": "MEDIUM",
                "findings": [{
                    "risk": "MEDIUM",
                    "regulation": "ECGT Directive (EU) 2024/825",
                    "article": "Annex I point 4a / Article 2(p)",
                    "issue": (
                        f"Generic environmental claim '{blacklisted_term}' is "
                        f"normally banned under Annex I point 4a. However {cert} "
                        f"certification provides partial substantiation. The broad "
                        f"claim still needs specification on same medium."
                    ),
                    "fix": (
                        f"Potentially compliant if {cert} certification is displayed "
                        f"on same medium as the claim and clearly linked to it. "
                        f"Verify {cert} covers the specific aspect being claimed."
                    ),
                    "compliant_rewrite": None,
                }],
                "gate_resolved_at": 3,
            }

    return None


# ═══════════════════════════════════════════════════════════════════════
# Gate 4 — Comparative claims (Python detection + focused LLM)
# ═══════════════════════════════════════════════════════════════════════

def _is_comparative(claim: str) -> bool:
    """Return True if claim matches a comparative pattern."""
    claim_lower = claim.lower()
    return any(re.search(p, claim_lower) for p in COMPARATIVE_PATTERNS)


def _is_self_comparison(claim: str) -> bool:
    """Return True if claim compares to the trader's own product."""
    claim_lower = claim.lower()
    return any(term in claim_lower for term in SELF_COMPARISON_TERMS)


def _is_verified_comparative(claim: str) -> bool:
    """Return True if claim mentions recognized verifier + published methodology."""
    claim_lower = claim.lower()
    has_verifier = any(v.lower() in claim_lower for v in VERIFICATION_PROVIDERS)
    has_methodology = any(kw in claim_lower for kw in [
        "published at", "methodology at", "at url",
        "published url", "published methodology",
    ])
    return has_verifier and has_methodology


GATE4_PROMPT = """A fashion brand makes this comparative environmental claim:
Claim: "{claim}"
Certifications held: {certs}

Check exactly three things and answer YES or NO for each:

1. VERIFICATION: Is there evidence of independent third-party verification
   of this specific measurement? (Recognised LCA providers — Carbonfact,
   Ecochain, Higg Index, SGS, Bureau Veritas — count. Internal testing does not.)

2. BASELINE: Is the comparison baseline specifically defined?
   (Must name specific product, year, or methodology.
   "conventional", "standard", "average", "traditional" alone
   are NOT defined baselines.)

3. METHODOLOGY: Is the measurement methodology publicly accessible?
   (A published URL, footnote, or accessible document counts.)

Also: if the claim references a verification label or scheme, is it a recognised
third-party certification or a self-created brand label? Self-created labels
violate ECGT Annex I point 4b.

Always cite ECGT Directive (EU) 2024/825 as the primary regulation.
For comparative/misleading claims, cite UCPD 2005/29/EC as amended by ECGT.

Output valid JSON only:
{{
  "verification": {{"present": true, "detail": "..."}},
  "baseline": {{"defined": true, "detail": "..."}},
  "methodology": {{"public": true, "detail": "..."}},
  "self_created_label": {{"present": false, "label_name": null}},
  "overall_risk": "HIGH"
}}"""


def gate_comparative(claim: str, certifications: list[str]) -> dict | None:
    """Detect and assess comparative claims. Uses focused LLM for unverified ones."""
    log.debug("Gate 4 COMPARATIVE — claim=%r", claim)
    if not _is_comparative(claim):
        log.debug("Gate 4 — not a comparative claim, skipping")
        return None

    # Self-comparisons go to Gate 5 for nuanced assessment
    if _is_self_comparison(claim):
        return None

    # Verified comparative claims → LOW
    if _is_verified_comparative(claim):
        return {
            "claim_type": "COMPARATIVE",
            "overall_risk": "LOW",
            "findings": [{
                "risk": "LOW",
                "regulation": "UCPD Directive 2005/29/EC",
                "article": "Article 6(1) / Article 7(7)",
                "issue": None,
                "fix": (
                    "Compliant. Third-party verification and published "
                    "methodology present."
                ),
                "compliant_rewrite": None,
            }],
            "gate_resolved_at": 4,
        }

    # Unverified comparative → focused LLM check
    certs_str = ", ".join(certifications) if certifications else "None"
    prompt = GATE4_PROMPT.format(claim=claim, certs=certs_str)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f'Assess: "{claim}"'},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        llm_result = json.loads(response.choices[0].message.content)
    except Exception:
        llm_result = {
            "verification": {"present": False, "detail": "Could not assess"},
            "baseline": {"defined": False, "detail": "Could not assess"},
            "methodology": {"public": False, "detail": "Could not assess"},
            "self_created_label": {"present": False},
            "overall_risk": "HIGH",
        }

    findings = []

    # Self-created label check
    if llm_result.get("self_created_label", {}).get("present"):
        label_name = llm_result["self_created_label"].get("label_name", "unknown")
        findings.append({
            "risk": "HIGH",
            "regulation": "ECGT Directive (EU) 2024/825",
            "article": "Annex I point 4b",
            "issue": (
                f"Self-created sustainability label '{label_name}' is not based "
                f"on a certification scheme with independent third-party "
                f"verification. Violates Annex I point 4b."
            ),
            "fix": "Replace with a recognised third-party certification scheme.",
            "compliant_rewrite": None,
        })

    if not llm_result.get("verification", {}).get("present", False):
        detail = llm_result.get("verification", {}).get("detail", "")
        findings.append({
            "risk": "HIGH",
            "regulation": "UCPD Directive 2005/29/EC as amended by ECGT",
            "article": "Article 6(1)",
            "issue": (
                f"Comparative claim requires independent third-party "
                f"verification. {detail}"
            ),
            "fix": (
                "Commission independent verification from a recognised "
                "provider (Carbonfact, Ecochain, Higg Index, SGS, "
                "Bureau Veritas)."
            ),
            "compliant_rewrite": None,
        })

    if not llm_result.get("baseline", {}).get("defined", False):
        detail = llm_result.get("baseline", {}).get("detail", "")
        findings.append({
            "risk": "HIGH",
            "regulation": "UCPD Directive 2005/29/EC as amended by ECGT",
            "article": "Article 7(7)",
            "issue": (
                f"Comparison baseline is not specifically defined. {detail}"
            ),
            "fix": (
                "Replace vague comparator with specific named baseline: "
                "product, year, and measurement scope must be stated."
            ),
            "compliant_rewrite": None,
        })

    if not llm_result.get("methodology", {}).get("public", False):
        detail = llm_result.get("methodology", {}).get("detail", "")
        findings.append({
            "risk": "MEDIUM",
            "regulation": "UCPD Directive 2005/29/EC as amended by ECGT",
            "article": "Article 7(7)",
            "issue": (
                f"Methodology is not publicly accessible. {detail}"
            ),
            "fix": (
                "Publish full methodology as a public URL or product "
                "page footnote."
            ),
            "compliant_rewrite": None,
        })

    overall = llm_result.get("overall_risk", "HIGH")
    if not findings:
        overall = "LOW"
        findings = [{
            "risk": "LOW",
            "regulation": "UCPD Directive 2005/29/EC",
            "article": "Article 7(7)",
            "issue": None,
            "fix": "Compliant comparative claim.",
            "compliant_rewrite": None,
        }]

    return {
        "claim_type": "COMPARATIVE",
        "overall_risk": overall,
        "findings": findings,
        "gate_resolved_at": 4,
    }


# ═══════════════════════════════════════════════════════════════════════
# Gate 5 — Everything else (RAG + LLM)
# ═══════════════════════════════════════════════════════════════════════

GATE5_PROMPT = """You are an EU fashion compliance expert with access to the actual
regulation text below.

Claim: "{claim}"
Certifications: {certs}

Based ONLY on the regulation text provided, assess compliance.
This claim has already been checked against:
- ECGT blacklist (not blacklisted)
- Offset/neutrality ban (not a neutrality claim)
- Certified specific pathway (not covered by certification)
- Comparative claim rules (not a comparative claim)

So this is a genuinely ambiguous case. Reason carefully from the actual text.
If you cannot determine compliance from the provided text, say so explicitly
rather than guessing.

IMPORTANT — Regulation citation:
Always cite ECGT Directive (EU) 2024/825 as the primary regulation for
environmental claims. ECGT amends UCPD Directive 2005/29/EC. For
comparative/misleading claims, cite UCPD with "as amended by ECGT."

Known nuances from Commission Q&A (November 2025):
- "Vegan" alone is a factual product characteristic (animal-free composition),
  NOT an environmental claim. Only becomes environmental if paired with benefit
  language like "better for the planet" (Q15).
- Implicit claims (green leaf logos) without accompanying text are NOT generic
  environmental claims. Combined with text, they are assessed together (Q5).
- Collection or product names containing "eco", "green", "sustainable" create
  environmental association and are subject to the same rules as explicit
  claims (Q3).
- Forward-looking commitments require: (1) clear measurable targets, (2)
  detailed implementation plan, (3) regular independent third-party
  verification (Q12). SBTi commitment alone is insufficient.
- Self-created sustainability labels violate Annex I point 4b unless based
  on a certification scheme with independent third-party verification (Q8).
- B Corp certification is under review for ECGT compliance — flag as
  MEDIUM (Q8).
- Comparison to the trader's own previous product is potentially compliant
  if baseline is specifically identified and methodology disclosed (Q13).
- "Organic" without certification is unsubstantiated under UCPD Article 6
  — MEDIUM risk (Q14).
- Chrome-free leather "better for environment" may violate Annex I point 4d
  if chrome-free is a legal requirement presented as a distinctive feature.

Regulation text:
{regulation_text}

Return valid JSON only:
{{
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "regulation": "full regulation name",
  "article": "specific article reference",
  "issue": "description of compliance issue, or null if compliant",
  "fix": "recommended fix, or 'Compliant' if no fix needed",
  "reasoning": "brief explanation citing the regulation text"
}}"""


def gate_llm(claim: str, certifications: list[str], product_name: str) -> dict:
    """LLM + RAG for genuinely ambiguous claims. Always returns a result."""
    log.debug("Gate 5 LLM — claim=%r, certs=%s (calling GPT-4o)", claim, certifications)
    query = f"{claim} {' '.join(certifications)} {product_name}".strip()
    try:
        chunks = retrieve_relevant_chunks(query)
        reg_text = _format_regulation_context(chunks)
    except Exception:
        reg_text = "(Regulation text not available — assess based on your knowledge.)"

    certs_str = ", ".join(certifications) if certifications else "None"
    prompt = GATE5_PROMPT.format(
        claim=claim, certs=certs_str, regulation_text=reg_text,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f'Assess: "{claim}"'},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        llm_result = json.loads(response.choices[0].message.content)
    except Exception as e:
        llm_result = {
            "risk_level": "MEDIUM",
            "regulation": "ECGT Directive (EU) 2024/825",
            "article": "Unknown",
            "issue": f"Could not assess: {e}",
            "fix": "Manual review required.",
            "reasoning": "",
        }

    risk = llm_result.get("risk_level", "MEDIUM")

    return {
        "claim_type": "AMBIGUOUS",
        "overall_risk": risk,
        "findings": [{
            "risk": risk,
            "regulation": llm_result.get("regulation", "ECGT Directive (EU) 2024/825"),
            "article": llm_result.get("article", ""),
            "issue": llm_result.get("issue"),
            "fix": llm_result.get("fix", ""),
            "compliant_rewrite": None,
        }],
        "gate_resolved_at": 5,
    }


# ═══════════════════════════════════════════════════════════════════════
# RAG Retrieval
# ═══════════════════════════════════════════════════════════════════════

def get_collection():
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_collection(name=COLLECTION_NAME)


def retrieve_relevant_chunks(query: str, per_regulation: int = 3) -> list[dict]:
    """Query ChromaDB per regulation for balanced retrieval."""
    collection = get_collection()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = response.data[0].embedding

    chunks = []
    for reg_key, reg_name in REGULATION_NAMES.items():
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=per_regulation,
            where={"regulation_name": reg_name},
        )
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "regulation_key": reg_key,
            })

    return chunks


def _format_regulation_context(chunks: list[dict]) -> str:
    """Format retrieved chunks grouped by regulation."""
    grouped: dict[str, list] = {}
    for chunk in chunks:
        key = chunk["regulation_key"]
        grouped.setdefault(key, []).append(chunk)

    parts = []
    for reg_key in ["ECGT", "UCPD", "ESPR"]:
        if reg_key not in grouped:
            continue
        severity = SEVERITY_LABELS[reg_key]
        parts.append(f"=== {reg_key} — {severity} ===")
        for chunk in grouped[reg_key]:
            meta = chunk["metadata"]
            parts.append(
                f"  Article {meta.get('article_number', '?')}: "
                f"{meta.get('article_title', '')}\n"
                f"  {chunk['text']}\n"
            )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def analyze_claim(claim: str, certifications: list[str], product_name: str = "") -> dict:
    """Run claim through five-gate pipeline. Returns structured result dict.

    Gate order depends on whether certifications are present:
      With certs:  Gate 3 → Gate 1 → Gate 2 → Gate 4 → Gate 5
      Without:     Gate 1 → Gate 2 → Gate 3 → Gate 4 → Gate 5

    If the product holds certifications, check whether any cert covers the
    claim (Gate 3) before checking the blacklist (Gate 1). This ensures
    partial substantiation (generic term + cert = MEDIUM) is assessed first.
    """
    log.debug("═══ analyze_claim START — claim=%r, certs=%s", claim, certifications)

    # When certifications present, run Gate 3 first — cert coverage
    # takes priority over blacklist.
    if certifications:
        result = gate_certified_specific(claim, certifications)
        if result is not None:
            log.debug("═══ RESOLVED at Gate 3 CERTIFIED_SPECIFIC — risk=%s", result["overall_risk"])
            result["claim"] = claim
            result["product_name"] = product_name
            return result

    result = gate_blacklist(claim, certifications)
    if result is not None:
        log.debug("═══ RESOLVED at Gate 1 BLACKLIST — risk=%s", result["overall_risk"])
        result["claim"] = claim
        result["product_name"] = product_name
        return result

    result = gate_neutrality(claim)
    if result is not None:
        log.debug("═══ RESOLVED at Gate 2 NEUTRALITY — risk=%s", result["overall_risk"])
        result["claim"] = claim
        result["product_name"] = product_name
        return result

    # Gate 3 again only if we skipped it above (no certs)
    if not certifications:
        result = gate_certified_specific(claim, certifications)
        if result is not None:
            log.debug("═══ RESOLVED at Gate 3 CERTIFIED_SPECIFIC — risk=%s", result["overall_risk"])
            result["claim"] = claim
            result["product_name"] = product_name
            return result

    result = gate_comparative(claim, certifications)
    if result is not None:
        log.debug("═══ RESOLVED at Gate 4 COMPARATIVE — risk=%s", result["overall_risk"])
        result["claim"] = claim
        result["product_name"] = product_name
        return result

    log.debug("═══ Gates 1-4 did not resolve — falling through to Gate 5 LLM")
    result = gate_llm(claim, certifications, product_name)
    log.debug("═══ RESOLVED at Gate 5 LLM — risk=%s", result["overall_risk"])

    result["claim"] = claim
    result["product_name"] = product_name
    return result


# ═══════════════════════════════════════════════════════════════════════
# Chat Interface (backward-compatible with main.py)
# ═══════════════════════════════════════════════════════════════════════

CHAT_SYSTEM_PROMPT = """You are an EU regulatory compliance advisor for fashion and textiles.

The system has analyzed all product claims through a deterministic rules engine.
The results are provided below as PRE-COMPUTED COMPLIANCE FINDINGS.

For claims resolved at Gates 1-3 (BLACKLISTED, NEUTRALITY, CERTIFIED_SPECIFIC),
the assessment is deterministic and definitive. Present the risk level, regulation,
and fix exactly as stated.

For claims resolved at Gates 4-5 (COMPARATIVE, AMBIGUOUS), the assessment includes
LLM reasoning. You may add context from the regulation text to elaborate.

Rules:
- Always cite ECGT Directive (EU) 2024/825 as the primary regulation for environmental
  claims — it amends and supersedes UCPD Directive 2005/29/EC from September 2026.
- Present each finding separately — never combine multiple issues into one recommendation.
- Be practical: give actionable recommendations fashion brands can implement.
- Use markdown formatting: bold for risk levels, bullet lists for findings.
- When analyzing multiple products, compare and prioritize by risk level.
- Maintain conversational context — refer back to previous questions when relevant.
"""


_chat_cache: dict[tuple, dict] = {}


def build_chat_context(user_message: str, products: list[dict]) -> str:
    """Run rules engine on all products, format results + RAG text as context."""
    sections = []
    all_results = []

    if products:
        for p in products:
            name = p.get("name") or p.get("product_name") or "Unknown"
            claims_text = p.get("claims") or p.get("marketing_claims") or ""
            certs_text = p.get("certifications") or ""
            claims = _parse_claims(claims_text)
            certs = _parse_certifications(certs_text)

            for claim_str in claims:
                key = (claim_str, tuple(certs), name)
                if key in _chat_cache:
                    result = _chat_cache[key]
                else:
                    result = analyze_claim(claim_str, certs, name)
                    _chat_cache[key] = result
                all_results.append(result)
    else:
        key = (user_message, (), "")
        if key in _chat_cache:
            result = _chat_cache[key]
        else:
            result = analyze_claim(user_message, [], "")
            _chat_cache[key] = result
        all_results.append(result)

    # Format gate results
    if all_results:
        lines = ["=== PRE-COMPUTED COMPLIANCE FINDINGS ==="]
        for r in all_results:
            prefix = f"[{r['product_name']}] " if r.get("product_name") else ""
            lines.append(f"\n  {prefix}Claim: \"{r['claim']}\"")
            lines.append(
                f"    Classification: {r['claim_type']} | "
                f"Risk: {r['overall_risk']} | Gate: {r['gate_resolved_at']}"
            )
            for f in r.get("findings", []):
                lines.append(
                    f"    [{f['risk']}] {f['regulation']} — {f['article']}"
                )
                if f.get("issue"):
                    lines.append(f"           {f['issue']}")
                if f.get("fix"):
                    lines.append(f"           Fix: {f['fix']}")
        sections.append("\n".join(lines))

    # RAG retrieval for regulation text
    query_parts = [user_message]
    for r in all_results:
        query_parts.append(r["claim"])
    for p in products[:10]:
        name = p.get("name") or p.get("product_name") or ""
        if name:
            query_parts.append(name)

    query = " | ".join(query_parts[:15])
    try:
        chunks = retrieve_relevant_chunks(query)
        reg_ctx = _format_regulation_context(chunks)
        sections.append(f"=== REGULATION TEXT ===\n{reg_ctx}")
    except Exception:
        pass

    # Product catalog
    if products:
        cat = ["=== PRODUCT CATALOG ==="]
        for i, p in enumerate(products, 1):
            name = p.get("name") or p.get("product_name") or f"Product {i}"
            claims = p.get("claims") or "No claims"
            materials = p.get("materials") or p.get("material_composition") or ""
            certs = p.get("certifications") or ""
            line = f"  {i}. {name} — Claims: {claims}"
            if materials:
                line += f" | Materials: {materials}"
            if certs:
                line += f" | Certifications: {certs}"
            cat.append(line)
        sections.append("\n".join(cat))

    return "\n\n" + "\n\n".join(sections)
