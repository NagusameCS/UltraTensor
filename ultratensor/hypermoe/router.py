"""Hyper-MoE domain router.

Classifies a request by PURPOSE (backend/frontend/data/devops/math)
and then by LANGUAGE, so the dispatcher can pick the right specialist
model.  Heuristic v1: weighted regex priors.  The measured upgrade
path is the 181k-param score controller (rho@192 Spearman 0.98) once
specialist traces exist.
"""

import re

# purpose domains first (they override language), then languages
DOMAIN_RULES = [
    ("math", [
        r"\bprove\b", r"\btheorem\b", r"\bintegral\b", r"\bderivative\b",
        r"\bmatrix\b", r"\beigenvalue", r"\bsolve\b", r"\bproof\b",
        r"\bgcd\b", r"\bprime", r"\bprobability of", r"\binduction\b",
        r"\bremainder\b", r"\bequation\b",
    ]),
    ("backend", [
        r"\bapi\b", r"\bendpoint\b", r"\bjwt\b", r"\bpostgres",
        r"\bdatabase\b", r"\bconnection pool\b", r"\bmessage queue\b",
        r"\bpagination\b", r"\bserver\b", r"\bauth", r"\bcursor",
        r"\brate.?limit", r"\bworker\b",
    ]),
    ("frontend", [
        r"\breact\b", r"\bcss\b", r"\bdom\b", r"\bcomponent\b",
        r"\bhtml\b", r"\bdebounce", r"\bdrag and drop\b",
        r"\bform validation\b", r"\bui\b", r"\blayout\b", r"\bgrid\b",
    ]),
    ("data", [
        r"\bpandas\b", r"\bdataframe\b", r"\bcsv\b", r"\bk-means\b",
        r"\bclustering\b", r"\bgradient descent\b",
        r"\blogistic regression\b", r"\bfeature engineering\b",
        r"\bnumpy\b", r"\btrain\b", r"\bvectorize\b",
    ]),
    ("devops", [
        r"\bdockerfile\b", r"\bkubernetes\b", r"\bdeployment\b",
        r"\bhealth check\b", r"\blog.?rotat", r"\bs3\b",
        r"\bbash script\b", r"\bdeploy", r"\brolling updates\b",
    ]),
    ("sql", [
        r"\bselect\b", r"\binsert\b", r"\bupdate\b", r"\bjoin\b",
        r"\bgroup by\b", r"\bwhere\b", r"\bindex\b",
        r"\bwindow function\b", r"\bhaving\b", r"\bcreate table\b",
    ]),
    ("rust", [
        r"\bfn\b", r"\bimpl\b", r"\bmatch\b", r"\btrait\b", r"->",
        r"\bcrate\b", r"\bResult<", r"\brust\b",
    ]),
    ("go", [
        r"\bfunc\b", r"\bchan\b", r"\bgoroutine\b", r":=",
        r"\bgo\b.*\bmodule\b", r"\bgolang\b",
    ]),
    ("cpp", [
        r"#include", r"\btemplate<", r"\bstd::", r"\bnamespace\b",
        r"\bc\+\+",
    ]),
    ("javascript", [
        r"\bfunction\b", r"=>", r"\bconst\b", r"\bawait\b",
        r"\bconsole\.", r"\bjavascript\b", r"\bjs\b", r"\bnode\b",
    ]),
    ("python", [
        r"\bdef\b", r"\bimport\b", r"\bself\.", r"\blambda\b",
        r"\belif\b", r"\b__init__", r"\bpython\b", r"\bdecorator\b",
        r"\bclass\b.*:", r"\bdict\b", r"\bcollections\b",
    ]),
]

_COMPILED = [(name, [re.compile(p, re.IGNORECASE) for p in pats])
             for name, pats in DOMAIN_RULES]

PURPOSE_FIRST = {"math", "backend", "frontend", "data", "devops"}


class DomainRouter:
    """Classify free text into (domain, score) rankings."""

    def rank(self, text: str):
        """-> list of (domain, score) sorted best-first; score>0 only."""
        scores = []
        for name, pats in _COMPILED:
            s = sum(1 for p in pats if p.search(text))
            if re.search(rf"\b{name}\b", text, re.IGNORECASE):
                s += 2                       # explicit domain mention
            if s:
                scores.append((name, s))
        # purpose domains outrank explicit language mentions
        scores.sort(key=lambda t: (t[0] not in PURPOSE_FIRST, -t[1], t[0]))
        return scores

    def classify(self, text: str):
        """-> (domain, score) with 'general' fallback."""
        ranked = self.rank(text)
        if ranked:
            return ranked[0]
        return ("general", 0)
