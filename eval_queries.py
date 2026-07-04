# 250 synthetic queries, 50 per tier (A named, B symptom-only, C multi-disorder,
# D comorbidity, E out-of-distribution). Reproducible; label is an index set, or
# None for OOD. Multi-disorder labels are subjective, so C/D use recall@k.
import random
from itertools import combinations

D, A, B, O, S = 0, 1, 2, 3, 4
CANON = {D: "depression", A: "anxiety", B: "bipolar",
         O: "ocd", S: "schizophrenia"}

NAMES = {
    D: ["depression", "major depression", "clinical depression"],
    A: ["anxiety", "an anxiety disorder", "generalized anxiety disorder"],
    B: ["bipolar disorder", "bipolar", "manic depression"],
    O: ["ocd", "obsessive compulsive disorder", "obsessive-compulsive disorder"],
    S: ["schizophrenia", "schizophrenic disorder"],
}
A_TEMPLATES = [
    "what is {n}?", "what does {n} mean?", "can you explain {n}?",
    "tell me about {n}", "what are the symptoms of {n}?", "how is {n} treated?",
    "what causes {n}?", "how is {n} diagnosed?", "what are the signs of {n}?",
    "is {n} curable?", "what medication is used for {n}?", "how common is {n}?",
    "describe {n}", "give an overview of {n}",
]
B_SYMPTOMS = {
    D: ["I feel worthless and can't get out of bed",
        "nothing interests me anymore and I'm always tired",
        "I feel hopeless and cry most days",
        "I've lost my appetite and can't sleep, everything feels grey",
        "I feel empty and think everyone would be better off without me",
        "I can't concentrate and feel guilty about everything",
        "I have no energy and have withdrawn from friends",
        "life feels pointless and I'm exhausted all the time",
        "I feel sad and slowed down, like I'm underwater",
        "I don't enjoy anything and feel like a failure"],
    A: ["my heart races and I feel like something terrible will happen",
        "I can't stop worrying about everything",
        "I feel tense, restless, and on edge constantly",
        "crowded places make me panic and short of breath",
        "I overthink every conversation for hours afterward",
        "I get sweaty and shaky before speaking in public",
        "a knot of dread sits in my stomach all day",
        "I avoid situations because I fear the worst",
        "my mind won't switch off and I feel jittery",
        "sudden waves of fear hit me for no reason"],
    B: ["for weeks I feel unstoppable then crash into despair",
        "my mood swings between euphoria and deep lows",
        "some nights I don't sleep and start a dozen projects",
        "I go from grandiose plans to hopelessness within weeks",
        "my energy cycles between frantic highs and empty lows",
        "during my highs I spend recklessly and talk nonstop",
        "I feel invincible for days then can't get out of bed",
        "my moods flip dramatically between extremes",
        "racing thoughts and no need for sleep, then a crash",
        "periods of intense productivity followed by total shutdown"],
    O: ["I wash my hands until they're raw to feel clean",
        "I check the stove over and over before leaving",
        "intrusive thoughts force me to repeat rituals",
        "I need things symmetrical or I feel unbearable dread",
        "I count and tap objects a certain number of times",
        "I re-read emails dozens of times fearing a mistake",
        "unwanted violent images make me perform routines",
        "I can't leave until I've checked the locks many times",
        "I repeat phrases silently to prevent bad things happening",
        "contamination fears make me avoid touching things"],
    S: ["I hear voices commenting on what I do",
        "I believe the government is monitoring my thoughts",
        "I see things that others say aren't there",
        "my thoughts feel inserted or controlled by others",
        "I feel people on TV are sending me secret messages",
        "I've withdrawn and my speech feels disorganized",
        "I'm convinced strangers are plotting against me",
        "reality feels distorted and I hear whispers",
        "I believe I have powers others can't perceive",
        "I feel detached, paranoid, and hear noises"],
}
C_TEMPLATES = [
    "difference between {x} and {y}", "can you have {x} and {y} at the same time?",
    "{x} versus {y}", "compare {x} and {y}", "is {x} related to {y}?",
]
D_CONNECTORS = [", and also ", "; on top of that, ", ", plus ",
                " and at the same time ", "; meanwhile "]

_C = ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "Norway", "Kenya"]
_DISH = ["lasagna", "fried rice", "banana bread", "a mushroom risotto", "tacos", "pancakes"]
_TECH = ["a jet engine", "GPS navigation", "blockchain", "photosynthesis",
         "a suspension bridge", "solar panels"]
_MOV = ["action", "comedy", "science fiction", "documentary"]
_FIX = ["a leaky faucet", "a flat bicycle tire", "a slow laptop", "a squeaky door hinge"]
_SPORT = [("the World Cup", "2018"), ("the NBA finals", "2016"),
          ("Wimbledon", "2019"), ("the Super Bowl", "2020")]
_CODE = ["reverse a string", "sort a list of numbers", "check if a number is prime",
         "read a CSV file"]
_MISC = [
    "what's a good beginner yoga routine?", "how do I start investing in index funds?",
    "what are the rules of chess?", "recommend a hiking trail near mountains",
    "how do I change the engine oil in a car?", "what's the best way to learn guitar?",
    "how long does it take to fly to Tokyo?", "explain the offside rule in soccer",
    "what should I pack for a beach vacation?", "how do I grow tomatoes at home?",
    "what year did the Berlin Wall fall?", "how do I make cold brew coffee?",
    "what's a good gift for a 10-year-old?", "convert 100 miles to kilometers",
]


def _build(seed=0):
    rng = random.Random(seed)
    out = []

    # A: named single-disorder
    for c, names in NAMES.items():
        combos = [t.format(n=n) for t in A_TEMPLATES for n in names]
        for q in rng.sample(combos, 10):
            out.append((q, {c}, "A"))

    # B: symptom-only single-disorder
    for c, syms in B_SYMPTOMS.items():
        for q in syms:
            out.append((q, {c}, "B"))

    # C: explicit multi-disorder (10 pairs x 5 templates = 50)
    for x, y in combinations(range(5), 2):
        for t in C_TEMPLATES:
            out.append((t.format(x=CANON[x], y=CANON[y]), {x, y}, "C"))

    # D: complex comorbidity (combine two disorders' symptoms)
    pairs = list(combinations(range(5), 2))
    for i in range(50):
        x, y = pairs[i % len(pairs)]
        sx, sy = rng.choice(B_SYMPTOMS[x]), rng.choice(B_SYMPTOMS[y])
        out.append((sx + rng.choice(D_CONNECTORS) + sy, {x, y}, "D"))

    # E: out-of-distribution (50 varied)
    e = ([f"what is the capital of {c}?" for c in _C]
         + [f"how do I cook {d}?" for d in _DISH]
         + [f"explain how {t} works" for t in _TECH]
         + [f"recommend a good {m} movie" for m in _MOV]
         + [f"how do I fix {h}?" for h in _FIX]
         + [f"who won {s} in {y}?" for s, y in _SPORT]
         + [f"write a python function to {t}" for t in _CODE]
         + _MISC)
    for q in e[:50]:
        out.append((q, None, "E"))

    return out


QUERIES = _build()
