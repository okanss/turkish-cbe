"""Literal lexical catalog derived from the frozen Turkish/English renderers.

The examples are controlled renderer strings, not newly generated paraphrases.
Morphological checks do not replace the pending human linguistic assessment.
"""

from renderers import english, turkish
from renderers.morphology import ablative, genitive, possessive_locative


# Closed lexical mappings for the literal P4 declarations, not a general
# suffix stripper or a guessed corpus lemma. Every key must match the actual
# renderer declaration before its root is exposed in the catalog.
_ATTRIBUTE_ROOTS = {
    "erkendir": "erken", "geçtir": "geç", "uzundur": "uzun", "kısadır": "kısa",
    "yüksektir": "yüksek", "düşüktür": "düşük", "fazladır": "fazla", "azdır": "az",
    "üsttedir": "üst", "alttadır": "alt",
}
_SPATIAL_SEGMENTS = {
    "ön": ["ön", "ü", "n", "de", "dir"],
    "arka": ["arka", "sı", "n", "da", "dır"],
    "sağ": ["sağ", "ı", "n", "da", "dır"],
    "sol": ["sol", "u", "n", "da", "dır"],
    "üst": ["üst", "ü", "n", "de", "dir"],
    "alt": ["alt", "ı", "n", "da", "dır"],
}


def build_catalog() -> list[dict]:
    """Return fresh dictionaries for nine relation families and two directions.

    ``term`` is the complete canonical Turkish predicate, whereas ``lemma`` is
    the relation's dictionary root.  ``surface_forms`` comprises exact canonical,
    query and P4 attribute predicate spans.  Their frequencies must stay separate
    from the syntactic-word lemma frequency.  English expressions use the actual
    English renderer and the same proper names as the Turkish expressions.
    """
    a, b = "Ayşe", "Bora"
    suffix_names = ("Ayşe", "Bora", "Ahmet", "İlker", "Işıl", "Gökçe", "Özgür")
    output = []
    for relation_id, spec in turkish.RELATIONS.items():
        directions = []
        en = english.RELATIONS[relation_id]
        for direction in ("before", "after"):
            lemma = spec.forward if direction == "before" else spec.reverse
            subject, compared, predicate = turkish._parts(a, b, spec, direction)
            prefix = f"{subject}, {compared} "
            expression = prefix + predicate + "."
            question = turkish._question(a, b, spec, direction)
            if not question.startswith(prefix) or not question.endswith("?"):
                raise ValueError("frozen query no longer matches its declared predicate frame")
            query_predicate = question[len(prefix):-1]
            attribute = turkish._attribute_statement(a, b, spec, direction)
            if spec.family == "spatial":
                attribute_predicate = possessive_locative(lemma)
                attribute_lemma = lemma
            else:
                if " daha " not in attribute or not attribute.endswith("."):
                    raise ValueError("frozen attribute statement has an unregistered frame")
                attribute_predicate = "daha " + attribute.rsplit(" daha ", 1)[1][:-1]
                attribute_decl = attribute_predicate.rsplit(" ", 1)[1]
                if attribute_decl not in _ATTRIBUTE_ROOTS:
                    raise ValueError("unregistered P4 attribute declaration/root mapping")
                attribute_lemma = _ATTRIBUTE_ROOTS[attribute_decl]
            surface_forms = list(dict.fromkeys((predicate, query_predicate, attribute_predicate)))
            morphology = {
                "canonical_predicate": predicate,
                "query_predicate": query_predicate,
                "attribute_predicate": attribute_predicate,
                "query_lemma_tokens": [lemma],
                "attribute_lemma_tokens": [attribute_lemma],
                "attribute_lexeme_changed": attribute_lemma != lemma,
                "canonical_lemma_scope": "question relation and canonical premise lexeme; excludes P4 alternate lexemes",
                "query_expression": question,
                "attribute_expression": attribute,
                "attribute_variant": "P4; morphology, syntax and sometimes lexical changes jointly",
                "proper_name_suffixes": {name: {"genitive": genitive(name), "ablative": ablative(name)}
                                         for name in suffix_names},
                "turkish_characters": sorted(set("çğıöşü") & set(
                    " ".join((lemma, *surface_forms, expression, question, attribute)))),
                "validation_scope": "frozen renderer helpers; human linguistic review pending",
                "sound_rules": {
                    "vowel_harmony": "Four-way high-vowel harmony in -(n)In; two-way harmony in -DAn; frozen positional/copula forms",
                    "consonant_harmony": "Ablative d→t after f,s,t,k,ç,ş,h,p; exact name spelling is preserved",
                    "buffers": "Genitive -n- after vowel-final names; positional possessive/buffer sequences listed literally",
                    "apostrophe": "ASCII apostrophe separates suffixes of proper names; common relation nouns have none",
                    "scope": "Declared names and frozen positional/attribute forms, not a general morphological analyser",
                },
            }
            morphology["proper_name_suffix_inventory"] = sorted({
                form.split("'", 1)[1]
                for forms in morphology["proper_name_suffixes"].values() for form in forms.values()})
            if spec.family == "spatial":
                morphology["possessive_locative_without_copula"] = possessive_locative(lemma, copula=False)
                segments = _SPATIAL_SEGMENTS[lemma]
                if "".join(segments) != predicate:
                    raise ValueError("positional suffix segmentation no longer matches renderer")
                morphology["canonical_morphemes"] = list(segments)
                morphology["canonical_morpheme_functions"] = ["root", "third_person_possessive", "buffer_n", "locative", "assertive_copula"]
            directions.append({
                "direction": direction, "lemma": lemma, "lemma_tokens": [lemma],
                "term": predicate, "surface_forms": surface_forms,
                "expression": expression,
                "english_expression": english.statement(a, b, relation_id, direction),
                "english_question": english.question(a, b, relation_id, direction),
                "english_term": en.forward if direction == "before" else en.reverse,
                "morphology": morphology,
            })
        output.append({"relation_id": relation_id, "family": spec.family,
                       "directions": directions, "human_review_status": "pending",
                       "source": "frozen renderers.turkish and renderers.english"})
    return output
