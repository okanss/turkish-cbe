"""Nine explicitly ordered Turkish relation vocabularies and isolated factors.

Closed templates preserve truth conditions; they are not a general paraphraser.
Returned prompts are checked by a separate surface-to-logic parser and oracle.
"""

from dataclasses import dataclass
import hashlib
import random
import re

from data_generation.generator import canonical_bytes
from data_generation.validation import validate_record

from . import RENDERER_VERSION
from .morphology import ablative, genitive, possessive_locative

VARIANTS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")


@dataclass(frozen=True)
class RelationSpec:
    key: str
    family: str
    forward: str
    reverse: str
    context: str


RELATIONS = {
    "spatial_front": RelationSpec("spatial_front", "spatial", "ön", "arka",
        "Kişiler, yönü sabit tek bir doğrusal eksen üzerindedir. Ön ve arka bu ortak "
        "eksene göredir; kişilerin baktığı yöne göre değişmez."),
    "spatial_right": RelationSpec("spatial_right", "spatial", "sağ", "sol",
        "Kişiler tek bir yatay doğru üzerindedir. Sağ ve sol, bütün kişiler için "
        "aynı sabit gözlem yönüne göre tanımlanmıştır."),
    "spatial_above": RelationSpec("spatial_above", "spatial", "üst", "alt",
        "Kişilerin konumları tek bir düşey eksen üzerinde karşılaştırılır. Üstünde "
        "yalnız daha yukarıda, altında yalnız daha aşağıda demektir; fiziksel temas "
        "veya bir nesnenin üzerinde durma anlamı taşımaz."),
    "temporal_arrival": RelationSpec("temporal_arrival", "temporal", "önce", "sonra",
        "Bütün kişiler aynı etkinliğe tam bir kez gelmiştir. Yalnız geliş zamanları "
        "karşılaştırılır; her kişinin geliş zamanı farklıdır."),
    "temporal_completion": RelationSpec("temporal_completion", "temporal", "önce", "sonra",
        "Varlık etiketleri birer görevin adıdır; kişi adı biçimindeki etiketler de "
        "görev adı olarak kullanılır. Bütün görevler tam bir kez tamamlanmıştır. "
        "Yalnız tamamlanma zamanları karşılaştırılır."),
    "comparison_height": RelationSpec("comparison_height", "comparison", "uzun", "kısa",
        "Kişilerin aynı birimle ölçülen boyları karşılaştırılır. Uzun ve kısa "
        "yalnız boy farkını belirtir; bütün boylar farklıdır."),
    "comparison_speed": RelationSpec("comparison_speed", "comparison", "hızlı", "yavaş",
        "Kişilerin aynı koşullarda ölçülen sabit hızları karşılaştırılır. Hızlı "
        "daha yüksek hız, yavaş daha düşük hız demektir."),
    "comparison_weight": RelationSpec("comparison_weight", "comparison", "ağır", "hafif",
        "Kişilerin aynı birimle ölçülen ağırlıkları karşılaştırılır. Ağır daha "
        "fazla ağırlık, hafif daha az ağırlık demektir."),
    "ranking": RelationSpec("ranking", "ranking", "üst", "alt",
        "Kişiler tek bir başarı sıralamasındadır. Üst sıra daha iyi, alt sıra "
        "daha kötü sıradır; paylaşılan bir sıra yoktur."),
}

_ADJECTIVE_DECL = {"uzun": "uzundur", "kısa": "kısadır", "hızlı": "hızlıdır",
                   "yavaş": "yavaştır", "ağır": "ağırdır", "hafif": "hafiftir"}
_ADJECTIVE_QUESTION = {"uzun": "uzun mudur", "kısa": "kısa mıdır",
                       "hızlı": "hızlı mıdır", "yavaş": "yavaş mıdır",
                       "ağır": "ağır mıdır", "hafif": "hafif midir"}
_ATTRIBUTES = {
    "comparison_height": ("boyu", "boyundan", "uzundur", "kısadır"),
    "comparison_speed": ("hızı", "hızından", "yüksektir", "düşüktür"),
    "comparison_weight": ("ağırlığı", "ağırlığından", "fazladır", "azdır"),
    "ranking": ("sırası", "sırasından", "üsttedir", "alttadır"),
}
_TRANSFORMS = {
    "P0": (),
    "P1": ("premise_order", "entity_order"),
    "P2": ("argument_swap", "converse_lexeme"),
    "P3": ("word_order",),
    "P4": ("possessive_nominalization", "case_change", "attribute_lexeme"),
    "P5": ("premise_order", "entity_order", "argument_swap", "converse_lexeme",
           "possessive_nominalization", "case_change", "attribute_lexeme"),
    "P6": ("controlled_negation", "opposite_lexeme"),
}
_AXES = {"P0": (), "P1": ("list_order",), "P2": ("expression_direction",),
         "P3": ("syntax",), "P4": ("morphology", "syntax", "lexical"),
         "P5": ("list_order", "expression_direction", "morphology", "syntax", "lexical"),
         "P6": ("negation", "lexical")}


def _digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def expected_context(relation_id: str, entity_names) -> str:
    if not isinstance(relation_id, str) or relation_id not in RELATIONS:
        raise ValueError("unknown relation_id")
    context = (RELATIONS[relation_id].context + " Bu problemde belirtilen ilişki katı bir tam "
               "sıralamadır: bütün değerler farklıdır, eşitlik ve karşılaştırılamazlık yoktur. "
               "Her öncül aynı sıralama eksenini kullanır.")
    if any(re.fullmatch(r"E[0-9]{4}", name) for name in entity_names):
        context += (" E ile başlayan alfanümerik adlar, E harfi ve rakamları tek tek "
                    "Türkçe okunarak adlandırılır; ekler son rakamın okunuşuna göre yazılır.")
    return context


def compose_prompt(context: str, entity_names, premises, query: str) -> str:
    return (context + "\nVarlıklar: " + ", ".join(entity_names) + ".\nÖncüller:\n"
            + "\n".join("- " + text for text in premises) + "\nSoru: " + query
            + "\nYanıtınızı yalnız YES veya NO olarak verin.")


def _parts(a: str, b: str, spec: RelationSpec, direction: str):
    word = spec.forward if direction == "before" else spec.reverse
    if spec.family == "spatial":
        return a, genitive(b), possessive_locative(word)
    if spec.key == "temporal_arrival":
        return a, ablative(b), f"{word} gelmiştir"
    if spec.key == "temporal_completion":
        return f"{a} adlı görev", f"{b} adlı görevden", f"{word} tamamlanmıştır"
    if spec.family == "comparison":
        return a, ablative(b), "daha " + _ADJECTIVE_DECL[word]
    return a, ablative(b), f"daha {word} sıradadır"


def _attribute_statement(a: str, b: str, spec: RelationSpec, direction: str) -> str:
    before = direction == "before"
    word = spec.forward if before else spec.reverse
    if spec.family == "spatial":
        return f"{genitive(a)} konumu, {genitive(b)} konumunun {possessive_locative(word)}."
    if spec.key == "temporal_arrival":
        pred = "erkendir" if before else "geçtir"
        return f"{genitive(a)} geliş zamanı, {genitive(b)} geliş zamanından daha {pred}."
    if spec.key == "temporal_completion":
        pred = "erkendir" if before else "geçtir"
        return (f"{a} adlı görevin tamamlanma zamanı, {b} adlı görevin tamamlanma "
                f"zamanından daha {pred}.")
    head, compared, forward_pred, reverse_pred = _ATTRIBUTES[spec.key]
    return f"{genitive(a)} {head}, {genitive(b)} {compared} daha {forward_pred if before else reverse_pred}."


def _negative_statement(a: str, b: str, spec: RelationSpec) -> str:
    # not inverse(a,b) iff forward(a,b), given distinct totally ordered values.
    if spec.family == "spatial":
        return f"{a}, {genitive(b)} {possessive_locative(spec.reverse, copula=False)} değildir."
    if spec.key == "temporal_arrival":
        return f"{a}, {ablative(b)} {spec.reverse} gelmemiştir."
    if spec.key == "temporal_completion":
        return f"{a} adlı görev, {b} adlı görevden {spec.reverse} tamamlanmamıştır."
    if spec.family == "comparison":
        return f"{a}, {ablative(b)} daha {spec.reverse} değildir."
    return f"{a}, {ablative(b)} daha {spec.reverse} sırada değildir."


def _question(a: str, b: str, spec: RelationSpec, direction: str) -> str:
    word = spec.forward if direction == "before" else spec.reverse
    if spec.family == "spatial":
        locative = possessive_locative(word, copula=False)
        particle = "midir" if locative.endswith("e") else "mıdır"
        return f"{a}, {genitive(b)} {locative} {particle}?"
    if spec.key == "temporal_arrival":
        return f"{a}, {ablative(b)} {word} gelmiş midir?"
    if spec.key == "temporal_completion":
        return f"{a} adlı görev, {b} adlı görevden {word} tamamlanmış mıdır?"
    if spec.family == "comparison":
        return f"{a}, {ablative(b)} daha {_ADJECTIVE_QUESTION[word]}?"
    return f"{a}, {ablative(b)} daha {word} sırada mıdır?"


def _permuted(values: list, rng: random.Random) -> list:
    output = list(values)
    rng.shuffle(output)
    if len(values) > 1 and output == values:
        output = output[1:] + output[:1]
    return output


def render(record: dict, relation_id: str = "spatial_front", variant: str = "P0",
           seed: int = 31, allow_negation: bool = False) -> dict:
    validate_record(record)
    if not isinstance(relation_id, str) or relation_id not in RELATIONS:
        raise ValueError("unknown relation_id")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("unknown variant")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("renderer seed must be an integer in [0, 2**64)")
    if type(allow_negation) is not bool:
        raise ValueError("allow_negation must be Boolean")
    if variant == "P6" and not allow_negation:
        raise ValueError("P6 requires explicit allow_negation=True")
    spec = RELATIONS[relation_id]
    names = sorted(record["latent_order"])
    names = _permuted(names, random.Random(int(_digest([RENDERER_VERSION, seed, record["instance_id"], "names"]), 16)))
    premises = []
    for original_a, original_b in record["premise_edges"]:
        a, b, direction = original_a, original_b, "before"
        if variant in ("P2", "P5"):
            a, b, direction = b, a, "after"
        if variant in ("P4", "P5"):
            text = _attribute_statement(a, b, spec, direction)
        elif variant == "P6":
            text = _negative_statement(a, b, spec)
        else:
            subject, compared, predicate = _parts(a, b, spec, direction)
            text = (f"{compared} {predicate} {subject}." if variant == "P3"
                    else f"{subject}, {compared} {predicate}.")
        premises.append(text)
    if variant in ("P1", "P5"):
        names = _permuted(names, random.Random(int(_digest([RENDERER_VERSION, seed, record["instance_id"], "P1_names"]), 16)))
        premises = _permuted(premises, random.Random(int(_digest([RENDERER_VERSION, seed, record["instance_id"], "P1_premises"]), 16)))
    context = expected_context(relation_id, names)
    query = _question(record["query_subject"], record["query_object"], spec, record["query_relation"])
    prompt = compose_prompt(context, names, premises, query)
    presentation = {
        "renderer_version": RENDERER_VERSION, "generator_version": record["generator_version"],
        "instance_id": record["instance_id"], "ordering_id": record["ordering_id"],
        "source_record_sha256": _digest(record), "seed": seed, "language": "tr",
        "relation_id": relation_id, "family": spec.family, "variant": variant,
        "strict_total_order": True, "entity_names": names, "premises": premises,
        "query": query, "context": context, "prompt": prompt,
        "transformations": list(_TRANSFORMS[variant]), "factor_axes": list(_AXES[variant]),
        "is_multi_transform": len(_AXES[variant]) > 1,
        "human_review_status": "pending", "eligible_for_confirmatory_evaluation": False,
        "negation_proof": ("Distinct total order: not inverse(a,b) iff forward(a,b)."
                           if variant == "P6" else None),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    presentation["presentation_id"] = "tr-" + _digest([RENDERER_VERSION, record["instance_id"],
                                                        relation_id, variant, seed, presentation["prompt_sha256"]])
    from validation.semantic_equivalence import validate_presentation
    presentation["validation"] = validate_presentation(record, presentation)
    return presentation
