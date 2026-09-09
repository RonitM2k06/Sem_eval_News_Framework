"""
Synthetic Data Generator for SemEval 2025 Task 10.
Generates realistic multilingual synthetic data (bg, en, hi, pt, ru) for Ukraine-Russia War and Climate Change domains.
Used for offline unit tests, integration tests, and smoke testing when official benchmark data is not present in data/raw/.
"""

import os
import json
import random
from typing import List, Dict, Any
from src.taxonomy.parser import TaxonomyParser, DOMAIN_TAXONOMY


LANGUAGES = ["bg", "en", "hi", "pt", "ru"]
DOMAINS = ["ukraine_russia", "climate_change"]

SAMPLE_TEXTS: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "ukraine_russia": [
            "Official statements from NATO members claimed that security guarantees were required for eastern expansion. However, Russian diplomats argued that Western expansion directly provoked tension. Energy prices in Europe increased sharply after trade sanctions were implemented. International observers called for immediate cease-fire negotiations in Geneva.",
            "Reports from Kyiv highlighted structural damage to military infrastructure during recent strikes. Western media outlets reported on foreign aid packages sent to assist civilian rehabilitation. Independent journalists documented ongoing diplomatic efforts aimed at resolving territorial disputes peacefully."
        ],
        "climate_change": [
            "Recent scientific reports emphasize that renewable energy adoption faces significant grid integration challenges. Skeptics claim global temperature fluctuations are part of natural centuries-long cycles. Industrial representatives warn that aggressive carbon taxation will destabilize manufacturing jobs.",
            "Environmental activists urge governments to phase out fossil fuel subsidies immediately. Economists suggest that green transition investments will create long-term employment in solar and wind sectors."
        ]
    },
    "bg": {
        "ukraine_russia": [
            "Официални изявления от представители на НАТО посочват, че разширяването на съюза е защитна мярка. Руските дипломати обаче твърдят, че това създава директно напрежение в региона. Санкциите засегнаха европейската икономика и енергийните пазари.",
            "Докладите от района на конфликта показват сериозни последици за инфраструктурата. Международни наблюдатели призовават за възобновяване на дипломатическия преговорен процес."
        ],
        "climate_change": [
            "Някои икономисти предупреждават, че бързият преход към зелена енергия може да повиши цените за потребителите. Климатичните учени обаче поддържат необходимостта от намаляване на въглеродните емисии."
        ]
    },
    "hi": {
        "ukraine_russia": [
            "नाटो के बयानों में कहा गया कि पूर्वी विस्तार सुरक्षा के लिए आवश्यक था। दूसरी ओर, रूसी अधिकारियों ने तर्क दिया कि पश्चिमी नीतियों ने तनाव बढ़ाया। यूरोप में ऊर्जा संकट के कारण आर्थिक प्रभाव देखा गया है।",
            "कूटनीतिक प्रयासों के बावजूद शांति वार्ता बाधित रही है। अंतर्राष्ट्रीय समुदाय ने मानवीय सहायता और युद्धविराम की अपील की है।"
        ],
        "climate_change": [
            "जलवायु परिवर्तन पर हालिया रिपोर्टों ने नवीकरणीय ऊर्जा के महत्व पर जोर दिया है। कुछ विशेषज्ञों का तर्क है कि कार्बन कर से उद्योगों पर वित्तीय दबाव बढ़ेगा।"
        ]
    },
    "pt": {
        "ukraine_russia": [
            "Declarações oficiais da OTAN afirmaram que a expansão visava garantir a estabilidade regional. Por outro lado, diplomatas russos sustentam que as decisões ocidentais provocaram a crise. As sanções econômicas geraram impactos diretos nos preços de energia na Europa.",
            "Relatórios de observadores internacionais destacam a urgência de negociações de paz para cessar as hostilidades."
        ],
        "climate_change": [
            "Debates sobre políticas climáticas apontam os desafios econômicos da transição energética. Críticos argumentam que regulamentações excessivas prejudicam a competitividade industrial."
        ]
    },
    "ru": {
        "ukraine_russia": [
            "Официальные представители заявили, что расширение блока НАТО создало угрозы безопасности. Дипломатические источники подчеркивают необходимость гарантий нейтрального статуса. Экономические санкции вызвали рост цен на энергоресурсы в Европе.",
            "Международные наблюдатели призывают к возобновлению мирных переговоров и дипломатическому урегулированию конфликта."
        ],
        "climate_change": [
            "Некоторые эксперты утверждают, что климатические модели не учитывают все природные факторы. В то же время экологи настаивают на ускоренном переходе к возобновляемым источникам энергии."
        ]
    }
}

SAMPLE_ENTITIES = {
    "ukraine_russia": ["NATO", "Russia", "Kyiv", "Western media", "European Union", "UN Observers"],
    "climate_change": ["Green Energy Council", "Fossil Fuel Coalition", "Climate Scientists", "Industrial Union"]
}

SAMPLE_ROLES = ["Protagonist", "Antagonist", "Innocent", "Hero", "Villain", "Victim"]


def generate_synthetic_article(article_id: str, lang: str, domain: str) -> Dict[str, Any]:
    texts = SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS["en"]).get(domain, SAMPLE_TEXTS["en"]["ukraine_russia"])
    text = random.choice(texts)
    
    # Extract sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    # Pick entity mentions
    entities_pool = SAMPLE_ENTITIES[domain]
    entity_name = random.choice(entities_pool)
    
    start_offset = text.find(entity_name)
    if start_offset == -1:
        start_offset = 0
        end_offset = len(entity_name)
    else:
        end_offset = start_offset + len(entity_name)
        
    role = random.choice(SAMPLE_ROLES)
    
    # Pick narrative and subnarrative
    parents = list(DOMAIN_TAXONOMY[domain].keys())
    parent_narrative = random.choice(parents)
    subnarrative = random.choice(DOMAIN_TAXONOMY[domain][parent_narrative])
    
    # Pick supporting evidence sentence (0-indexed)
    evidence_sentence_id = random.randint(0, max(0, len(sentences) - 1))
    
    explanation = f"The article highlights how {entity_name} acts as {role.lower()} within the scope of {parent_narrative.lower()}, specifically referencing '{subnarrative}' based on textual evidence."
    
    return {
        "article_id": article_id,
        "language": lang,
        "domain": domain,
        "text": text,
        "sentences": sentences,
        "entities": [
            {
                "mention": entity_name,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "roles": [role],
                "main_role": role if role in ["Protagonist", "Antagonist", "Innocent"] else "Protagonist"
            }
        ],
        "narrative": parent_narrative,
        "subnarrative": subnarrative,
        "evidence_sentence_ids": [evidence_sentence_id],
        "explanation": explanation
    }


def generate_synthetic_dataset(output_dir: str, samples_per_split: int = 20) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "dev", "test"]
    output_files = {}

    idx = 1
    for split in splits:
        records = []
        num_samples = samples_per_split if split != "test" else int(samples_per_split * 0.5)
        for _ in range(num_samples):
            lang = random.choice(LANGUAGES)
            domain = random.choice(DOMAINS)
            article_id = f"art_{split}_{idx:04d}"
            rec = generate_synthetic_article(article_id, lang, domain)
            records.append(rec)
            idx += 1

        filepath = os.path.join(output_dir, f"synthetic_{split}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        output_files[split] = filepath

    print(f"[SyntheticData] Generated synthetic dataset files in {output_dir}")
    return output_files


if __name__ == "__main__":
    generate_synthetic_dataset("data/synthetic", samples_per_split=30)
