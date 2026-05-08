//! Use-case → tag mapping and ranking.

use crate::catalog::CatalogEntry;
use crate::fit::{pick, Pick, Strategy};
use crate::schema::{Profile, Tier};

pub struct Recommendation<'a> {
    pub entry: &'a CatalogEntry,
    pub pick: Pick,
    pub score: f32,
}

/// Map a free-text use-case query to the set of tags/use_cases that should
/// match a model. Lowercase substring matching keeps it forgiving — pt-BR
/// triggers ("código", "raciocínio", "imagem") work the same as English.
fn use_case_tags(q: &str) -> Vec<&'static str> {
    let q = q.to_lowercase();
    let mut tags = Vec::new();
    let map: &[(&[&str], &[&str])] = &[
        (
            &["code", "código", "codigo", "coder", "coding"],
            &["code", "coder"],
        ),
        (
            &["chat", "geral", "general", "assistant"],
            &["chat", "general"],
        ),
        (
            &["tool", "agent", "agente", "function"],
            &["tools", "agent"],
        ),
        (
            &["reason", "raciocí", "raciocin", "lógic", "logic", "math"],
            &["reasoning", "thinking"],
        ),
        (
            &["vision", "imagem", "image", "visão", "visao", "ocr"],
            &["vision", "multimodal"],
        ),
        (&["video", "vídeo"], &["video"]),
        (
            &[
                "multilíng",
                "multiling",
                "português",
                "portugues",
                "pt-br",
                "pt_br",
            ],
            &["multilingual"],
        ),
        (
            &["small", "pequeno", "leve", "fast", "rápido", "rapido"],
            &["small", "fast"],
        ),
        (&["uncensor"], &["uncensored"]),
    ];
    for (triggers, tag_list) in map {
        if triggers.iter().any(|t| q.contains(t)) {
            tags.extend(*tag_list);
        }
    }
    tags
}

pub fn rank<'a>(
    entries: &'a [CatalogEntry],
    profile: &Profile,
    query: &str,
    limit: usize,
) -> Vec<Recommendation<'a>> {
    let target_tags = use_case_tags(query);
    let mut scored: Vec<Recommendation> = entries
        .iter()
        .filter_map(|e| {
            let p = pick(&e.model, profile, Strategy::Balanced)?;
            // Skip Tier D from recommendations — never useful.
            if matches!(p.tier, Tier::D) {
                return None;
            }
            let matched_count = target_tags
                .iter()
                .filter(|t| e.model.tags.iter().any(|mt| mt == *t))
                .count();
            let mut score = if target_tags.is_empty() {
                // No specific use case → just rank by tier
                p.tier.factor()
            } else if matched_count == 0 {
                // Tag query but no match → exclude
                return None;
            } else {
                let coverage = matched_count as f32 / target_tags.len().max(1) as f32;
                p.tier.factor() * (0.5 + 0.5 * coverage)
            };
            if e.model.experimental {
                score *= 0.85;
            }
            Some(Recommendation {
                entry: e,
                pick: p,
                score,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn use_case_tags_maps_code_in_pt_and_en() {
        let en = use_case_tags("code");
        let pt = use_case_tags("código");
        assert!(en.contains(&"code"));
        assert!(pt.contains(&"code"));
    }

    #[test]
    fn use_case_tags_maps_vision() {
        let tags = use_case_tags("vision");
        assert!(tags.contains(&"vision"));
        assert!(tags.contains(&"multimodal"));
    }

    #[test]
    fn use_case_tags_maps_reasoning() {
        let tags = use_case_tags("raciocínio");
        assert!(tags.contains(&"reasoning"));
    }

    #[test]
    fn use_case_tags_unknown_query_returns_empty() {
        assert!(use_case_tags("zzz-not-a-thing").is_empty());
    }
}
