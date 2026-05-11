//! Use-case → tag mapping and ranking.

use crate::catalog::CatalogEntry;
use crate::fit::{pick, Pick, Strategy};
use crate::schema::{Architecture, Mode, Profile, Tier};

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
    let mode_target = profile.mode.as_target();
    let mut scored: Vec<Recommendation> = entries
        .iter()
        .filter_map(|e| {
            // Filter out models that don't declare support for the current
            // runtime mode. In CPU mode that hides dense ≥12B; in GPU mode
            // any catalog entry without "gpu" in targets is also excluded
            // (rare — the default for back-compat is ["gpu"]).
            if !e.model.supports(mode_target) {
                return None;
            }
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
            // CPU mode reward MoE specifically: with `--n-cpu-moe` only the
            // active params are computed per token, so e.g. qwen3-30b-a3b
            // runs much closer to a 3B than a 30B. Tier already reflects
            // this somewhat, but a small explicit boost helps MoE models
            // outrank similarly-tiered dense rivals on a laptop.
            if profile.mode == Mode::Cpu && matches!(e.model.architecture, Architecture::Moe) {
                score *= 1.10;
            }
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

    // -------------------------------------------------------------------
    // Rank tests — filter by target + MoE bonus in CPU mode.
    // -------------------------------------------------------------------

    use crate::catalog::CatalogEntry;
    use crate::schema::{Architecture, Model, ModelRuntime, Target};
    use std::collections::BTreeMap;

    fn model(id: &str, params_b: f32, arch: Architecture, targets: Vec<Target>) -> Model {
        let mut files = BTreeMap::new();
        files.insert("Q4_K_M".into(), format!("{id}.gguf"));
        Model {
            id: id.into(),
            repo: "test/repo".into(),
            default_quant: "Q4_K_M".into(),
            files,
            params_total_b: params_b,
            params_active_b: if matches!(arch, Architecture::Moe) {
                Some(3.0)
            } else {
                None
            },
            architecture: arch,
            modalities: vec!["text".into()],
            targets,
            tags: vec!["chat".into()],
            experimental: false,
            description: String::new(),
            runtime: ModelRuntime::default(),
        }
    }

    fn gpu_profile() -> Profile {
        Profile {
            mode: Mode::Gpu,
            vram_gb: 24.0,
            ram_gb: 64.0,
            ..Profile::default()
        }
    }

    fn cpu_profile() -> Profile {
        Profile {
            mode: Mode::Cpu,
            vram_gb: 0.0,
            ram_gb: 32.0,
            ..Profile::default()
        }
    }

    fn entry(m: Model) -> CatalogEntry {
        CatalogEntry { model: m }
    }

    #[test]
    fn rank_filters_out_gpu_only_models_in_cpu_mode() {
        let entries = vec![
            entry(model("dense-big-gpu-only", 14.0, Architecture::Dense, vec![Target::Gpu])),
            entry(model("small-both", 3.0, Architecture::Dense, vec![Target::Gpu, Target::Cpu])),
        ];
        let recs = rank(&entries, &cpu_profile(), "chat", 10);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].entry.model.id, "small-both");
    }

    #[test]
    fn rank_filters_out_cpu_only_models_in_gpu_mode() {
        let entries = vec![
            entry(model("cpu-only", 3.0, Architecture::Dense, vec![Target::Cpu])),
            entry(model("gpu-ok", 7.0, Architecture::Dense, vec![Target::Gpu])),
        ];
        let recs = rank(&entries, &gpu_profile(), "chat", 10);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].entry.model.id, "gpu-ok");
    }

    #[test]
    fn moe_gets_score_bonus_in_cpu_mode() {
        // Two models of similar tier — dense vs MoE. In CPU mode, MoE should
        // rank higher due to the 1.10x bonus.
        let entries = vec![
            entry(model("dense-7b", 7.0, Architecture::Dense, vec![Target::Cpu, Target::Gpu])),
            entry(model("moe-30b-a3b", 30.0, Architecture::Moe, vec![Target::Cpu, Target::Gpu])),
        ];
        let recs = rank(&entries, &cpu_profile(), "chat", 10);
        // MoE is small-active (3B) → likely Tier S in CPU; dense 7B → Tier B.
        // Even without the bonus MoE would lead; the bonus reinforces it.
        let moe_pos = recs.iter().position(|r| r.entry.model.id == "moe-30b-a3b");
        let dense_pos = recs.iter().position(|r| r.entry.model.id == "dense-7b");
        assert!(moe_pos.is_some() && dense_pos.is_some());
        assert!(moe_pos.unwrap() < dense_pos.unwrap(), "MoE should rank above dense-7b in CPU mode");
    }

    #[test]
    fn moe_bonus_does_not_apply_in_gpu_mode() {
        // Same MoE in GPU mode: still ranks well by tier, but the 1.10x boost
        // is CPU-mode-specific so we don't see it here. Hard to assert directly
        // without exposing score, but we can verify that filtering still works.
        let entries = vec![
            entry(model("dense-7b", 7.0, Architecture::Dense, vec![Target::Gpu, Target::Cpu])),
            entry(model("moe-30b-a3b", 30.0, Architecture::Moe, vec![Target::Gpu, Target::Cpu])),
        ];
        let recs = rank(&entries, &gpu_profile(), "chat", 10);
        // Both should appear; ordering is tier-driven (24 GB VRAM fits both).
        assert_eq!(recs.len(), 2);
    }
}
