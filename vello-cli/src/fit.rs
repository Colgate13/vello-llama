//! Auto-pick of quantization + tier calculation.
//!
//! Strategy: prefer the "balanced" quant (Q5_K_M if it fits in VRAM, else Q4_K_M).
//! Fall back through Q3_K_M / IQ3_M / IQ2_M as a last resort. For MoE, only the
//! active params need to live in VRAM hot path; total weight goes to RAM via
//! llama.cpp's --n-cpu-moe.

use crate::schema::{Architecture, Model, Profile, Tier};

/// Quant order, from highest quality to lowest.
const QUANT_PREFERENCE: &[&str] = &[
    "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "IQ4_XS", "IQ3_M", "IQ2_M",
    "IQ2_XS",
];

/// Which quant strategies the user picked. Today only Balanced is wired in, but
/// the enum is here so we can add Aggressive / Conservative later without
/// schema changes.
#[derive(Debug, Clone, Copy)]
pub enum Strategy {
    /// Q5 if it fits in VRAM, otherwise Q4. Falls through if neither exists.
    Balanced,
}

#[derive(Debug, Clone)]
pub struct Pick {
    pub quant: String,
    pub file: String,
    pub size_gb: f32,
    pub vram_need_gb: f32,
    pub ram_need_gb: f32,
    pub tier: Tier,
}

/// Pick a specific quantization explicitly. Returns None if the quant is not
/// declared in the model's files map. Tier is still calculated honestly (might
/// be D if the user picked something that won't fit).
pub fn pick_explicit(model: &Model, profile: &Profile, quant: &str) -> Option<Pick> {
    let file = model.files.get(quant)?;
    let size = quant_file_size_gb(model.params_total_b, quant);
    let (vram_need, ram_need) = need(model, size);
    Some(Pick {
        quant: quant.to_string(),
        file: file.clone(),
        size_gb: size,
        vram_need_gb: vram_need,
        ram_need_gb: ram_need,
        tier: classify(model, vram_need, ram_need, profile),
    })
}

pub fn pick(model: &Model, profile: &Profile, _strategy: Strategy) -> Option<Pick> {
    let budget_vram = (profile.vram_gb - profile.vram_reserve_gb - kv_cache_gb(profile)).max(0.0);
    let budget_ram = (profile.ram_gb - profile.ram_reserve_gb).max(0.0);

    // Balanced strategy: try Q5_K_M for VRAM-fit; if it doesn't fit, fall back
    // to Q4_K_M; if that also overflows, descend through preference order.
    let preferred = ["Q5_K_M", "Q4_K_M"];
    for q in preferred {
        if let Some(file) = model.files.get(q) {
            let size = quant_file_size_gb(model.params_total_b, q);
            let (vram_need, ram_need) = need(model, size);
            if vram_need <= budget_vram && ram_need <= budget_ram {
                return Some(Pick {
                    quant: q.into(),
                    file: file.clone(),
                    size_gb: size,
                    vram_need_gb: vram_need,
                    ram_need_gb: ram_need,
                    tier: classify(model, vram_need, ram_need, profile),
                });
            }
        }
    }

    // Fall-through, in order of decreasing VRAM strictness. Each pass walks
    // the QUANT_PREFERENCE list (largest → smallest) and returns the first
    // match. The intent: prefer Q4 with mild offload over Q5 with heavy
    // offload for borderline-sized dense models like 14B on 8 GB.
    let thresholds: [Option<f32>; 3] = [Some(1.0), Some(1.5), None];
    for threshold in thresholds {
        for q in QUANT_PREFERENCE {
            if let Some(file) = model.files.get(*q) {
                let size = quant_file_size_gb(model.params_total_b, q);
                let (vram_need, ram_need) = need(model, size);
                let fits = match threshold {
                    Some(t) => vram_need <= budget_vram * t,
                    None => (vram_need + ram_need) <= (budget_vram + budget_ram),
                };
                if fits {
                    return Some(Pick {
                        quant: (*q).into(),
                        file: file.clone(),
                        size_gb: size,
                        vram_need_gb: vram_need,
                        ram_need_gb: ram_need,
                        tier: classify(model, vram_need, ram_need, profile),
                    });
                }
            }
        }
    }
    None
}

/// Even when nothing fits, we still want to display a tier ("D") for the
/// model's default quant. This bypasses the budget check.
pub fn forced_tier(model: &Model, profile: &Profile) -> Tier {
    let q = &model.default_quant;
    let size = quant_file_size_gb(model.params_total_b, q);
    let (vram_need, ram_need) = need(model, size);
    classify(model, vram_need, ram_need, profile)
}

fn need(model: &Model, size_gb: f32) -> (f32, f32) {
    match model.architecture {
        Architecture::Dense => (size_gb, 0.0),
        Architecture::Moe => {
            // For MoE, only active experts need VRAM; rest goes to RAM.
            let active = model.params_active_b.unwrap_or(model.params_total_b);
            let active_size = size_gb * (active / model.params_total_b);
            let rest = (size_gb - active_size).max(0.0);
            // Add a small constant for shared layers / attention.
            (active_size + 0.5, rest)
        }
    }
}

fn classify(model: &Model, vram_need: f32, ram_need: f32, profile: &Profile) -> Tier {
    let budget_vram = (profile.vram_gb - profile.vram_reserve_gb - kv_cache_gb(profile)).max(0.0);
    let budget_ram = (profile.ram_gb - profile.ram_reserve_gb).max(0.0);

    if vram_need + ram_need > budget_vram + budget_ram {
        return Tier::D;
    }

    // Dense path
    if matches!(model.architecture, Architecture::Dense) {
        if vram_need <= budget_vram * 0.95 {
            return Tier::S;
        }
        if vram_need <= budget_vram * 1.05 {
            return Tier::A;
        }
        let ram_share = (vram_need - budget_vram) / (budget_vram + budget_ram);
        if ram_share < 0.4 {
            return Tier::B;
        }
        return Tier::C;
    }

    // MoE: B is the natural home (some VRAM hot path, rest in RAM).
    if vram_need <= budget_vram && ram_need <= budget_ram * 0.5 {
        return Tier::A;
    }
    if vram_need <= budget_vram && ram_need <= budget_ram {
        return Tier::B;
    }
    Tier::C
}

/// Empirical bytes/parameter for common quants. Used to estimate file size
/// when the catalog doesn't list one. Numbers come from llama.cpp release
/// notes and bartowski quantization tables.
fn bytes_per_param(quant: &str) -> f32 {
    match quant {
        "F16" | "FP16" => 2.0,
        "Q8_0" => 1.07,
        "Q6_K" => 0.82,
        "Q5_K_M" => 0.70,
        "Q5_K_S" => 0.68,
        "Q4_K_M" => 0.59,
        "Q4_K_S" => 0.57,
        "IQ4_XS" => 0.54,
        "Q3_K_M" => 0.49,
        "Q3_K_S" => 0.46,
        "IQ3_M" => 0.45,
        "IQ2_M" => 0.36,
        "IQ2_XS" => 0.33,
        _ => 0.6, // sane default
    }
}

pub fn quant_file_size_gb(params_b: f32, quant: &str) -> f32 {
    params_b * bytes_per_param(quant)
}

/// KV cache size estimate (q8_0 K+V), assuming ~0.0000003 GB per token across
/// typical 7B-30B configs. Rough but good enough for ranking.
fn kv_cache_gb(profile: &Profile) -> f32 {
    let ctx = profile.default_ctx as f32;
    (ctx * 3.0e-5).clamp(0.3, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Architecture;
    use std::collections::BTreeMap;

    fn dense_model(id: &str, params_b: f32, quants: &[&str]) -> Model {
        let mut files = BTreeMap::new();
        for q in quants {
            files.insert((*q).into(), format!("{id}-{q}.gguf"));
        }
        Model {
            id: id.into(),
            repo: "test/repo".into(),
            default_quant: quants[0].into(),
            files,
            params_total_b: params_b,
            params_active_b: None,
            architecture: Architecture::Dense,
            modalities: vec!["text".into()],
            tags: vec![],
            experimental: false,
            description: String::new(),
            runtime: Default::default(),
        }
    }

    fn moe_model(id: &str, total_b: f32, active_b: f32) -> Model {
        let mut m = dense_model(id, total_b, &["Q5_K_M", "Q4_K_M"]);
        m.architecture = Architecture::Moe;
        m.params_active_b = Some(active_b);
        m
    }

    fn rtx_4060_profile() -> Profile {
        Profile {
            vram_gb: 8.0,
            ram_gb: 31.0,
            ..Profile::default()
        }
    }

    #[test]
    fn pick_explicit_returns_none_for_unknown_quant() {
        let m = dense_model("x", 7.0, &["Q5_K_M"]);
        let p = rtx_4060_profile();
        assert!(pick_explicit(&m, &p, "Q9_K_NOPE").is_none());
    }

    #[test]
    fn pick_explicit_returns_some_for_known_quant() {
        let m = dense_model("x", 7.0, &["Q5_K_M", "Q4_K_M"]);
        let p = rtx_4060_profile();
        let pick = pick_explicit(&m, &p, "Q4_K_M").unwrap();
        assert_eq!(pick.quant, "Q4_K_M");
        assert_eq!(pick.file, "x-Q4_K_M.gguf");
    }

    #[test]
    fn dense_7b_q5_fits_vram_tier_s() {
        let m = dense_model("qwen", 7.6, &["Q5_K_M", "Q4_K_M"]);
        let p = rtx_4060_profile();
        let pick = pick(&m, &p, Strategy::Balanced).unwrap();
        assert_eq!(pick.quant, "Q5_K_M");
        assert!(matches!(pick.tier, Tier::S | Tier::A));
    }

    #[test]
    fn dense_14b_q5_overflows_falls_back_to_q4() {
        let m = dense_model("big", 14.0, &["Q5_K_M", "Q4_K_M"]);
        let p = rtx_4060_profile();
        let pick = pick(&m, &p, Strategy::Balanced).unwrap();
        assert_eq!(pick.quant, "Q4_K_M");
    }

    #[test]
    fn moe_30b_a3b_lands_in_tier_b() {
        let m = moe_model("qwen3-30b-a3b", 30.0, 3.0);
        let p = rtx_4060_profile();
        let pick = pick(&m, &p, Strategy::Balanced).unwrap();
        assert!(matches!(pick.tier, Tier::A | Tier::B));
        assert!(pick.ram_need_gb > pick.vram_need_gb);
    }

    #[test]
    fn dense_70b_at_8gb_returns_tier_d_or_c() {
        let m = dense_model("big", 70.0, &["Q4_K_M", "Q3_K_M", "IQ2_M"]);
        let p = rtx_4060_profile();
        // forced_tier never returns None.
        let tier = forced_tier(&m, &p);
        assert!(matches!(tier, Tier::C | Tier::D));
    }

    #[test]
    fn bytes_per_param_q5_smaller_than_q8() {
        assert!(bytes_per_param("Q5_K_M") < bytes_per_param("Q8_0"));
        assert!(bytes_per_param("Q4_K_M") < bytes_per_param("Q5_K_M"));
        assert!(bytes_per_param("IQ2_XS") < bytes_per_param("Q3_K_M"));
    }

    #[test]
    fn quant_file_size_scales_linearly_with_params() {
        let small = quant_file_size_gb(7.0, "Q4_K_M");
        let big = quant_file_size_gb(14.0, "Q4_K_M");
        assert!((big - 2.0 * small).abs() < 0.01);
    }
}
