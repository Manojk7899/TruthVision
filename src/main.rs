//! AI Video Detector - Gemini explanation module.
//!
//! Input (from the ML model, as text):
//!     Prediction: FAKE
//!     Confidence: 0.93
//!     Probability: [0.93, 0.07]        <- [fake, real]
//!
//! Optional keywords (future): "lip sync missing", "unnatural blinking", ...
//!
//! Usage:
//!     cat model_output.txt | cargo run -- --keywords "lip sync missing,blurry edges"

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::{env, io::Read};

// ---------- 1. Data model ----------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Label {
    Real,
    Fake,
}

#[derive(Debug)]
struct Detection {
    label: Label,
    confidence: f64,
    prob_fake: f64,
    prob_real: f64,
    keywords: Vec<String>,
}

// ---------- 2. Parse the ML model's text output ----------

fn parse_model_output(raw: &str) -> Result<Detection> {
    let (mut label, mut confidence, mut probs) = (None, None, None);

    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else { continue };
        let value = value.trim();

        match key.trim().to_lowercase().as_str() {
            "prediction" => {
                let v = value.trim_matches(|c| c == '"' || c == '\'').to_uppercase();
                label = Some(match v.as_str() {
                    "REAL" => Label::Real,
                    "FAKE" => Label::Fake,
                    other => bail!("Unknown prediction '{other}', expected REAL or FAKE"),
                });
            }
            "confidence" => {
                confidence = Some(value.parse::<f64>().context("Bad confidence value")?);
            }
            "probability" => {
                let nums = value
                    .trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .context("Bad probability list")?;
                if nums.len() != 2 {
                    bail!("Probability must have 2 values: [fake, real]");
                }
                probs = Some((nums[0], nums[1]));
            }
            _ => {}
        }
    }

    let (prob_fake, prob_real) = probs.ok_or_else(|| anyhow!("Missing 'Probability' line"))?;
    Ok(Detection {
        label: label.ok_or_else(|| anyhow!("Missing 'Prediction' line"))?,
        confidence: confidence.ok_or_else(|| anyhow!("Missing 'Confidence' line"))?,
        prob_fake,
        prob_real,
        keywords: vec![],
    })
}

// ---------- 3. Build the prompt ----------

const SYSTEM_PROMPT: &str = "You are an assistant inside an AI-generated-video detection tool. \
You do NOT see the video. You only receive the output of a separate ML classifier and, \
optionally, keywords describing artifacts it flagged. Explain the result in clear, simple language \
for a non-technical user. Never invent evidence that was not given to you. Be honest about \
uncertainty: if confidence is low or probabilities are close, say the result is not conclusive. \
Do not claim certainty; detectors can be wrong. Keep the answer under 150 words.";

fn build_prompt(d: &Detection) -> String {
    let label = match d.label {
        Label::Fake => "FAKE (AI-generated / manipulated)",
        Label::Real => "REAL (authentic)",
    };

    let mut p = format!(
        "Classifier output:\n- Prediction: {label}\n- Confidence: {:.2}\n- Probability fake: {:.1}%\n- Probability real: {:.1}%\n",
        d.confidence,
        d.prob_fake * 100.0,
        d.prob_real * 100.0
    );

    if d.keywords.is_empty() {
        p.push_str("\nNo specific artifact keywords were provided. Explain what the verdict and confidence mean, and what kinds of signs usually lead detectors to such a verdict, without claiming any were observed.\n");
    } else {
        p.push_str(&format!(
            "\nFlagged artifact keywords: {}\n\nExplain how each keyword relates to the verdict ({}), and why it is a typical sign of AI-generated video (or, if the verdict is REAL, why it may be a false alarm).\n",
            d.keywords.join(", "),
            if d.label == Label::Fake { "FAKE" } else { "REAL" }
        ));
    }
    p
}

// ---------- 4. Call the Gemini API ----------

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
}
#[derive(Deserialize)]
struct Candidate {
    content: Option<Content>,
}
#[derive(Deserialize)]
struct Content {
    parts: Option<Vec<Part>>,
}
#[derive(Deserialize)]
struct Part {
    text: Option<String>,
}

/// One single call to one model.
async fn explain_once(d: &Detection, api_key: &str, model: &str) -> Result<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    );

    let body = json!({
        "systemInstruction": { "parts": [{ "text": SYSTEM_PROMPT }] },
        "contents": [{ "role": "user", "parts": [{ "text": build_prompt(d) }] }],
        "generationConfig": { "temperature": 0.4, "maxOutputTokens": 1024 }
    });

    let resp = reqwest::Client::new()
        .post(&url)
        .header("x-goog-api-key", api_key)
        .json(&body)
        .send()
        .await
        .context("Failed to reach Gemini API")?;

    let status = resp.status();
    if !status.is_success() {
        let err = resp.text().await.unwrap_or_default();
        bail!("Gemini API error {status}: {err}");
    }

    let parsed: GeminiResponse = resp.json().await.context("Bad Gemini response JSON")?;
    parsed
        .candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .map(|parts| parts.into_iter().filter_map(|p| p.text).collect::<String>())
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| anyhow!("Gemini returned no text (possibly blocked)"))
}

/// Retries one model on 503/429 with growing delays (1s, 2s, 4s).
async fn explain(d: &Detection, api_key: &str, model: &str) -> Result<String> {
    let mut delay = 1u64;
    for attempt in 1..=4 {
        match explain_once(d, api_key, model).await {
            Ok(text) => return Ok(text),
            Err(e) => {
                let msg = e.to_string();
                let retryable = msg.contains("503") || msg.contains("429");
                if !retryable || attempt == 4 {
                    return Err(e);
                }
                eprintln!("{model} busy (attempt {attempt}/4), retrying in {delay}s...");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                delay *= 2;
            }
        }
    }
    unreachable!()
}

/// Tries each model in order until one works.
async fn explain_with_fallback(d: &Detection, api_key: &str, models: &[String]) -> Result<String> {
    let mut last_err = anyhow!("No models configured");
    for m in models {
        match explain(d, api_key, m).await {
            Ok(text) => return Ok(text),
            Err(e) => {
                eprintln!("Model {m} failed, trying next...");
                last_err = e;
            }
        }
    }
    Err(last_err)
}

// ---------- 4b. Offline fallback (used when Gemini is down) ----------

fn keyword_hint(keyword: &str) -> &'static str {
    let k = keyword.to_lowercase();
    if k.contains("lip") || k.contains("sync") {
        "mouth movements that don't match the audio are a common artifact of generated or manipulated video"
    } else if k.contains("blink") || k.contains("eye") {
        "unnatural or missing blinking and odd eye movement are often seen in synthetic faces"
    } else if k.contains("blur") || k.contains("edge") {
        "soft or smeared edges, especially around the face or hair, can appear where a fake face is blended in"
    } else if k.contains("light") || k.contains("shadow") {
        "lighting or shadows that don't match the scene can suggest parts of the frame were generated"
    } else if k.contains("skin") || k.contains("texture") {
        "overly smooth or inconsistent skin texture is a frequent sign of AI-generated faces"
    } else {
        "detectors often associate this kind of artifact with manipulated video"
    }
}

fn offline_explanation(d: &Detection) -> String {
    let (verdict, p) = match d.label {
        Label::Fake => ("FAKE (likely AI-generated or manipulated)", d.prob_fake),
        Label::Real => ("REAL (likely authentic)", d.prob_real),
    };

    let gap = (d.prob_fake - d.prob_real).abs();
    let strength = if gap < 0.2 {
        "The result is not conclusive because the two probabilities are close."
    } else if p >= 0.9 {
        "The detector is very confident in this result."
    } else {
        "The detector is fairly confident, but not certain."
    };

    let mut out = format!(
        "[Offline summary - AI explanation unavailable]\nThe detector predicts {verdict} with {:.1}% probability. {strength}",
        p * 100.0
    );

    if !d.keywords.is_empty() {
        out.push_str("\n\nFlagged signs:");
        for k in &d.keywords {
            out.push_str(&format!("\n- {k}: {}", keyword_hint(k)));
        }
        if d.label == Label::Real {
            out.push_str("\n\nSince the verdict is REAL, these flags may be false alarms (for example caused by compression or low video quality).");
        }
    }

    out.push_str("\n\nNote: detectors can be wrong, so treat this as a guide and not proof.");
    out
}

// ---------- 5. CLI entry point ----------

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok(); // loads .env if present
    let api_key = env::var("GEMINI_API_KEY").context("Set GEMINI_API_KEY in env or .env")?;

    let models: Vec<String> = env::var("GEMINI_MODELS")
        .unwrap_or_else(|_| {
            "gemini-3.1-flash-lite,gemini-flash-latest,gemini-3-flash-preview".into()
        })
        .split(',')
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .collect();

    // --keywords "lip sync missing,blurry edges"
    let args: Vec<String> = env::args().collect();
    let keywords: Vec<String> = args
        .iter()
        .position(|a| a == "--keywords")
        .and_then(|i| args.get(i + 1))
        .map(|s| {
            s.split(',')
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw)?;

    let mut detection = parse_model_output(&raw)?;
    detection.keywords = keywords;

    let text = match explain_with_fallback(&detection, &api_key, &models).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Gemini unavailable, using offline summary. Reason: {e}");
            offline_explanation(&detection)
        }
    };
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_output() {
        let d = parse_model_output("Prediction: FAKE\nConfidence: 0.93\nProbability: [0.93, 0.07]").unwrap();
        assert_eq!(d.label, Label::Fake);
        assert!((d.prob_real - 0.07).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_label() {
        assert!(parse_model_output("Prediction: MAYBE\nConfidence: 0.5\nProbability: [0.5,0.5]").is_err());
    }
}