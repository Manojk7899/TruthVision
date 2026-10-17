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
use std::{
    env,
    io::{IsTerminal, Read},
};

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

const SYSTEM_PROMPT: &str = "You write the explanation shown to everyday users of an AI-video detection app, in the style of a short, professional report. \
The readers know nothing about AI or video technology. Write clearly and politely, as if explaining to an intelligent adult with no technical background.

You do NOT see the video. You only receive the result of a separate detector and, sometimes, a list of warning signs it flagged. \
Never say you watched the video and never add details you were not given. \
Never say a video IS fake or real for certain. Use words like 'likely' or 'probably'.

Style rules:
- Plain everyday words only. No jargon such as artifacts, synthesized, algorithm, model or deepfake. If you must mention a technical idea, explain it in simple words.
- Short sentences. Professional, calm and neutral tone. Do not alarm the reader or accuse whoever shared the video.
- For each warning sign: explain what it means in simple words and give a comparison from real life (for example a badly dubbed movie).
- Be honest about uncertainty. If the two chances are within 20 points of each other, clearly say the result is unclear and do not push a conclusion.
- Do not give advice, tips or instructions about what the reader should do.
- Highlighting: use **double asterisks** for bold, but ONLY for: the four section headings, the verdict (for example **probably AI-made**), the percentage (for example **93%**), and the name of each warning sign at the start of its bullet. Use no other formatting: no # symbols, no italics, no tables.

Structure: write exactly these four sections in this order. Put each heading on its own line, then its content, with a blank line between sections.

**Overview**
One short paragraph of 2 to 3 sentences: what the detector concluded and how sure it is, using the detector's confidence percentage.

**What the detector found**
One short paragraph introducing the findings. Then, if warning signs were given, one bullet per sign, each starting with '- ' in the form '- **Sign name:** explanation with a real-life comparison'. If no warning signs were given, write instead a paragraph explaining what the two percentages mean, and do not claim any specific problems were seen.

**What this means**
One paragraph explaining in plain words how to read this result: how strong it is and how much uncertainty remains. If the verdict is REAL but warning signs were flagged, explain that these can be false alarms, for example from low video quality or heavy compression.

**Conclusion**
Two to three sentences summing up the overall finding in plain words, using 'likely' or 'probably', and making clear this is an estimate and not proof.

Write about 200 to 260 words in total.";

fn build_prompt(d: &Detection) -> String {
    let verdict = match d.label {
        Label::Fake => "FAKE (likely AI-generated or edited)",
        Label::Real => "REAL (likely a genuine camera recording)",
    };

    let mut p = format!(
        "Detector result:\n- Verdict: {verdict}\n- Chance the video is fake: {:.0}%\n- Chance the video is real: {:.0}%\n",
        d.prob_fake * 100.0,
        d.prob_real * 100.0
    );

    // Confidence is only shown when it is a proper 0-1 value.
    if (0.0..=1.0).contains(&d.confidence) {
        p.push_str(&format!(
            "- Detector confidence in its verdict: {:.0}%\n",
            d.confidence * 100.0
        ));
    }

    if d.keywords.is_empty() {
        p.push_str("\nNo specific warning signs were given. Explain what this result means for an ordinary person. Do not claim any specific problems were seen.\n");
    } else {
        p.push_str(&format!(
            "\nWarning signs the detector flagged: {}\n\nFor each one, explain in simple words what it means and give a real-life comparison. Say the detector flagged it, since you cannot check the video yourself.",
            d.keywords.join(", ")
        ));
        if d.label == Label::Real {
            p.push_str(" The verdict is REAL, so also explain that these flags can be false alarms, for example from low video quality or heavy compression.");
        }
        p.push('\n');
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
        "generationConfig": { "temperature": 0.4, "maxOutputTokens": 2048 }
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

// ---------- 4c. Display helper ----------

/// Turns **bold** markers into real bold text for the terminal.
/// (When output is piped to another program, the ** markers are kept
/// so the frontend can render them as bold.)
fn bold_for_terminal(s: &str) -> String {
    s.split("**")
        .enumerate()
        .map(|(i, part)| {
            if i % 2 == 1 {
                format!("\x1b[1m{part}\x1b[0m")
            } else {
                part.to_string()
            }
        })
        .collect()
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
    let text = if std::io::stdout().is_terminal() {
        bold_for_terminal(&text)
    } else {
        text
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