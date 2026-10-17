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
- If the verdict is UNCLEAR: say in the Overview that the detector could not reach a clear answer and that the video could be either real or fake. Never call the video real or fake in this case. Explain in the second section that the chances were close, and end the Conclusion by saying no firm answer can be given.
- If the verdict is REAL: say clearly in the Overview that the video is likely a genuine, real recording. In the second section, explain that the detector did not find strong signs of AI creation or editing, without naming specific features you were not given. Keep the same calm, professional tone, and do not sound suspicious or disappointed.
- Highlighting: use **double asterisks** for bold, but ONLY for: the four section headings, the verdict (for example **probably AI-made** or **likely a real recording**), the percentage (for example **93%**), and the name of each warning sign at the start of its bullet. Use no other formatting: no # symbols, no italics, no tables.

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
    // If the two chances are close, the result is unclear. Decide this in code
    // so Gemini cannot talk itself into a firm verdict.
    let unclear = (d.prob_fake - d.prob_real).abs() < 0.2;

    let verdict = if unclear {
        match d.label {
            Label::Fake => "UNCLEAR (too close to call, leaning very slightly towards fake)",
            Label::Real => "UNCLEAR (too close to call, leaning very slightly towards real)",
        }
    } else {
        match d.label {
            Label::Fake => "FAKE (likely AI-generated or edited)",
            Label::Real => "REAL (likely a genuine camera recording)",
        }
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

    if unclear {
        p.push_str("\nIMPORTANT: the two chances are close, so the result is UNCLEAR. Do NOT say the video is real and do NOT say it is fake. Say clearly that the detector could not reach a clear answer, and that the video could be either. In the Conclusion, state that no firm answer can be given.\n");
    }

    if d.keywords.is_empty() {
        if unclear {
            // nothing more to add
        } else if d.label == Label::Real {
            p.push_str("\nThe verdict is REAL. Clearly tell the reader that this video is likely genuine, and explain that the detector found no strong signs of AI creation or editing. Do not invent specific reasons or features.\n");
        } else {
            p.push_str("\nNo specific warning signs were given. Explain what this result means for an ordinary person. Do not claim any specific problems were seen.\n");
        }
    } else {
        p.push_str(&format!(
            "\nWarning signs the detector flagged: {}\n\nFor each one, explain in simple words what it means and give a real-life comparison. Say the detector flagged it, since you cannot check the video yourself.",
            d.keywords.join(", ")
        ));
        if d.label == Label::Real && !unclear {
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
        "The movement of the mouth does not line up well with the words being spoken, a bit like a badly dubbed movie."
    } else if k.contains("blink") || k.contains("eye") {
        "The eyes blink too little, too much, or in an unnatural way, like a character that does not quite move like a real person."
    } else if k.contains("blur") || k.contains("edge") {
        "The outline around the person looks soft or smudged, like a cut-out picture pasted onto a background."
    } else if k.contains("light") || k.contains("shadow") {
        "The lighting or shadows on the person do not match the rest of the scene, like a sticker placed on top of a photo."
    } else if k.contains("skin") || k.contains("texture") {
        "The skin looks unusually smooth or uneven, like a heavily airbrushed photo."
    } else {
        "The detector noticed something unusual here that it often sees in edited or computer-made videos."
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Same four-section layout as the Gemini answer, built without any API call.
fn offline_explanation(d: &Detection) -> String {
    let (verdict_text, p) = match d.label {
        Label::Fake => ("probably AI-made or edited", d.prob_fake),
        Label::Real => ("probably a genuine camera recording", d.prob_real),
    };
    let pct = p * 100.0;
    let unclear = (d.prob_fake - d.prob_real).abs() < 0.2;

    let mut out = String::new();

    // 1. Overview
    out.push_str("**Overview**\n");
    if unclear {
        out.push_str(&format!(
            "The detector could not reach a clear answer for this video. The chances are close: about **{:.0}%** fake and **{:.0}%** real.\n\n",
            d.prob_fake * 100.0,
            d.prob_real * 100.0
        ));
    } else {
        out.push_str(&format!(
            "The detector checked this video and concluded that it is **{verdict_text}**. It is about **{pct:.0}%** sure of this result.\n\n"
        ));
    }

    // 2. What the detector found
    out.push_str("**What the detector found**\n");
    if d.keywords.is_empty() {
        if d.label == Label::Real {
            out.push_str(&format!(
                "The detector did not find strong signs that this video was made or edited by AI. It gave a **{:.0}%** chance that the video is real and a **{:.0}%** chance that it is fake.\n\n",
                d.prob_real * 100.0,
                d.prob_fake * 100.0
            ));
        } else {
            out.push_str(&format!(
                "The detector gave two percentages: a **{:.0}%** chance that the video is fake and a **{:.0}%** chance that it is real. It did not point out any specific problems in the video.\n\n",
                d.prob_fake * 100.0,
                d.prob_real * 100.0
            ));
        }
    } else {
        out.push_str("The detector flagged the following warning signs in this video:\n");
        for k in &d.keywords {
            out.push_str(&format!("- **{}:** {}\n", capitalize(k), keyword_hint(k)));
        }
        out.push('\n');
    }

    // 3. What this means
    out.push_str("**What this means**\n");
    let meaning = if unclear {
        "Because the two chances are so close, this result is unclear. The detector cannot say with any confidence whether the video is real or fake."
    } else if d.label == Label::Real {
        if p >= 0.9 {
            "The detector is very sure that this looks like genuine camera footage. It did not find strong signs of AI creation or editing."
        } else {
            "The detector leans towards this being genuine camera footage, but it is not completely certain. Some doubt remains."
        }
    } else if p >= 0.9 {
        "The detector is very sure about this result, which means the patterns it looks for were strong."
    } else {
        "The detector leans towards this result, but it is not completely certain. Some doubt remains."
    };
    out.push_str(meaning);
    if d.label == Label::Real && !d.keywords.is_empty() {
        out.push_str(" Some warning signs were flagged even though the verdict is real. These can be false alarms, for example when a video is low quality or has been heavily compressed.");
    }
    out.push_str("\n\n");

    // 4. Conclusion
    out.push_str("**Conclusion**\n");
    if unclear {
        out.push_str("Overall, the result is unclear. It is an estimate and should not be treated as proof either way.");
    } else {
        out.push_str(&format!(
            "Overall, this video is {verdict_text}. This is an estimate based on the detector's analysis, and not proof."
        ));
    }
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
    fn offline_has_all_sections() {
        let mut d = parse_model_output("Prediction: FAKE\nConfidence: 0.93\nProbability: [0.93, 0.07]").unwrap();
        d.keywords = vec!["lip sync missing".into()];
        let t = offline_explanation(&d);
        for h in ["**Overview**", "**What the detector found**", "**What this means**", "**Conclusion**"] {
            assert!(t.contains(h), "missing {h}");
        }
        assert!(t.contains("Lip sync missing"));
    }

    #[test]
    fn offline_real_says_genuine() {
        let d = parse_model_output("Prediction: REAL\nConfidence: 0.95\nProbability: [0.05, 0.95]").unwrap();
        let t = offline_explanation(&d);
        assert!(t.contains("genuine camera recording"));
        assert!(t.contains("did not find strong signs"));
    }

    #[test]
    fn rejects_bad_label() {
        assert!(parse_model_output("Prediction: MAYBE\nConfidence: 0.5\nProbability: [0.5,0.5]").is_err());
    }
}