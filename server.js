const express = require("express");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
app.use(express.json({ limit: "20mb" })); // Allow large base64 payloads
app.use(express.static(__dirname));

// ── Initialize Gemini ──────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── Helper: build structured prompt ───────────────────────────────────────────
function buildPrompt(symptoms) {
  return `
You are a preliminary medical triage assistant for a student project.

${symptoms ? `USER SYMPTOMS: "${symptoms}"` : "No symptoms text provided. Analyze the image only."}

TASK:
Analyze the provided information and respond ONLY with a valid JSON object (no markdown, no extra text) in this exact format:

{
  "risk_level": "Low" | "Medium" | "High" | "Critical",
  "risk_score": <integer 1–10>,
  "potential_concerns": ["concern1", "concern2"],
  "recommendation": "<clear action recommendation>",
  "seek_emergency_care": <true | false>,
  "follow_up_timeframe": "<e.g., 'Within 24 hours', 'Within 1 week', 'Immediately'>",
  "general_observations": "<brief summary of what was assessed>",
  "disclaimer": "This is an AI-generated preliminary assessment for educational purposes only. It is NOT professional medical advice. Always consult a licensed healthcare provider for diagnosis and treatment."
}

Be conservative: when uncertain, escalate the risk level.
Also describe your reasoning in the "general_observations" field based on the inputs you received.

Give the description of image in large paragraph as a doctor and potential diagnoses based on the image and symptoms in the "general_observations" field. Be detailed and thorough in your analysis.
`.trim();
}

// ── Helper: validate and clean base64 string ───────────────────────────────────
function parseBase64Image(raw) {
  if (!raw || typeof raw !== "string") return null;

  // Strip data URI prefix if present: "data:image/png;base64,<data>"
  const match = raw.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
  if (match) {
    return { mimeType: match[1], data: match[2] };
  }

  // Plain base64 string (no prefix) — default to jpeg
  return { mimeType: "image/jpeg", data: raw };
}

// ── POST /api/triage ───────────────────────────────────────────────────────────
// Accepts: JSON body with optional base64 image and optional symptoms text
// Body: { symptoms?: string, image_base64?: string, image_mime_type?: string }
app.post("/api/triage", async (req, res) => {
  const startTime = Date.now();
  const { symptoms: rawSymptoms, image_base64, image_mime_type } = req.body;

  const symptoms = rawSymptoms?.trim() || null;
  const parsedImage = parseBase64Image(image_base64);

  // Override mime type if explicitly provided
  if (parsedImage && image_mime_type) {
    parsedImage.mimeType = image_mime_type;
  }

  if (!symptoms && !parsedImage) {
    return res.status(400).json({
      success: false,
      error: "Provide at least one of: `symptoms` (text) or `image_base64` (base64 string).",
    });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const parts = [];

    if (parsedImage) {
      parts.push({
        inlineData: {
          data: parsedImage.data,
          mimeType: parsedImage.mimeType,
        },
      });
    }

    parts.push({ text: buildPrompt(symptoms) });

    const result = await model.generateContent(parts);
    const rawText = result.response.text().trim();

    // Strip markdown fences if present
    const jsonText = rawText
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/```\s*$/i, "")
      .trim();

    let assessment;
    try {
      assessment = JSON.parse(jsonText);
    } catch {
      return res.status(502).json({
        success: false,
        error: "Model returned non-JSON response.",
        raw_response: rawText,
      });
    }

    return res.status(200).json({
      success: true,
      meta: {
        model: "gemini-2.5-flash",
        inputs_provided: {
          symptoms: !!symptoms,
          image: !!parsedImage,
        },
        processing_time_ms: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      },
      assessment,
    });
  } catch (err) {
    const status = err.status || err.statusCode || 500;
    return res.status(status).json({
      success: false,
      error: err.message || "Internal server error.",
    });
  }
});

// ── GET /api/health ────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "medical-triage-api", version: "2.0.0" });
});

// ── Start ──────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Triage API running on http://localhost:${PORT}`);
  console.log(`   POST /api/triage   → application/json (base64 image + symptoms text)`);
  console.log(`   GET  /api/health   → health check`);
});