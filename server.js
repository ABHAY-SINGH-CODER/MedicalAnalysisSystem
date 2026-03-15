const express = require("express");
const multer = require("multer");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require("fs");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/", limits: { fileSize: 10 * 1024 * 1024 } }); // 10MB limit

app.use(express.json());

// ── Initialize Gemini ──────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY );

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
also describe your reasoning in the "general_observations" field based on the inputs you received.
`.trim();
}

// ── POST /api/triage ───────────────────────────────────────────────────────────
// Accepts: multipart/form-data with optional `image` file and optional `symptoms` text
app.post("/api/triage", upload.single("image"), async (req, res) => {
  const startTime = Date.now();
  const symptoms = req.body.symptoms?.trim() || null;
  const imageFile = req.file || null;

  // Require at least one input
  if (!symptoms && !imageFile) {
    return res.status(400).json({
      success: false,
      error: "Provide at least one of: `symptoms` (text) or `image` (file).",
    });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const parts = [];

    // Add image part if provided
    if (imageFile) {
      const imageData = fs.readFileSync(imageFile.path).toString("base64");
      parts.push({
        inlineData: {
          data: imageData,
          mimeType: imageFile.mimetype || "image/jpeg",
        },
      });
      fs.unlinkSync(imageFile.path); // Clean up temp file
    }

    // Add prompt
    parts.push({ text: buildPrompt(symptoms) });

    const result = await model.generateContent(parts);
    const rawText = result.response.text().trim();

    // Strip markdown fences if present
    const jsonText = rawText.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```\s*$/i, "").trim();

    let assessment;
    try {
      assessment = JSON.parse(jsonText);
    } catch {
      // If JSON parse fails, return raw as fallback
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
          image: !!imageFile,
        },
        processing_time_ms: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      },
      assessment,
    });
  } catch (err) {
    // Clean up temp file on error
    if (imageFile?.path && fs.existsSync(imageFile.path)) {
      fs.unlinkSync(imageFile.path);
    }

    const status = err.status || err.statusCode || 500;
    return res.status(status).json({
      success: false,
      error: err.message || "Internal server error.",
    });
  }
});

// ── POST /api/triage/json ──────────────────────────────────────────────────────
// Alternative: accepts JSON body with base64 image
app.post("/api/triage/json", async (req, res) => {
  const startTime = Date.now();
  const { symptoms, image_base64, image_mime_type } = req.body;

  if (!symptoms && !image_base64) {
    return res.status(400).json({
      success: false,
      error: "Provide at least one of: `symptoms` (string) or `image_base64` (base64 string).",
    });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const parts = [];

    if (image_base64) {
      parts.push({
        inlineData: {
          data: image_base64,
          mimeType: image_mime_type || "image/jpeg",
        },
      });
    }

    parts.push({ text: buildPrompt(symptoms) });

    const result = await model.generateContent(parts);
    const rawText = result.response.text().trim();
    const jsonText = rawText.replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```\s*$/i, "").trim();

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
        inputs_provided: { symptoms: !!symptoms, image: !!image_base64 },
        processing_time_ms: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      },
      assessment,
    });
  } catch (err) {
    return res.status(err.status || 500).json({
      success: false,
      error: err.message || "Internal server error.",
    });
  }
});

// ── GET /api/health ────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "medical-triage-api", version: "1.0.0" });
});

// ── Start ──────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Triage API running on http://localhost:${PORT}`);
  console.log(`   POST /api/triage        → multipart/form-data (image file + symptoms text)`);
  console.log(`   POST /api/triage/json   → application/json    (base64 image + symptoms text)`);
  console.log(`   GET  /api/health        → health check`);
});