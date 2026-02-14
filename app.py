import gradio as gr
import torch
from PIL import Image
import logging
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    EMERGENCY = "Emergency"

@dataclass
class SymptomAnalysis:
    severity_score: int
    severity_level: str
    possible_condition: str
    visual_observations: str
    doctor_urgency: str
    specialist_type: str
    warning_signs: List[str]
    immediate_remedies: List[str]
    medications: List[str]
    lifestyle_adjustments: List[str]
    detailed_explanation: str


class BioMistralMedicalSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vision_model = None
        self.vision_processor = None

        self.llm_model = None
        self.llm_tokenizer = None

    def load_models(self):
        if self.llm_model:
            return

        logger.info("Loading models...")

        # Load ultra-lightweight vision model (Moondream2 - 2B parameters)
        vision_id = "vikhyatk/moondream2"
        try:
            from transformers import AutoModelForCausalLM
            self.vision_model = AutoModelForCausalLM.from_pretrained(
                vision_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).eval()
            self.vision_processor = AutoProcessor.from_pretrained(vision_id, trust_remote_code=True)
            logger.info(f"Loaded lightweight vision model: {vision_id}")
        except Exception as e:
            logger.warning(f"Failed to load Moondream: {e}. Trying even lighter model...")
            # Fallback to BLIP-2 (very lightweight)
            try:
                from transformers import Blip2ForConditionalGeneration, Blip2Processor
                vision_id = "Salesforce/blip2-opt-2.7b"
                self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
                    vision_id,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).eval()
                self.vision_processor = Blip2Processor.from_pretrained(vision_id)
                logger.info(f"Loaded fallback vision model: {vision_id}")
            except Exception as e2:
                logger.error(f"Failed to load vision models: {e2}")
                raise

        # Load lightweight LLM (Llama 3.2 1B Instruct)
        llm_id = "meta-llama/Llama-3.2-1B-Instruct"
        try:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True) if self.device == "cuda" else None

            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_id,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).eval()
            logger.info(f"Loaded lightweight LLM: {llm_id}")
        except Exception as e:
            logger.warning(f"Failed to load Llama 3.2: {e}. Trying TinyLlama...")
            # Fallback to even lighter TinyLlama (1.1B)
            try:
                llm_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_id,
                    quantization_config=bnb_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).eval()
                logger.info(f"Loaded fallback LLM: {llm_id}")
            except Exception as e2:
                logger.error(f"Failed to load LLM models: {e2}")
                raise

    def analyze_visuals(self, image: Image.Image):
        # Moondream2 has a simple encode_image + query interface
        try:
            # For Moondream2
            image_embeds = self.vision_model.encode_image(image)
            prompt = "Describe any visible medical symptoms, skin conditions, injuries, or abnormalities in this image. Be specific and clinical."
            
            response = self.vision_model.answer_question(
                image_embeds, 
                prompt, 
                self.vision_processor.tokenizer
            )
            return response
        except AttributeError:
            # Fallback for BLIP-2 or other models
            prompt = "Question: Describe visible medical symptoms in this image. Answer:"
            inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vision_model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    temperature=0.3
                )
            
            response = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
            return response

    def get_biomistral_analysis(self, symptoms, visual_text):
        # Llama 3.2 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical AI assistant providing preliminary health analysis. Be professional, clear, and helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

**Patient Symptoms:**
{symptoms}

**Visual Findings:**
{visual_text}

Please provide a detailed medical analysis including:
1. Clinical interpretation of symptoms
2. Possible conditions or causes
3. Risk level assessment
4. When to seek medical care (urgent vs routine)
5. Self-care recommendations
6. Warning signs to watch for

Be thorough but concise.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

        text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in text:
            text = text.split("assistant")[-1].strip()
        if "<|eot_id|>" in text:
            text = text.split("<|eot_id|>")[0].strip()
            
        return text

    def run_triage(self, image, symptoms):
        self.load_models()

        visual_desc = "No image provided."
        if image:
            visual_desc = self.analyze_visuals(image)

        llm_resp = self.get_biomistral_analysis(symptoms, visual_desc)

        score = 8 if "chest" in symptoms.lower() else 4
        level = SeverityLevel.HIGH.value if score >= 8 else SeverityLevel.MODERATE.value

        return SymptomAnalysis(
            severity_score=score,
            severity_level=level,
            possible_condition="Refer analysis",
            visual_observations=visual_desc,
            doctor_urgency="Emergency care recommended" if score >= 8 else "Routine consultation",
            specialist_type="General Physician",
            warning_signs=["Breathing difficulty", "Severe pain"],
            immediate_remedies=["Rest", "Hydration"],
            medications=["Consult physician"],
            lifestyle_adjustments=["Monitor symptoms"],
            detailed_explanation=llm_resp
        )


medical_system = BioMistralMedicalSystem()

def format_output(a: SymptomAnalysis):
    return f"""
MEDICAL TRIAGE REPORT
================================================

Severity Level : {a.severity_level}
Severity Score : {a.severity_score}/10
Urgency : {a.doctor_urgency}

------------------------------------------------
CLINICAL ANALYSIS
------------------------------------------------
{a.detailed_explanation}

------------------------------------------------
VISUAL ASSESSMENT
------------------------------------------------
{a.visual_observations}

------------------------------------------------
WARNING SIGNS
------------------------------------------------
{chr(10).join(['- ' + s for s in a.warning_signs])}

DISCLAIMER:
This AI output is informational only and not medical advice.
"""

def process(img, text, progress=gr.Progress()):
    try:
        progress(0, desc="Initializing medical AI system...")
        
        # Load models
        progress(0.2, desc="Loading AI models (this may take a moment)...")
        medical_system.load_models()
        
        # Analyze image if provided
        if img:
            progress(0.5, desc="Analyzing medical image...")
        else:
            progress(0.5, desc="Processing symptoms...")
        
        # Run triage
        progress(0.7, desc="Running medical analysis...")
        result = medical_system.run_triage(img, text)
        
        progress(1.0, desc="Analysis complete!")
        return format_output(result)
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return f"""
# Error During Analysis

An error occurred while processing your request:
```
{str(e)}
```

Please try again or contact support if the issue persists.
"""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Medical AI Triage System")
    gr.Markdown("**⚠️ Disclaimer:** This is for informational purposes only and not a substitute for professional medical advice.")
    gr.Markdown("*Powered by Moondream2 (2B) + Llama 3.2 (1B) - Ultra-Lightweight & Fast*")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload Medical Image (Optional)")
        with gr.Column():
            txt_in = gr.Textbox(
                lines=5, 
                label="Describe Your Symptoms", 
                placeholder="e.g., chest pain, difficulty breathing, rash on arm..."
            )
    
    btn = gr.Button("🔍 Analyze Symptoms", variant="primary", size="lg")
    
    with gr.Column():
        out = gr.Markdown(label="Analysis Results")
    
    gr.Markdown("""
    ### How to use:
    1. (Optional) Upload a photo of the affected area for visual analysis
    2. Describe your symptoms in detail
    3. Click "Analyze Symptoms" and wait for the analysis
    
    **⚡ Lightning Fast:** Ultra-lightweight models load in ~20-30 seconds
    **🤖 AI Models:** Moondream2 (2B) for images + Llama 3.2 (1B) for medical reasoning
    **💾 Low Resources:** Works on modest hardware (4GB+ RAM recommended)
    """)

    btn.click(
        fn=process, 
        inputs=[img_in, txt_in], 
        outputs=out,
        show_progress=True
    )

demo.launch(share=False)