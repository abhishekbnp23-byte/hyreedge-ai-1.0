import google.generativeai as genai
import gradio as gr
import PIL.Image
import io
import requests
import urllib.parse
import os

# ==========================================
# 1. API CONFIGURATION
# ==========================================
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("CRITICAL WARNING: GEMINI_API_KEY is missing!")

SYSTEM_PERSONA = (
    "You are 'HyreEdge Enterprise AI', an elite Senior Technology Architect and Senior Legal Consultant/Advocate.\n"
    "Provide authoritative, accurate, structured, and deeply analytical responses.\n\n"
    "1. TECH EXPERT ROLE: Offer clean, robust code, cloud architectures, debugging, and system engineering guidance.\n"
    "2. LEGAL EXPERT ROLE: Offer clear legal analysis, contractual drafting structures, compliance insights, and procedural frameworks.\n\n"
    "Maintain a professional, clear, and highly competent tone at all times."
)

# ==========================================
# 2. CORE ENGINES
# ==========================================
def generate_image_internal(prompt):
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42&model=flux"
    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            img = PIL.Image.open(io.BytesIO(response.content))
            return img
    except Exception:
        pass
    return None

def bot_response(message, history):
    text_content = ""
    file_obj = None

    # Gradio ChatInterface से इनपुट हैंडलिंग
    if isinstance(message, dict):
        text_content = message.get("text", "").strip()
        files = message.get("files", [])
        if files:
            file_obj = files[0]
    else:
        text_content = str(message).strip()

    if not text_content and not file_obj:
        return "Please provide text or attach a document."

    input_lower = text_content.lower()

    # 1. इमेज जनरेशन चेक
    image_triggers = ["generate image", "create image", "draw", "तस्वीर बनाओ", "फोटो बनाओ", "चित्र बनाओ", "image of", "photo of", "flux"]
    is_image = any(trigger in input_lower for trigger in image_triggers)

    if is_image and not file_obj:
        try:
            prompt_refinement = f"Convert into a detailed English visual image prompt: {text_content}. Return ONLY prompt text."
            refined_prompt = model.generate_content(prompt_refinement).text.strip()
            img_out = generate_image_internal(refined_prompt)
            if img_out:
                return gr.FileData(value=img_out)
        except Exception:
            pass

    # 2. Gemini API प्रोसेसिंग (System Persona के साथ)
    try:
        gemini_messages = [
            {'role': 'user', 'parts': [SYSTEM_PERSONA]},
            {'role': 'model', 'parts': ["Understood. HyreEdge Enterprise Tech & Legal AI Engine is online."]}
        ]

        # पुरानी चैट हिस्ट्री जोड़ना
        for h in history:
            role = 'user' if h['role'] == 'user' else 'model'
            content = h['content']
            if isinstance(content, str):
                gemini_messages.append({'role': role, 'parts': [content]})

        current_parts = []
        if text_content:
            current_parts.append(text_content)

        if file_obj:
            try:
                file_path = file_obj.get("path") if isinstance(file_obj, dict) else str(file_obj)
                if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
                    pil_img = PIL.Image.open(file_path)
                    current_parts.append(pil_img)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_text = f.read()
                    current_parts.append(f"\n[Uploaded File Content]:\n{file_text}")
            except Exception as fe:
                current_parts.append(f"\n[File Reading Note: {str(fe)}]")

        gemini_messages.append({'role': 'user', 'parts': current_parts})

        response = model.generate_content(gemini_messages)
        return response.text if response and response.text else "Unable to generate response."

    except Exception as e:
        return f"System Processing Error: {str(e)}"

# ==========================================
# 3. STABLE GRADIO CHATINTERFACE UI
# ==========================================
custom_css = """
body { background-color: #0f172a !important; color: #f8fafc !important; }
#main-container { max-width: 950px; margin: 0 auto; font-family: 'Inter', system-ui, sans-serif; }
.header-panel { text-align: center; padding: 15px 0; margin-bottom: 5px; }
.header-panel h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_classes="header-panel"):
            gr.Markdown("# ✦ HyreEdge Enterprise AI")
            gr.Markdown("Dual-Core Intelligence • **Senior Tech Architect** & **Legal Compliance Expert**")

        gr.ChatInterface(
            fn=bot_response,
            type="messages",
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                placeholder="Ask complex tech questions, write code, or request legal analysis...",
                container=False,
                scale=7
            )
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
