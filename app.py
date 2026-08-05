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
    print("CRITICAL WARNING: GEMINI_API_KEY is missing from environment variables!")

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

def process_ai_request(user_input, history, uploaded_file):
    if history is None:
        history = []
        
    input_text = user_input.strip() if user_input else ""
    input_lower = input_text.lower()

    if not input_text and not uploaded_file:
        return "", history

    # 1. इमेज जनरेशन चेक
    image_triggers = ["generate image", "create image", "draw", "तस्वीर बनाओ", "फोटो बनाओ", "चित्र बनाओ", "image of", "photo of", "flux"]
    is_image_request = any(trigger in input_lower for trigger in image_triggers)

    if is_image_request and not uploaded_file:
        try:
            prompt_refinement = f"Convert into a detailed English visual image prompt: {input_text}. Return ONLY prompt text."
            refined_prompt = model.generate_content(prompt_refinement).text.strip()
            img_out = generate_image_internal(refined_prompt)
            if img_out:
                history.append((input_text, (img_out,)))
                return "", history
        except Exception:
            pass

    # 2. Gemini API प्रोसेसिंग
    try:
        gemini_messages = [
            {'role': 'user', 'parts': [SYSTEM_PERSONA]},
            {'role': 'model', 'parts': ["Understood. HyreEdge Enterprise Tech & Legal AI Engine is online."]}
        ]

        for user_msg, ai_msg in history:
            if user_msg:
                gemini_messages.append({'role': 'user', 'parts': [str(user_msg)]})
            if ai_msg and isinstance(ai_msg, str):
                gemini_messages.append({'role': 'model', 'parts': [ai_msg]})

        current_parts = []
        if input_text:
            current_parts.append(input_text)

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
                    pil_img = PIL.Image.open(uploaded_file.name)
                    current_parts.append(pil_img)
                else:
                    with open(uploaded_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        file_text = f.read()
                    current_parts.append(f"\n[Uploaded File Content]:\n{file_text}")
            except Exception as fe:
                current_parts.append(f"\n[File Reading Note: {str(fe)}]")

        gemini_messages.append({'role': 'user', 'parts': current_parts})

        response = model.generate_content(gemini_messages)
        ai_reply = response.text if response and response.text else "Response generated successfully."

        history.append((input_text if input_text else "[File Uploaded]", ai_reply))
        return "", history

    except Exception as e:
        error_msg = f"System Processing Error: {str(e)}"
        history.append((input_text, error_msg))
        return "", history

# ==========================================
# 3. UI SETUP
# ==========================================
custom_css = """
body {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}
#main-container {
    max-width: 950px;
    margin: 0 auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.header-panel {
    text-align: center;
    padding: 20px 0 10px 0;
    margin-bottom: 10px;
}
.header-panel h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#chatbot-box {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_classes="header-panel"):
            gr.Markdown("# ✦ HyreEdge Enterprise AI")
            gr.Markdown("Dual-Core Intelligence • **Senior Tech Architect** & **Legal Compliance Expert**")

        chatbot = gr.Chatbot(
            elem_id="chatbot-box",
            height=580,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=HyreEdgeSecure")
        )

        with gr.Row():
            with gr.Column(scale=7):
                msg = gr.Textbox(
                    placeholder="Ask complex tech questions, write code, or request legal analysis...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=2):
                file_upload = gr.File(label="Upload File", file_count="single", container=False)

        with gr.Row():
            submit_btn = gr.Button("Execute Query ➔", variant="primary", scale=4)
            clear = gr.ClearButton([msg, chatbot, file_upload], value="Clear Workspace", scale=1)

        msg.submit(process_ai_request, [msg, chatbot, file_upload], [msg, chatbot])
        submit_btn.click(process_ai_request, [msg, chatbot, file_upload], [msg, chatbot])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
