import google.generativeai as genai
import gradio as gr
import PIL.Image
import io
import requests
import urllib.parse
import os

# ==========================================
# 1. API CONFIGURATION & SAFETY
# ==========================================
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # gemini-1.5-pro या flash का उपयोग (Pro लीगल और कोडिंग रीज़निंग के लिए सबसे बेहतरीन है)
    model = genai.GenerativeModel('gemini-1.5-pro')
else:
    print("CRITICAL WARNING: GEMINI_API_KEY is missing from environment variables!")

# सिस्टम पर्सनालिटी (Dual Expert: Tech + Legal Core)
SYSTEM_PERSONA = """
You are 'HyreEdge Enterprise AI', a world-class elite Senior Technology Architect and Senior Legal Consultant/Advocate. 
Your responses must be structured, highly professional, accurate, and deeply analytical.

DOMAIN 1 - ADVANCED TECH EXPERT:
- Provide production-ready, clean, secure code (Python, JavaScript, Cloud, System Architecture).
- Explain complex architectural trade-offs, debugging, and scaling strategies.

DOMAIN 2 - LEGAL EXPERT:
- Analyze scenarios through legal frameworks, contractual obligations, compliance, and procedural logic.
- Structure arguments or draft clauses with precision (Note: Add a standard disclaimer that this is AI-assisted legal structuring, not formal court representation).

If an image generation or visual asset is requested, process it seamlessly. Always maintain an authoritative, helpful, and ultra-professional tone.
"""

# ==========================================
# 2. CORE ENGINES (IMAGE + TEXT)
# ==========================================
def generate_image_internal(prompt):
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42&model=flux"
    try:
        response = requests.get(image_url, timeout=30)
        img = PIL.Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        return None

def process_ai_request(user_input, history, uploaded_file):
    if not user_input and not uploaded_file:
        return "", history

    input_lower = user_input.lower()
    
    # इमेज जनरेशन ट्रिगर चेक
    image_triggers = ["generate image", "create image", "draw", "तस्वीर बनाओ", "फोटो बनाओ", "image of", "photo of", "flux"]
    is_image_request = any(trigger in input_lower for trigger in image_triggers)
    
    if is_image_request and not uploaded_file:
        try:
            prompt_refinement = f"Convert into a detailed graphic prompt: {user_input}. Return ONLY the prompt."
            refined_prompt = model.generate_content(prompt_refinement).text.strip()
            img_out = generate_image_internal(refined_prompt)
            if img_out:
                history.append((user_input, (img_out,)))
                return "", history
        except Exception:
            pass

    # टेक्स्ट, कोडिंग और लीगल एनालिसिस के लिए मल्टी-टर्न चैट विथ सिस्टम प्रॉम्ट
    try:
        chat_history = []
        # सिस्टम पर्सनालिटी जोड़ना
        chat_history.append({'role': 'user', 'parts': [SYSTEM_PERSONA]})
        chat_history.append({'role': 'model', 'parts': ["Understood. I am operational as the HyreEdge Dual-Expert Tech & Legal Intelligence Engine. How may I assist you today?"]})
        
        for u_msg, a_msg in history:
            if u_msg:
                chat_history.append({'role': 'user', 'parts': [u_msg]})
            if a_msg and isinstance(a_msg, str):
                chat_history.append({'role': 'model', 'parts': [a_msg]})
                
        # यदि फाइल (PDF/Text/Image) अपलोड की गई है
        current_content = [user_input]
        if uploaded_file is not None:
            try:
                # अगर फाइल इमेज है या डॉक्यूमेंट
                if uploaded_file.name.endswith(('png', 'jpg', 'jpeg', 'webp')):
                    pil_img = PIL.Image.open(uploaded_file.name)
                    current_content.append(pil_img)
                else:
                    with open(uploaded_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        file_text = f.read()
                    current_content.append(f"\n[Attached Document Content]:\n{file_text}")
            except Exception as fe:
                current_content.append(f"\n[Note: File read error: {str(fe)}]️")

        chat_history.append({'role': 'user', 'parts': current_content})
        
        response = model.generate_content(chat_history)
        ai_response = response.text
        
        history.append((user_input, ai_response))
        return "", history
        
    except Exception as e:
        err_msg = f"System Error encountered: {str(e)}"
        history.append((user_input, err_msg))
        return "", history

# ==========================================
# 3. ADVANCED PREMIUM UI (DARK GLASS THEME)
# ==========================================
custom_css = """
body {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}
#main-container {
    max-width: 1000px;
    margin: 0 auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.header-panel {
    text-align: center;
    padding: 20px 0 10px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 15px;
}
.header-panel h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.header-panel p {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 5px;
}
#chatbot-box {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_classes="header-panel"):
            gr.Markdown("# ✦ HyreEdge Enterprise AI")
            gr.Markdown("Dual-Core Intelligence • **Senior Tech Architect** & **Legal Compliance Expert**")
        
        with gr.Row():
            chatbot = gr.Chatbot(
                elem_id="chatbot-box",
                height=600,
                show_label=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=HyreEdgeSecure")
            )
            
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    placeholder="Ask complex system architecture, write code, or analyze legal drafts...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1, min_width=80):
                file_upload = gr.File(label="Upload File", file_count="single", scale=1)
                
        with gr.Row():
            submit_btn = gr.Button("Execute Query ➔", variant="primary", scale=4)
            clear = gr.ClearButton([msg, chatbot, file_upload], value="Clear Workspace", scale=1)

        # इवेंट बाइंडिंग
        msg.submit(process_ai_request, [msg, chatbot, file_upload], [msg, chatbot])
        submit_btn.click(process_ai_request, [msg, chatbot, file_upload], [msg, chatbot])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)

