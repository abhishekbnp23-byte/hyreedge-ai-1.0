
import google.generativeai as genai
import gradio as gr
import PIL.Image
import io
import requests
import urllib.parse
import os

# ==========================================
# 1. API KEY CONFIGURATION (SAFE & SECURE)
# ==========================================
# API Key को Render के Environment Variables से लिया जाएगा
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_API_KEY environment variable is missing!")

# ==========================================
# 2. CORE LOGIC FUNCTIONS
# ==========================================
def generate_image_internal(prompt):
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42&model=flux"
    response = requests.get(image_url, timeout=30)
    img = PIL.Image.open(io.BytesIO(response.content))
    return img

def ask_multimodal_ai(user_input, history):
    input_lower = user_input.lower()
    
    image_triggers = ["generate image", "create image", "draw", "तस्वीर बनाओ", "फोटो बनाओ", "चित्र बनाओ", "image of", "photo of"]
    is_image = any(trigger in input_lower for trigger in image_triggers)
    
    if is_image:
        prompt_refinement = f"Convert the user request into a detailed English image prompt: {user_input}. Return ONLY the prompt text."
        refined_prompt = model.generate_content(prompt_refinement).text.strip()
        image_out = generate_image_internal(refined_prompt)
        return None, image_out
    else:
        messages = []
        for user_msg, ai_msg in history:
            if user_msg:
                messages.append({'role': 'user', 'parts': [user_msg]})
            if ai_msg and isinstance(ai_msg, str):
                messages.append({'role': 'model', 'parts': [ai_msg]})
                
        messages.append({'role': 'user', 'parts': [user_input]})
        response = model.generate_content(messages)
        return response.text, None

def user_chat(message, history):
    text_out, img_out = ask_multimodal_ai(message, history)
    
    if img_out is not None:
        history.append((message, (img_out,)))
    else:
        history.append((message, text_out))
        
    return "", history

# ==========================================
# 3. MODERN UI & CSS SETUP
# ==========================================
custom_css = """
#main-container {
    max-width: 900px;
    margin: 0 auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.header-box {
    text-align: center;
    padding: 15px 0;
    margin-bottom: 10px;
}
#chatbot-box {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        with gr.Column(elem_classes="header-box"):
            gr.Markdown("# ✦ HyreEdge AI Engine")
            gr.Markdown("Next-Generation Multi-Modal Intelligence • Code • Reasoning • Vision")
        
        chatbot = gr.Chatbot(
            elem_id="chatbot-box",
            height=550,
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=HyreEdge")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask anything or request an image generation...",
                show_label=False,
                scale=8,
                container=False
            )
            submit_btn = gr.Button("Send ➔", variant="primary", scale=1)
        
        gr.ClearButton([msg, chatbot], value="Clear Conversation")

        msg.submit(user_chat, [msg, chatbot], [msg, chatbot])
        submit_btn.click(user_chat, [msg, chatbot], [msg, chatbot])

# ==========================================
# 4. RENDER SERVER BINDING
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
