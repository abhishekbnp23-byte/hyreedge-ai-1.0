import google.generativeai as genai
import gradio as gr
import PIL.Image
import io
import requests
import urllib.parse
import os

# Render Environment Variable से API Key रीड करना
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_API_KEY environment variable is missing!")

def generate_image_internal(prompt):
    """Pollinations Flux Engine से छवि डाउनलोड करके PIL फ़ॉर्मेट में देना"""
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed=42&model=flux"
    
    response = requests.get(image_url, timeout=30)
    img = PIL.Image.open(io.BytesIO(response.content))
    return img

def ask_multimodal_ai(user_input, history):
    input_lower = user_input.lower()
    
    # छवि जनरेशन की पहचान
    image_triggers = ["generate image", "create image", "draw", "तस्वीर बनाओ", "फोटो बनाओ", "चित्र बनाओ", "image of", "photo of"]
    is_image = any(trigger in input_lower for trigger in image_triggers)
    
    if is_image:
        # LLM से प्रॉम्ट को इमेज-फ्रेंडली इंग्लिश में बदलना
        prompt_refinement = f"Convert the user request into a detailed English image prompt: {user_input}. Return ONLY the prompt text, no extra conversational filler."
        refined_prompt = model.generate_content(prompt_refinement).text.strip()
        
        image_out = generate_image_internal(refined_prompt)
        return None, image_out
    
    else:
        # टेक्स्ट और कोडिंग रिस्पॉन्स
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

# कस्टम UI डिजाइन
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# HYREEDGE AI ENGINE 1.0")
    gr.Markdown("An enterprise-grade multi-modal AI for text, code reasoning, and image generation.")
    
    chatbot = gr.Chatbot(height=550)
    msg = gr.Textbox(placeholder="अपनी क्वेरी या प्रॉम्ट लिखें...")
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(user_chat, [msg, chatbot], [msg, chatbot])

# Render वेब सर्वर सेटिंग्स
if __name__ == "__main__":
    # Render का पोर्ट असाइनमेंट (PORT 7860 पर रन करना आवश्यक है)
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
