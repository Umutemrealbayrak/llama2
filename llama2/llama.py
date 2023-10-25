from flask import Flask, render_template, request
from llama2_wrapper import LLAMA2_WRAPPER, get_prompt

app = Flask(__name__)

llama_model = LLAMA2_WRAPPER(
    model_path="",  # Model dosya yolu
    backend_type="llama.cpp",
    max_tokens=4000,
    load_in_8bit=True,
    verbose=True
)

@app.route('/', methods=['GET', 'POST'])
def index():
    completed_text = ""
    
    if request.method == 'POST':
        prompt = request.form['prompt']
        chat_history = []  # Giriş geçmişini burada tutabilirsiniz, önceki kullanıcı girişleri ve cevapları
        system_prompt = "System prompt"  # Sistem prompt metni
        prompt_with_history = get_prompt(prompt, chat_history, system_prompt)
        completed_text = llama_model(prompt_with_history, stream=False, max_new_tokens=100)
        
    return render_template('index.html', completed_text=completed_text)

if __name__ == '__main__':
    app.run(debug=True)
