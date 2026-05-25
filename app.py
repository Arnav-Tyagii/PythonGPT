import os
import glob
import time
import pandas as pd
import streamlit as st
import torch

from training.config import GPTConfig
from tokenizer.bpe_tokenizer import BPETokenizer
from model.gpt import GPT
from inference.generate import generate_code, generate_with_constraints
from evaluation.metrics import compute_perplexity, check_syntax
from training.dataset import get_dataloaders

st.set_page_config(
    page_title="PythonGPT",
    page_icon="⎔",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = ""

# ==========================================
# GLOBAL STYLES
# ==========================================
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global reset */
*, *::before, *::after { box-sizing: border-box; }

/* App background — deep navy radial gradient */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0a0e1a 0%, #050810 50%, #000208 100%) !important;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome and completely hide sidebar */
#MainMenu, footer, header, [data-testid="stSidebar"] { display: none !important; visibility: hidden; width: 0 !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

.block-container {
    padding: 2rem 1rem 4rem 1rem !important;
    max-width: 800px !important;
}

/* Custom SVG-like Title Icon */
.logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #7c6af7, #06b6d4);
    border-radius: 10px;
    display: inline-flex; align-items: center; justify-content: center;
    color: white; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
}

/* Chat Box / Input Box */
.st-key-prompt_input div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.st-key-prompt_input textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 15px !important;
    padding: 24px 130px 24px 28px !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    resize: vertical !important;
    min-height: 80px !important;
}
.st-key-prompt_input textarea:focus {
    border: 1px solid rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.1) !important;
    outline: none !important;
}

/* Circular Buttons INSIDE the chat box */
.st-key-load_btn, .st-key-gen_btn {
    height: 0px !important;
    min-height: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-end !important;
    z-index: 100 !important;
    overflow: visible !important;
}

.st-key-load_btn > div {
    margin-top: -76px !important;
    margin-right: 72px !important;
    width: 40px !important;
}

.st-key-gen_btn > div {
    margin-top: -92px !important;
    margin-right: 24px !important;
    width: 40px !important;
}

.st-key-load_btn button, .st-key-gen_btn button {
    border-radius: 50% !important;
    height: 40px !important;
    width: 40px !important;
    padding: 0 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    font-size: 1.2rem !important;
    border: 1px solid #30363d !important;
    background: #161b22 !important;
    color: #c9d1d9 !important;
    transition: all 0.2s ease !important;
}
.st-key-load_btn button:hover {
    background: #21262d !important;
    border-color: #6366f1 !important;
}
.st-key-gen_btn button {
    background: linear-gradient(135deg, #f56565, #e53e3e) !important;
    border: none !important;
    color: white !important;
}
.st-key-gen_btn button:hover {
    box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4) !important;
    transform: scale(1.05) !important;
}

/* Code output block */
[data-testid="stCode"] {
    border-radius: 12px !important;
    border: 1px solid #21262d !important;
    background: #0d1117 !important;
}

/* All other standard buttons */
div.stButton > button {
    background: transparent !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    border-color: #6366f1 !important;
    background: rgba(99, 102, 241, 0.08) !important;
}

/* Snippet cards - using native buttons directly */
.st-key-snip1 button, .st-key-snip2 button, .st-key-snip3 button, .st-key-snip4 button {
    height: 48px !important;
    border-radius: 8px !important;
    border: 1px solid #1a1f2e !important;
    background: rgba(255,255,255,0.03) !important;
    color: #c9d1d9 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
.st-key-snip1 button:hover, .st-key-snip2 button:hover, .st-key-snip3 button:hover, .st-key-snip4 button:hover {
    border-color: #6366f1 !important;
    background: rgba(99,102,241,0.05) !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
    background: #0d1117 !important;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)

# Blue Glow Background
st.markdown("""
<div style="
    position: fixed; top: -200px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 400px;
    background: radial-gradient(circle, 
        rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
"></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoints = glob.glob("checkpoints/*.pt")
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in checkpoints/ directory.")
        
        # Prioritize finetuned_model.pt, otherwise use best_model.pt
        finetune_path = "checkpoints/finetuned_model.pt"
        finetune_path_win = "checkpoints\\finetuned_model.pt"
        best_path = "checkpoints/best_model.pt"
        best_path_win = "checkpoints\\best_model.pt"
        
        if finetune_path in checkpoints or finetune_path_win in checkpoints:
            checkpoint_path = finetune_path if finetune_path in checkpoints else finetune_path_win
        elif best_path in checkpoints or best_path_win in checkpoints:
            checkpoint_path = best_path if best_path in checkpoints else best_path_win
        else:
            checkpoint_path = max(checkpoints, key=os.path.getctime)
            
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = ckpt['config']
    
    if torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
        
    tokenizer = BPETokenizer.load(config.tokenizer_path)
    model = GPT(config)
    model.load_state_dict(ckpt['model_state'])
    model.to(config.device)
    model.eval()
    return model, tokenizer, config


# ==========================================
# TOP NAV BAR
# ==========================================
nav_col1, nav_col2, nav_col3 = st.columns([7, 1.5, 1.5])
with nav_col2:
    with st.popover("History ▾"):
        if not st.session_state.history:
            st.caption("No history yet.")
        else:
            for i, past_prompt in enumerate(st.session_state.history):
                if st.button(f"Prompt {i+1}", key=f"hist_{i}"):
                    st.session_state['prompt'] = past_prompt
                    st.rerun()
                st.caption(f"{past_prompt[:50]}...")

with nav_col3:
    with st.popover("Status ▾"):
        if 'model' in st.session_state:
            st.markdown(f"<span style='color:#5cdb5c; font-size:13px;'>Engine Active ●</span><br><span style='font-size:12px; color:#888;'>{st.session_state['model'].get_num_params():,} params</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#ff4d4d; font-size:13px;'>Engine Offline ○</span>", unsafe_allow_html=True)
            
        st.divider()
        available_checkpoints = glob.glob("checkpoints/*.pt")
        
        default_index = 0
        for i, ckpt_name in enumerate(available_checkpoints):
            if "finetuned_model.pt" in ckpt_name:
                default_index = i
                break
                
        # Store checkpoint selection in session state so the main load button can access it
        selected_ckpt = st.selectbox("Checkpoint", available_checkpoints, index=default_index if available_checkpoints else None, label_visibility="collapsed", key="selected_ckpt")
                    
        st.divider()
        st.markdown("**Decoding options**")
        constrained_decoding = st.toggle("Constrained decoding", value=st.session_state.get('constrained_decoding', False), key="constrained_decoding")
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, st.session_state.get('repetition_penalty', 1.05), key="repetition_penalty")
        st.divider()
        temperature = st.slider("Temperature", 0.1, 2.0, st.session_state.get('temperature', 0.2), key="temperature")
        top_k = st.slider("Top-k", 1, 100, st.session_state.get('top_k', 40), key="top_k")
        top_p = st.slider("Top-p", 0.1, 1.0, st.session_state.get('top_p', 0.95), key="top_p")
        max_len = st.slider("Max tokens", 50, 1000, st.session_state.get('max_len', 500), key="max_len")


# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div style="text-align:center; padding: 0 0 1rem 0; margin-top: -10px;">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
        <h1 style="
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
            color: #e6edf3;
            font-family: 'Inter', sans-serif;
        ">PythonGPT</h1>
    </div>
    <p style="
        color: #4a5568;
        font-size: 0.85rem;
        letter-spacing: 0.1em;
        margin: 0;
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    ">A GPT TRAINED FROM SCRATCH ON 1GB OF PYTHON SOURCE CODE</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)

# ==========================================
# SNIPPET CARDS
# ==========================================
st.markdown("""
<div style="
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #4a5568;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
">QUICK TOPICS</div>
""", unsafe_allow_html=True)

SNIPPETS = {
    'Read CSV': 'import csv\n\ndef read_csv_file(filepath: str) -> list:\n    """Read a CSV file and return rows as list of dicts.\n    \n    Args:\n        filepath: Path to the CSV file\n    Returns:\n        List of row dictionaries\n    """\n    results = []\n    with open(filepath, \'r\') as f:\n        reader = csv.DictReader(f)\n        for row in',
    
    'Pandas Load': 'import pandas as pd\n\ndef load_and_clean(filepath: str) -> pd.DataFrame:\n    """Load CSV, drop nulls, reset index.\n    \n    Args:\n        filepath: Path to CSV\n    Returns:\n        Cleaned DataFrame\n    """\n    df = pd.read_csv(filepath)\n    df = df.dropna()\n    df =',
    
    'OS Walk': 'import os\n\ndef find_python_files(directory: str) -> list:\n    """Recursively find all .py files in a directory.\n    \n    Args:\n        directory: Root directory to search\n    Returns:\n        List of file paths\n    """\n    python_files = []\n    for root, dirs, files in os.walk(directory):\n        for file in files:\n            if file.endswith(',
    
    'JSON Parser': 'import json\n\ndef parse_config(filepath: str) -> dict:\n    """Load and parse a JSON config file.\n    \n    Args:\n        filepath: Path to JSON file\n    Returns:\n        Parsed config dictionary\n    """\n    with open(filepath, \'r\') as f:\n        config = json.load(f)\n    return',
}
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button('Read CSV', key='snip1', use_container_width=True):
        st.session_state.prompt_input = SNIPPETS['Read CSV']
with col2:
    if st.button('Pandas Load', key='snip2', use_container_width=True):
        st.session_state.prompt_input = SNIPPETS['Pandas Load']
with col3:
    if st.button('OS Walk', key='snip3', use_container_width=True):
        st.session_state.prompt_input = SNIPPETS['OS Walk']
with col4:
    if st.button('JSON Parser', key='snip4', use_container_width=True):
        st.session_state.prompt_input = SNIPPETS['JSON Parser']

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# ==========================================
# PROMPT INPUT AREA & INLINE BUTTONS
# ==========================================
prompt = st.text_area(
    label="",
    height=80,
    placeholder="Enter a Python prompt or select a snippet above...",
    label_visibility="collapsed",
    key="prompt_input"
)

# Floating buttons placed via CSS directly over the text area
load_clicked = st.button("⎔", key="load_btn", help="Load Model")
generate_clicked = st.button("➤", key="gen_btn", help="Generate Python Code")

# Handle Load Model click
if load_clicked:
    with st.spinner("Booting neural engine..."):
        try:
            model, tokenizer, config = load_model_components(st.session_state.get('selected_ckpt'))
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['config'] = config
            st.success("Engine ready!")
            time.sleep(1) # brief pause to show success message before it clears
            st.rerun()
        except Exception as e:
            st.error(f"Failed to boot: {e}")

# Token Estimate
token_estimate = int(len(prompt.split()) * 1.3) if prompt else 0
st.markdown(f"""
<div style="display: flex; justify-content: flex-end; width: 100%; padding-right: 15px; margin-top: -45px;">
    <p style="color: #4a5568; font-size: 12px; margin: 4px 0 0 0; font-family: 'Inter', sans-serif;">
        ~{token_estimate} tokens
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)


# ==========================================
# OUTPUT SECTION
# ==========================================
if generate_clicked and prompt.strip() != "":
    
    # Save to history
    if prompt not in st.session_state.history:
        st.session_state.history.insert(0, prompt)
        st.session_state.history = st.session_state.history[:3]

    if 'model' not in st.session_state:
        st.error("Please load the model using the ⎔ button in the chat box first.")
    else:
        with st.spinner("Generating..."):
            try:
                start_time = time.time()
                result = generate_with_constraints(
                    prompt=prompt,
                    model=st.session_state.model,
                    tokenizer=st.session_state.tokenizer,
                    config=st.session_state.config,
                    max_new_tokens=max_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_constraints=constrained_decoding,
                    repetition_penalty=repetition_penalty,
                )
                gen_time = time.time() - start_time
                full_code = result['full']
                
                st.markdown("""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin: 1.5rem 0 0.5rem 0;
                ">
                    <div style="
                        width: 8px; height: 8px;
                        background: #3fb950;
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                    "></div>
                    <span style="
                        color: #7d8590;
                        font-size: 12px;
                        font-family: 'Inter', sans-serif;
                        font-weight: 500;
                        letter-spacing: 0.05em;
                        text-transform: uppercase;
                    ">Generated output</span>
                </div>
                """, unsafe_allow_html=True)

                st.code(full_code, language="python")
                
                syntax_valid = check_syntax(full_code)['valid']
                stopped_naturally = result['stopped_naturally']
                constraints_applied = result['constraints_applied']
                use_constraints = constrained_decoding
                fallbacks = result['fallbacks']
                st.markdown(f"""
<div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
    <span style="background: {'rgba(63,185,80,0.1)' if syntax_valid else 'rgba(248,81,73,0.1)'}; border: 1px solid {'#3fb950' if syntax_valid else '#f85149'}; color: {'#3fb950' if syntax_valid else '#f85149'}; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-family: 'Inter', sans-serif; font-weight: 500;">
        {'Syntax valid' if syntax_valid else 'Syntax invalid'}
    </span>
    <span style="background: {'rgba(63,185,80,0.1)' if stopped_naturally else 'rgba(210,153,34,0.1)'}; border: 1px solid {'#3fb950' if stopped_naturally else '#d2993a'}; color: {'#3fb950' if stopped_naturally else '#d2993a'}; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-family: 'Inter', sans-serif; font-weight: 500;">
        {'Complete' if stopped_naturally else 'Truncated'}
    </span>
    <span style="background: rgba(56,139,253,0.1); border: 1px solid #388bfd; color: #388bfd; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-family: 'Inter', sans-serif; font-weight: 500;">
        {gen_time:.2f}s
    </span>
    <span style="background: rgba(124,106,247,0.1); border: 1px solid #7c6af7; color: #7c6af7; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-family: 'Inter', sans-serif; font-weight: 500;">
        {'Constraints on' if use_constraints else 'Constraints off'}
    </span>
    {f'''<span style="background: rgba(110,118,129,0.1); border: 1px solid #8b949e; color: #8b949e; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-family: 'Inter', sans-serif; font-weight: 500;">Auto-fixed</span>''' if result.get('was_fixed', False) else ''}
</div>
""", unsafe_allow_html=True)
                
                if fallbacks > 0:
                    st.caption(f"{fallbacks} fallbacks — model was uncertain")
            except Exception as e:
                st.error(f"Error during generation: {e}")

# ==========================================
# BOTTOM EXPANDERS
# ==========================================
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

with st.expander("Training loss curve"):
    if os.path.exists("logs/loss_log.csv"):
        df = pd.read_csv("logs/loss_log.csv")
        st.line_chart(df[['train_loss', 'val_loss']])
    else:
        st.info("Run `python train.py` first to generate loss logs.")

with st.expander("Model info"):
    if st.button("Compute perplexity"):
        if 'model' not in st.session_state:
            st.error("Please load a model first.")
        else:
            with st.spinner("Computing..."):
                try:
                    cfg = st.session_state['config']
                    tok = st.session_state['tokenizer']
                    mod = st.session_state['model']
                    _, val_loader = get_dataloaders(cfg, tok)
                    ppl = compute_perplexity(mod, val_loader, cfg)
                    st.success(f"Perplexity: {ppl:.4f}")
                except Exception as e:
                    st.error(f"Error: {e}")
