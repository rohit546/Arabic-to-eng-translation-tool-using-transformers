import streamlit as st
import torch
import json

# Load vocabularies
with open("src_vocab.json", "r") as f:
    src_vocab = json.load(f)
with open("tgt_vocab.json", "r") as f:
    tgt_vocab = json.load(f)
tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}  # Inverse vocabulary for decoding

# Load the model
device = torch.device("cpu")  # Use CPU for inference
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load("transformer_nmt.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Preprocess input text
def preprocess_input(text, vocab):
    tokens = text.split() + ["<eos>"]
    indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    return torch.LongTensor(indices).unsqueeze(0).to(device)  # Add batch dimension

# Generate translation
def translate(text, max_len=100):
    # Preprocess input
    src = preprocess_input(text, src_vocab)
    
    # Initialize target sequence with <sos>
    tgt = torch.LongTensor([[tgt_vocab["<sos>"]]]).to(device)
    
    # Generate translation token by token
    for _ in range(max_len):
        with torch.no_grad():  # Disable gradient calculation
            output = model(src, tgt)
        next_token = output.argmax(-1)[:, -1:]  # Greedy decoding
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if <eos> is generated
        if next_token.item() == tgt_vocab["<eos>"]:
            break
    
    # Convert indices to tokens
    translation = " ".join([tgt_vocab_inv.get(idx.item(), "") for idx in tgt[0][1:-1]])
    return translation

# Streamlit app
st.title("Arabic to English Translator")

# Input text box
input_text = st.text_area("Enter Arabic text:", "مرحبًا كيف حالك؟")

# Translate button
if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some Arabic text.")
    else:
        # Generate translation
        translated_text = translate(input_text)
        
        # Display translation
        st.success("Translation:")
        st.write(translated_text)