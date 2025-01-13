import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer


# Load the pretrained T5 model and tokenizer
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("../model_chatbox")
    tokenizer = T5Tokenizer.from_pretrained("../model_chatbox")
    return model, tokenizer


model, tokenizer = load_model()

# Streamlit UI
st.title("Health Care and Finance Chatbot")
st.write("Ask a question, and the T5 model will generate a response!")

# User Input
user_input = st.text_area("Your Query:", "")

if st.button("Generate Response"):
    if user_input.strip():
        # Preprocess the input
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids

        # Model prediction
        outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the response
        st.subheader("Response:")
        st.write(response.capitalize())
    else:
        st.warning("Please enter a query to get a response.")