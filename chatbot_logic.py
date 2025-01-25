from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize chatbot model and tokenizer
chatbot_model_path = "chatbot_model"
tokenizer = AutoTokenizer.from_pretrained(chatbot_model_path)
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_model_path)

def generate_response(user_input, detected_emotion):
    # Use detected emotion as system context
    system_context = f"The user is feeling {detected_emotion}. Respond mindfully."

    # Generate chatbot response
    input_text = f"{system_context} {user_input}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response_ids = chatbot_model.generate(input_ids, max_length=150, num_return_sequences=1)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response_text
