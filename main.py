from caregraph import get_chain

def start_caregraph():
    print("--- Loading CareGraph reasoning engine ---")
    
    try:
        qa_chain = get_chain()
    except Exception as e:
        print(f"Error initializing CareGraph: {e}")
        return

    # 6. RUN A TEST QUERY
    print("\nCareGraph is ready. Ask a question (e.g., 'What is diabetes?') or type 'exit'")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            print("\nCareGraph: Searching medical guidelines...")
            response = qa_chain.invoke(user_input)
            print(f"\n{response}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Tip: Check if your API Key is valid and you have 'Generative AI' enabled in Google AI Studio.")

if __name__ == "__main__":
    start_caregraph()
