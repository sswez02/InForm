from src.ft.generation import DomainLLM


def main():
    llm = DomainLLM(
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        lora_dir="model_lora",
        max_new_tokens=16,
        max_input_tokens=256,
    )

    instruction = (
        "Answer the user's question using the provided study excerpts. "
        "Include inline numeric citations like [1], [2]."
    )

    query = "Does creatine help beginners gain strength?"
    context = [
        {
            "study_id": 2,
            "citation_index": 1,
            "section": "abstract",
            "text": "This randomized trial examined the effects of daily creatine "
            "monohydrate supplementation on strength and muscle hypertrophy "
            "in resistance-trained men over 12 weeks...",
        },
        {
            "study_id": 1,
            "citation_index": 2,
            "section": "introduction",
            "text": "Creatine monohydrate has become a popular ergogenic aid due "
            "to its effects on strength and lean body mass...",
        },
    ]

    print("Calling LLM.generate_answer()...")
    answer = llm.generate_answer(
        instruction=instruction,
        query=query,
        context_passages=context,
    )
    print("Got answer:\n", answer)


if __name__ == "__main__":
    main()
