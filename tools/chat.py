import llm

def short(text):
    return llm.run_llm(text)

def long(text):
    return "Let me think about that."
    # later: call bigger model / remote API
