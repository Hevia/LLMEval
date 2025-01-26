from enum import Enum
from typing import List
from ollama import chat, ChatResponse, Options, ResponseError, pull


class OllamaModel(Enum):
    """
    Models supported by Ollama we can use for testing.
    We picked models that come in multiple sizes.
    We ideally picked the latest version of each model if that version and had multiple sizes.
    - https://ollama.com/library/gemma
    - https://ollama.com/library/qwen2.5
    - https://ollama.com/library/phi3
    - https://ollama.com/library/vicuna
    - https://ollama.com/library/smollm2
    - https://ollama.com/library/aya
    - https://ollama.com/library/stablelm2
    - https://ollama.com/library/llama3.2
    - https://ollama.com/library/llama2
    - https://ollama.com/library/deepseek-r1
    - https://ollama.com/library/falcon3
    """

    GEMMA2B = "gemma:2b"
    GEMMA7B = "gemma:7b"
    
    
    def __str__(self):
        return self.value



class OllamaInference:
    def predict(self, model: OllamaModel, prompt: str, options: Options) -> str:
        try:
            response: ChatResponse = chat(
                model=model,
                messages=[
                    {
                    'role': 'user',
                    'content': prompt
                    }
                ],
                options=options
            )

            return response.message.content
        except ResponseError as e:
            if e.status_code == 404:
                print("Model not downloaded.... Downloading and trying again...")
                pull(model)
                return self.predict(model, prompt, options)
            else:
                print('Unknown Error:', e.error)
                return "None"

    def predict_batch(self, model: OllamaModel, prompts: List[str]) -> List[str]:
        pass


# Example usage
ollama_client = OllamaInference()
options = Options()
options.temperature = 0.0

print(ollama_client.predict(OllamaModel.GEMMA2B, "test",    options))