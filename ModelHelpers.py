from enum import Enum
from typing import List
from ollama import chat, ChatResponse, Options, ResponseError, pull, delete


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
    - https://ollama.com/library/stablelm2
    - https://ollama.com/library/llama3.2
    - https://ollama.com/library/llama2
    - https://ollama.com/library/deepseek-r1
    - https://ollama.com/library/falcon3
    """

    GEMMA2B = "gemma:2b"
    GEMMA7B = "gemma:7b"
    FALCON37B = "falcon3:7b"
    FALCON31B = "falcon3:1b"
    FALCON33B = "falcon3:3b"
    FALCON310B = "falcon3:10b"
    LLAMA27B = "llama2:7b"
    LLAMA213B = "llama2:13b"
    STABLEM21p6B = "stablelm2:1.6b"
    STABLEM212B = "stablelm2:12b"
    SMOLLM2135MB = "smollm2:135m"
    SMOLLM2360MB = "smollm2:360m"
    SMOLLM21p7B = "smollm2:1.7b"
    VICUNA7B = "vicuna:7b"
    VICUNA13B = "vicuna:13b"
    PHI33p8B = "phi3:3.8b"
    PHI314B = "phi3:14b"
    LLAMA3p21B = "llama3.2:1b"
    LLAMA3p23B = "llama3.2:3b"
    QWEN2p50p5B = "qwen2.5:0.5b"
    QWEN2p51p5B = "qwen2.5:1.5b"
    QWEN2p53B = "qwen2.5:3b"
    QWEN2p57B = "qwen2.5:7b"
    QWEN2p514B = "qwen2.5:14b"
    DEEPSEEKR11p5B = "deepseek-r1:1.5b"
    DEEPSEEKR17B = "deepseek-r1:7b"
    DEEPSEEKR18B = "deepseek-r1:8b"
    DEEPSEEKR114B = "deepseek-r1:14b"

    def __str__(self):
        return self.value



class OllamaInference:
    def predict(self, model: OllamaModel, prompt: str, options: Options) -> str:
        """
        Predicts the output of a model given a prompt.
        """
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

    def clear_models(self):
        """
        Clears all models from the local machine.
        """
        
        # Iterate through the OllamaModel enum and delete each model
        for model in OllamaModel:
            try:
                delete(model)
            except Exception as e:
                print(f"Error deleting model {model}: {e}")

