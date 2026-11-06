
MODEL_ID = "google/gemma-4-E2B-it" # @param ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-12B-it", "google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it"]

from transformers import pipeline
import dialog
from transformers import GenerationConfig

txt_pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    device_map="auto",
    dtype="auto"
)

output = txt_pipe(text_inputs="<|turn>user\nRoses are..\n<|turn>model\n")
print(output[0]['generated_text'].removesuffix(""))

config = GenerationConfig.from_pretrained(MODEL_ID)
config.max_new_tokens = 512

conv = dialog.Conversation(
    dialog.User("Roses are...")
)
output = txt_pipe(text_inputs=conv.as_text(), return_full_text=False, generation_config=config)
conv += dialog.Model(output[0]['generated_text'].removesuffix(""))

print(conv.as_text())
conv.show()