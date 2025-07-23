from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="moonshotai/kimi-k2-instruct",
  messages=[{"role":"user","content":"can you please talk about basics in ros2 and how it is different from ros1?"}],
  temperature=0.6,
  top_p=0.9,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")