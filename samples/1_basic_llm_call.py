from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "complete the sentence"},
        {"role": "user", "content": "the students opened their"}
    ]
)

print(completion.choices[0].message.content)