from openai import OpenAI

client = OpenAI(
    api_key="sk-pzaqzrvdukylanotdfbvszyelepbsdhggmacpzrdjghqavhr",
    base_url="https://api.siliconflow.cn/v1",
)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "为我编写一段文字，字数在50至100字，文字内含随机生成的身份证号（但是身份证号的格式必须符合标准格式，即身份证号码由18位数字组成，前17位是居民的出生日期、性别及顺序码，其中前六位是地址码，表示户籍所在地；接下来的8位是出生日期，格式为YYYYMMDD；第17位为性别识别码，奇数代表男性，偶数代表女性；最后一位是校验码，通过特定算法计算得出，以验证身份证号码的正确性。）要求模拟人类间交流的语气和方式。",
        }
    ],
    temperature=0.7,
    max_tokens=4096,
)

print(response.choices[0].message.content)
