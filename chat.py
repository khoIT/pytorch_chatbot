import random
import json
import os
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "iNexus"

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
import base64

client = OpenAI(api_key=OPENAI_API_KEY)

def find_industry_with_gpt(prompts):
    # Prepare the messages
    industries = {
        "Bạc": "Identify whether the conversation is discussing silver, which is a precious metal used in industries such as jewelry, electronics, and investment.",
        "Bạch Kim": "Check if the conversation is about platinum, a precious metal used in jewelry, automotive catalytic converters, and electronics.",
        "Bông - Sợi": "Look for any mention of cotton and fiber, typically related to textiles, fabric production, and agriculture.",
        "Cá da trơn": "Identify if the conversation is referring to catfish, an industry focused on fish farming and seafood production.",
        "Cá ngừ": "Determine whether the conversation mentions tuna, a key industry for seafood production and export.",
        "Cà phê": "Look for references to coffee, a major agricultural product related to the beverage industry and global trade.",
        "Cacao": "Check for mentions of cacao, a key agricultural product used in the production of chocolate and other foods.",
        "Cao su": "Identify if the conversation is about rubber, typically used in the manufacturing of tires and various industrial products.",
        "Chất dẻo": "Determine whether plastic materials or polymers are being discussed, related to the production of synthetic materials.",
        "Chè": "Check for mentions of tea, a widely consumed beverage and agricultural product.",
        "Da giày": "Look for references to the leather and footwear industry, including the manufacturing and export of shoes and leather goods.",
        "Đá quý": "Identify mentions of gemstones, an industry focused on the mining and sale of precious stones such as diamonds, rubies, and sapphires.",
        "Dầu - Hạt dầu": "Determine if oils and oilseeds, such as vegetable oils or soybean oil, are being discussed in relation to agriculture or food production.",
        "Dầu mỏ": "Look for references to petroleum, an industry related to the extraction, refining, and sale of oil products.",
        "Dệt may": "Check for any discussion related to textiles and garments, an industry that includes clothing manufacturing and fashion.",
        "Điện": "Identify mentions of electricity or power generation, including energy production and distribution.",
        "Đồng": "Determine if the conversation involves copper, used in electrical wiring, electronics, and construction.",
        "Đường": "Check for any discussion of sugar, an industry related to agriculture, food production, and export.",
        "Ethanol": "Look for references to ethanol, a biofuel made from crops like corn, used in the energy and automotive industries.",
        "Gạo": "Identify mentions of rice, a staple food product, central to agriculture and food export.",
        "Gia súc - Gia cầm": "Check if the conversation mentions livestock or poultry, related to farming, meat production, and agriculture.",
        "Giấy": "Determine if the conversation is about paper, including its production, use in packaging, and recycling.",
        "Gỗ": "Look for references to wood and lumber, used in construction, furniture, and manufacturing.",
        "Hạt điều": "Identify mentions of cashew nuts, an agricultural product important in the snack food industry and global trade.",
        "Hồ tiêu - Hạt tiêu": "Check for discussions around pepper (black or white), a spice and key agricultural export.",
        "Khí đốt": "Determine whether natural gas is being discussed, related to energy production and heating.",
        "Kim loại khác": "Look for references to other metals, such as aluminum, nickel, or zinc, typically used in manufacturing and industry.",
        "Mắc ca": "Identify mentions of macadamia nuts, a high-value agricultural product used in the food industry.",
        "Muối": "Check for discussions about salt, an essential mineral used in food production, preservation, and industrial processes.",
        "Ngũ cốc": "Look for references to grains, including wheat, corn, and oats, key in agriculture and food production.",
        "Nhựa - Hạt nhựa": "Determine if plastic resins or synthetic materials are being discussed, used in manufacturing and packaging.",
        "Palladium": "Identify mentions of palladium, a precious metal used in electronics, jewelry, and automotive catalytic converters.",
        "Phân bón": "Check for discussions about fertilizers, used in agriculture to improve crop yields.",
        "Rau - Củ - Quả": "Look for references to vegetables, fruits, and other produce, related to agriculture and food markets.",
        "Sắt thép": "Identify mentions of iron and steel, used in construction, manufacturing, and industrial applications.",
        "Sữa": "Check for discussions about milk and dairy products, including production, processing, and distribution.",
        "Than": "Determine if coal is being discussed, related to energy production, mining, and heavy industry.",
        "Thức ăn chăn nuôi": "Look for references to animal feed, used in livestock and poultry farming.",
        "Thủy hải sản khác": "Identify mentions of other seafood products, including shellfish and non-fish marine products.",
        "Tôm": "Check if shrimp is being discussed, a key product in seafood farming and global trade.",
        "Vàng": "Determine if gold is being referenced, a precious metal used in jewelry, investment, and industry.",
        "VLXD khác": "Look for mentions of other construction materials, including bricks, concrete, and other building products.",
        "Xăng dầu": "Identify discussions around gasoline and fuel, related to energy, automotive use, and oil refining.",
        "Xi măng - Clynker": "Check for mentions of cement and clinker, used in construction and infrastructure development."
    }

    messages = [{"role": "system",
                 "content": """You are given a Vietnamese sentence asking about an industry. Please use the dictionary above: {industries} to return the
                            closest matching key (industries) that are being mentioned. Then search the internet for the 5 most updated and recent reports about it
                            for Vietnam. Please include download links. If more than one keys are found, simply write Bạn muốn hỏi về ngành nào ? [keys found]""" % industries},
                {"role": "user", "content":
                    [{"type": "text",
                         "text": prompts
                    }]
                }]

    # Use GPT-4 model with a multimodal feature to perform OCR
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
    except Exception as e:
        print("Error when requesting to GPT: " % e)

    print("Result: ", response.choices[0])
    return response.choices[0].message.content


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() > 0.9:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return find_industry_with_gpt(msg)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

