"""12 类文本语种分类：基础版 Transformer 训练示例。"""
from pathlib import Path
import argparse
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 每类 10 条示例。正式作业可以在这里增加训练样本。
DATA = {
    "zh": ["你好世界", "今天天气很好", "我喜欢学习", "这是一本书", "欢迎来到中国", "请问你叫什么名字", "明天我们去学校", "谢谢你的帮助", "我正在吃晚饭", "祝你周末愉快"],
    "en": ["Hello world", "How are you", "I like reading", "This is a book", "Welcome to our school", "What is your name", "See you tomorrow", "Thank you for helping", "I am eating dinner", "Have a nice weekend"],
    "ja": ["こんにちは世界", "今日はいい天気です", "私は勉強が好きです", "これは本です", "学校へようこそ", "お名前は何ですか", "また明日会いましょう", "ありがとうございます", "夕食を食べています", "良い週末を"],
    "ko": ["안녕하세요 세계", "오늘 날씨가 좋아요", "저는 공부를 좋아해요", "이것은 책입니다", "학교에 오신 것을 환영합니다", "이름이 무엇인가요", "내일 다시 만나요", "도와주셔서 감사합니다", "저녁을 먹고 있어요", "좋은 주말 보내세요"],
    "fr": ["Bonjour le monde", "Comment allez vous", "J'aime apprendre", "C'est un livre", "Bienvenue à notre école", "Comment vous appelez vous", "À demain", "Merci pour votre aide", "Je mange mon dîner", "Bon week end"],
    "de": ["Hallo Welt", "Wie geht es dir", "Ich lerne gern", "Das ist ein Buch", "Willkommen in der Schule", "Wie heißt du", "Bis morgen", "Vielen Dank für deine Hilfe", "Ich esse zu Abend", "Schönes Wochenende"],
    "es": ["Hola mundo", "Cómo estás", "Me gusta aprender", "Este es un libro", "Bienvenido a la escuela", "Cómo te llamas", "Hasta mañana", "Gracias por tu ayuda", "Estoy cenando", "Que tengas buen fin de semana"],
    "ru": ["Привет мир", "Как дела", "Я люблю учиться", "Это книга", "Добро пожаловать в школу", "Как тебя зовут", "Увидимся завтра", "Спасибо за помощь", "Я ужинаю", "Хороших выходных"],
    "ar": ["مرحبا بالعالم", "كيف حالك", "أحب التعلم", "هذا كتاب", "مرحبا بك في المدرسة", "ما اسمك", "أراك غدا", "شكرا لمساعدتك", "أنا أتناول العشاء", "عطلة نهاية أسبوع سعيدة"],
    "pt": ["Olá mundo", "Como você está", "Gosto de aprender", "Este é um livro", "Bem vindo à escola", "Qual é o seu nome", "Até amanhã", "Obrigado pela ajuda", "Estou jantando", "Tenha um bom fim de semana"],
    "hi": ["नमस्ते दुनिया", "आप कैसे हैं", "मुझे पढ़ना पसंद है", "यह एक किताब है", "स्कूल में आपका स्वागत है", "आपका नाम क्या है", "कल मिलते हैं", "मदद के लिए धन्यवाद", "मैं रात का खाना खा रहा हूँ", "आपका सप्ताहांत अच्छा हो"],
    "it": ["Ciao mondo", "Come stai", "Mi piace imparare", "Questo è un libro", "Benvenuto a scuola", "Come ti chiami", "Ci vediamo domani", "Grazie per il tuo aiuto", "Sto cenando", "Buon fine settimana"],
}

MAX_LEN = 48
MODEL_PATH = Path(__file__).resolve().parent / "language_model.pth"


def encode(text, vocab):
    """按字符编码，0 表示补齐，1 表示未见过的字符。"""
    ids = [vocab.get(char, 1) for char in text[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))


class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.position = nn.Embedding(MAX_LEN, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        features = self.embedding(x) + self.position(positions)
        padding = x.eq(0)
        features = self.transformer(features, src_key_padding_mask=padding)
        valid = (~padding).unsqueeze(-1)
        features = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.fc(features)


def main(epochs):
    random.seed(42)
    torch.manual_seed(42)

    languages = list(DATA.keys())
    train_samples, test_samples = [], []
    for label, language in enumerate(languages):
        samples = DATA[language].copy()
        random.shuffle(samples)
        train_samples += [(text, label) for text in samples[:8]]
        test_samples += [(text, label) for text in samples[8:]]

    # 从训练文本构建简单的字符词表。
    chars = sorted(set("".join(text for text, _ in train_samples)))
    vocab = {char: index + 2 for index, char in enumerate(chars)}

    def make_loader(samples, shuffle):
        texts = torch.tensor([encode(text, vocab) for text, _ in samples])
        labels = torch.tensor([label for _, label in samples])
        return DataLoader(TensorDataset(texts, labels), batch_size=16, shuffle=shuffle)

    train_loader = make_loader(train_samples, True)
    test_loader = make_loader(test_samples, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageTransformer(len(vocab) + 2, len(languages)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(texts), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts.to(device))
                correct += (outputs.argmax(dim=1).cpu() == labels).sum().item()
                total += len(labels)
        accuracy = correct / total
        print(f"第 {epoch + 1:02d} 轮 | Loss: {total_loss / len(train_loader):.4f} | 验证准确率: {accuracy:.1%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "model": model.state_dict(), "vocab": vocab, "languages": languages
            }, MODEL_PATH)

    print(f"训练完成，最佳验证准确率: {best_accuracy:.1%}")
    print(f"模型保存到: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练基础 Transformer 语种分类模型")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数，默认 20")
    args = parser.parse_args()
    main(args.epochs)
