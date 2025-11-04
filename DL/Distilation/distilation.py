import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import trange

def generate_synthetic_data(n_samples=2000):
    X = torch.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1] > 1).long()
    return X, y

X_train, y_train = generate_synthetic_data(1000)
X_test, y_test = generate_synthetic_data(300)

class TeacherNet(nn.Module):
    """Teacher - большая сеть"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class StudentNet(nn.Module):
    """Student - небольшая сеть"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

def distillation_loss(student_logits, teacher_logits, labels, T: float = 3.0, alpha: float = 0.5):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T * T)
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_teacher(model, X, y, epochs: int = 50, lr: float = 1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    pbar = trange(epochs, desc="Teacher Distillation")
    for epoch in pbar:
        model.train()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.write(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f}")

def train_student(student, teacher, X, y, epochs: int = 50, T: float = 3.0,
                  alpha: float = 0.5, lr: float = 1e-3):
    opt = optim.Adam(student.parameters(), lr=lr)
    pbar = trange(epochs, desc="Student Distillation")
    for epoch in pbar:
        student.train()
        with torch.no_grad():
            teacher_logits = teacher(X)
        student_logits = student(X)
        loss = distillation_loss(student_logits, teacher_logits, y, T, alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.write(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f}")

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
        acc = (preds == y).float().mean().item() * 100
    return acc

teacher = TeacherNet()
student = StudentNet()
student_baseline = StudentNet()

print("=== 🧠 Обучаем Teacher ===")
train_teacher(teacher, X_train, y_train, epochs=50)
teacher_acc = evaluate(teacher, X_test, y_test)
print(f"📊 Teacher Accuracy: {teacher_acc:.2f}%\n")

print("=== 👩‍🎓 Обучаем Student с дистилляцией ===")
train_student(student, teacher, X_train, y_train, epochs=50, T=3.0, alpha=0.5)
student_acc = evaluate(student, X_test, y_test)
print(f"📊 Student (Distilled) Accuracy: {student_acc:.2f}%\n")

print("=== ⚪ Обучаем Student без дистилляции ===")
train_teacher(student_baseline, X_train, y_train, epochs=50)
baseline_acc = evaluate(student_baseline, X_test, y_test)
print(f"📉 Student (Baseline) Accuracy: {baseline_acc:.2f}%\n")

print("=== ✅ Сравнение ===")
print(f"Teacher:            {teacher_acc:.2f}%")
print(f"Student (Distilled): {student_acc:.2f}%")
print(f"Student (Baseline):  {baseline_acc:.2f}%")
