
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

output = Path("data") / "deep_learning_intro.pdf"
output.parent.mkdir(parents=True, exist_ok=True)

pdf = canvas.Canvas(str(output), pagesize=letter)

content = [
    "Introduction to Deep Learning",
    "",
    "Deep learning is a branch of machine learning.",
    "It uses neural networks with multiple layers.",
    "Neural networks learn patterns from training data.",
    "CNNs are commonly used for image classification.",
    "Transformers are widely used in natural language processing.",
    "Training adjusts model weights to reduce prediction error.",
]

y = 750

for line in content:
    pdf.drawString(72, y, line)
    y -= 28

pdf.save()

print(f"Created PDF: {output}")

