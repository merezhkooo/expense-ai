import os
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# ----- завантажуємо змінні з .env -----

from config import MONGO_URI as LOCAL_MONGO_URI, SECRET_KEY as LOCAL_SECRET_KEY

app = Flask(__name__)

# ---------- Налаштування Flask + MongoDB ----------
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", LOCAL_SECRET_KEY)
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", LOCAL_MONGO_URI)

mongo = PyMongo(app)
db = mongo.db  # db.users, db.expenses


# ---------- AI: завантаження моделі з Hugging Face Hub ----------
# Твоя модель у Hub:
MODEL_ID = "merezhkooo/expense-category-model"

# Завантажується 1 раз при старті сервера
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# якщо є GPU (рідко локально), переносимо модель
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(_device)

id2label = model.config.id2label


def classify_expense_description(text: str):
    """
    Повертає (label:str, score:float) або (None, None) якщо щось пішло не так.
    """
    text = (text or "").strip()
    if not text:
        return None, None

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)

        probs = torch.softmax(out.logits, dim=-1)[0]
        best_idx = int(torch.argmax(probs).item())
        label = id2label.get(best_idx) or str(best_idx)
        score = float(probs[best_idx].item())

        return label, score
    except Exception as e:
        print("AI classify error:", e)
        return None, None


# ----------------- Хелпери -----------------
def current_user():
    email = session.get("user_email")
    if not email:
        return None
    return db.users.find_one({"email": email})


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_email"):
            flash("Спочатку увійдіть у систему", "warning")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return wrapper


# ----------------- Маршрути -----------------
@app.route("/")
def index():
    user = current_user()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("Заповніть усі обов'язкові поля", "danger")
            return render_template("register.html")

        existing = db.users.find_one({"email": email})
        if existing:
            flash("Користувач з таким email уже існує", "danger")
            return render_template("register.html")

        db.users.insert_one({
            "name": name,
            "email": email,
            "password": password,  # для лаби ок, але краще хешувати
            "created_at": datetime.utcnow()
        })

        session["user_email"] = email
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if not email or not password:
        flash("Введіть email та пароль", "danger")
        return redirect(url_for("index"))

    user = db.users.find_one({"email": email, "password": password})
    if not user:
        flash("Невірний email або пароль", "danger")
        return redirect(url_for("index"))

    session["user_email"] = user["email"]
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.pop("user_email", None)
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    user = current_user()

    # -------- POST: додавання нової витрати --------
    if request.method == "POST":
        amount_str = request.form.get("amount", "").replace(",", ".")
        category = request.form.get("category", "").strip()
        description = request.form.get("description", "").strip()
        date_str = request.form.get("date", "").strip()

        if not amount_str or not category or not date_str:
            flash("Сума, категорія та дата є обов'язковими", "danger")
            return redirect(url_for("dashboard"))

        try:
            amount = float(amount_str)
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            flash("Невірний формат суми або дати", "danger")
            return redirect(url_for("dashboard"))

        db.expenses.insert_one({
            "user_email": user["email"],
            "amount": amount,
            "category": category,
            "description": description,
            "date": date_str,  # 'YYYY-MM-DD'
            "created_at": datetime.utcnow()
        })

        return redirect(url_for("dashboard"))

    # -------- GET: показ дашборду --------
    user_expenses = list(db.expenses.find({"user_email": user["email"]}))
    user_expenses.sort(key=lambda e: e.get("date", ""), reverse=True)

    total = sum(e.get("amount", 0) for e in user_expenses) if user_expenses else 0

    totals_by_category = {}
    for e in user_expenses:
        cat = e.get("category") or "Other"
        totals_by_category[cat] = totals_by_category.get(cat, 0) + float(e.get("amount", 0))

    for e in user_expenses:
        date_str = e.get("date", "")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            e["date_display"] = dt.strftime("%d.%m.%Y")
        except ValueError:
            e["date_display"] = date_str

    category_breakdown = []
    if total > 0:
        for cat, value in totals_by_category.items():
            percent = round(value / total * 100)
            category_breakdown.append({"name": cat, "value": round(value, 2), "percent": percent})
    else:
        for cat, value in totals_by_category.items():
            category_breakdown.append({"name": cat, "value": round(value, 2), "percent": 0})

    return render_template(
        "dashboard.html",
        user=user,
        expenses=user_expenses,
        total=total,
        totals_by_category=totals_by_category,
        category_breakdown=category_breakdown
    )


# ---------- AI route для модалки ----------
@app.route("/ai/suggest-category", methods=["POST"])
@login_required
def ai_suggest_category():
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return {"error": "empty_description"}, 400

    label, score = classify_expense_description(description)
    if not label:
        return {"error": "ai_failed"}, 502

    return {"category": label, "confidence": score}, 200

@app.route("/model-report")
def model_report():
    metrics = {
        "model": "distilbert-base-uncased (fine-tuned)",
        "dataset_size": 240,
        "train_size": 192,
        "test_size": 48,
        "best_accuracy": 0.8125,
        "best_f1_macro": 0.785517,
        "labels": ["Education","Entertainment","Food","Health","Home","Other","Shopping","Transport"]
    }

    # ТВОЇ значення з навчання (як у таблиці)
    history = [
        {"epoch": 1, "train_loss": None,     "val_loss": 1.918202, "accuracy": 0.645833, "f1_macro": 0.553208},
        {"epoch": 2, "train_loss": 1.971400, "val_loss": 1.656146, "accuracy": 0.812500, "f1_macro": 0.785517},
        {"epoch": 3, "train_loss": 1.971400, "val_loss": 1.470109, "accuracy": 0.812500, "f1_macro": 0.781445},
        {"epoch": 4, "train_loss": 1.531600, "val_loss": 1.387108, "accuracy": 0.812500, "f1_macro": 0.781445},
    ]

    # короткий текст для звіту
    summary = (
        "Модель досягла стабільної якості після другої епохи навчання. "
        "Найкраще значення точності (Accuracy) становить 0.8125, "
        "а значення F1 macro — 0.7855."
        "Значення функції втрат на валідаційній вибірці зменшувалося з кожною епохою, "
        "що свідчить про поступовий прогрес навчання моделі."
    )

    return render_template("model_report.html", metrics=metrics, history=history, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
