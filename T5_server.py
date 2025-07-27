from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import torch
import psycopg2
import os

# Load environment variables
load_dotenv()

# DB Configuration
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'port': int(os.getenv("DB_PORT")),
    'sslmode': os.getenv("DB_SSLMODE", "require")
}

#  Setup Flask & Model
app = Flask(__name__)
CORS(app)

MODEL_PATH = "./t5-sql-finetuned"  # Local fine-tuned model folder
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

FIXED_SCHEMA = "employees (serial_no, empoyeid, empoyename, phone_no, email_address, department, position, date_of_joining, attendance, salary, performance_review)"

# SQL Query Executor
def execute_query(sql):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(sql)
        if sql.strip().lower().startswith("select"):
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            results = [dict(zip(columns, row)) for row in rows]
        else:
            conn.commit()
            results = {"status": "Query executed successfully"}
        cur.close()
        conn.close()
        return results
    except Exception as e:
        return {"error": str(e)}


# API Endpoint
@app.route("/generate_sql", methods=["POST"])
def generate_sql():
    data = request.json
    instruction = data.get("instruction", "").strip()

    if not instruction:
        return jsonify({"error": "Instruction is required"}), 400

    input_text = f"{instruction} <schema> {FIXED_SCHEMA}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True).split("schema>")[0].strip()

    result = execute_query(sql_query)

    if isinstance(result, list) and result:
        formatted = ""
        for row in result:
            formatted += ', '.join([f"{k}: {v}" for k, v in row.items()]) + "\n"
    elif isinstance(result, dict) and result.get("error"):
        formatted = f"❌ Error: {result['error']}"
    else:
        formatted = "✅ Query executed successfully but no data returned."

    return jsonify({"response": formatted.strip()})

# Run Server 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)